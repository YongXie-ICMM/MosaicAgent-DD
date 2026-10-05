"""Reconcile one live scan checkpoint after storage recovery, without hardware I/O.

The caller owns the live-device, position, and human-confirmation gates. This
module never opens an old scan directory as a new process, moves the stage,
captures an image, or interprets controller counters as physical metrology.
"""
import hashlib
from pathlib import Path
import uuid


def _validate_promoted_image(checkpoint, image_decoder):
    """Verify the exact bytes retained by the live pre-promotion checkpoint."""
    path = Path(checkpoint['filepath'])
    expected_hash = checkpoint['image_sha256']
    expected_bytes = checkpoint['image_bytes']
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(c not in '0123456789abcdef' for c in expected_hash)):
        raise ValueError('Saved-image checkpoint has no valid SHA-256 identity.')
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError('Saved-image checkpoint has no valid byte count.')
    raw = path.read_bytes()
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Saved image changed after capture; automatic continuation is blocked.')
    if image_decoder is not None:
        if image_decoder(path) is not True:
            raise ValueError('Saved image could not be decoded; automatic continuation is blocked.')
        # A decoder may reread the file; catch changes during that verification.
        raw_after = path.read_bytes()
        if raw_after != raw:
            raise ValueError('Saved image changed during validation; continuation is blocked.')
    return path


def recover_scan_checkpoint(session, checkpoint, *, image_decoder=None):
    """Repair retained logs and reconcile an already-promoted image exactly once.

    Only an in-memory ``phase='promoted'`` checkpoint is eligible for reuse.
    Merely finding a PNG in a scan folder never proves that it was accepted.
    The stable reconciliation ID is retained before ``record`` so that a retry
    after any partial logging failure can find the original, now-durable event.
    A decoded image is required when the caller supplies ``image_decoder``.
    This function deliberately does not increment GUI counters or resume a scan.
    """
    promoted = bool(checkpoint and checkpoint.get('phase') == 'promoted')
    if promoted:
        path = _validate_promoted_image(checkpoint, image_decoder)
        for name in ('point', 'row', 'col', 'photo_count_before',
                     'human_count_before', 'sharpness_len_before'):
            value = checkpoint[name]
            if type(value) is not int or value < (1 if name == 'point' else 0):
                raise ValueError('Invalid live checkpoint field: ' + name)
        if type(checkpoint['human_confirmed']) is not bool:
            raise ValueError('Invalid live checkpoint human-confirmation field.')

    recovery = session.recover_history()
    if not promoted:
        return {'action': 'retry_current', 'history_recovery': recovery}

    recovery_point_id = checkpoint.setdefault('recovery_point_id', uuid.uuid4().hex)
    event = session.find_event('recovery_point_reconciled', recovery_point_id=recovery_point_id)
    if event is None:
        if checkpoint.get('reconciled'):
            raise ValueError('Checkpoint says reconciled but its audit event is missing.')
        event = session.record(
            'recovery_point_reconciled', recovery_point_id=recovery_point_id,
            filepath=str(path), point=checkpoint['point'], row=checkpoint['row'],
            col=checkpoint['col'], image_sha256=checkpoint['image_sha256'],
            image_bytes=checkpoint['image_bytes'], sharpness=checkpoint['sharpness'],
            retries=checkpoint['retries'], human_confirmed=checkpoint['human_confirmed'],
            action='reuse_saved_image',
            original_acceptance_basis='live_checkpoint_after_os_replace',
            physical_position_verified=False,
        )
    else:
        # A retained ID is not sufficient if its meaning conflicts with the
        # current checkpoint. Never authorize two images under one identity.
        expected = {
            'point': checkpoint['point'], 'row': checkpoint['row'],
            'col': checkpoint['col'], 'image_sha256': checkpoint['image_sha256'],
            'image_bytes': checkpoint['image_bytes'], 'action': 'reuse_saved_image',
        }
        if any(event.get(key) != value for key, value in expected.items()):
            raise ValueError('Reconciliation event conflicts with the live checkpoint.')
    checkpoint['reconciled'] = True
    return {
        'action': 'reuse_saved_image', 'point': checkpoint['point'],
        'row': checkpoint['row'], 'col': checkpoint['col'],
        'filepath': str(path), 'image_sha256': checkpoint['image_sha256'],
        'image_bytes': checkpoint['image_bytes'],
        'reconciliation_event_id': event['event_id'],
        'history_recovery': recovery,
    }

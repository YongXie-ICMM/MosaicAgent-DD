"""Read-only scan/log intake for the DD student workflow; never move source data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def inspect_transfer(path, repo):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('请选择包含原始照片及 session.json 的扫描文件夹。')
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.stitch_setup.service import _inspect
    # An existing packed scan day may contain several independent sessions.
    # Never silently select the largest sample or merge different specimens.
    if (root / 'handover.json').is_file():
        manifest_path = root / 'handover.json'
        if manifest_path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('交接清单过大，请老师核查。')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        runs = manifest.get('runs', [])
        if manifest.get('kind') != 'scan_handover' or not isinstance(runs, list):
            raise ValueError('交接清单格式不正确。')
        candidates = []
        for run in runs:
            relative = run.get('folder') if isinstance(run, dict) else None
            if not isinstance(relative, str):
                raise ValueError('交接清单缺少扫描文件夹。')
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError('交接文件夹越界。')
            candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError('这个交接包含多次扫描，请老师先核对并合并同一样品；学生不要自行混合批次。')
        root = candidates[0]
    checked = _inspect({'data_dir': str(root)})
    session_path = root / 'session.json'
    events_path = root / 'events.jsonl'
    if not session_path.is_file() or not events_path.is_file():
        raise ValueError('原始照片旁缺少 session.json 或 events.jsonl，请保留完整扫描记录再交接。')
    if session_path.stat().st_size > 2 * 1024 * 1024 or events_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('扫描记录超过当前读取限制，请老师核查。')
    session = json.loads(session_path.read_text(encoding='utf-8-sig'))
    if session.get('status') != 'completed':
        raise ValueError('扫描记录不是完成状态；请先检查中断原因和缺图，不要直接继续。')
    captures, failures = 0, []
    with events_path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('event') == 'capture_success':
                captures += 1
            if row.get('event') in {'terminal_failure', 'move_failed', 'scan_failed'}:
                failures.append(row.get('event'))
    files = []
    for name in ('session.json', 'events.jsonl', 'sharpness.csv', 'device_diagnostics.json'):
        item = root / name
        if item.is_file():
            value = hashlib.sha256()
            with item.open('rb') as stream:
                for part in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(part)
            files.append({'path': str(item), 'bytes': item.stat().st_size, 'sha256': value.hexdigest()})
    count = checked['inspection']['tile_count']
    if captures != count or session.get('photos_saved') != count:
        raise ValueError('照片张数与扫描日志不一致，请老师核查，未进入拼接。')
    return {'status': 'verified', 'path': str(root), 'session_id': session.get('session_id'),
            'sample_id': session.get('scan_config', {}).get('sample_id'),
            'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'photos': count, 'capture_success_records': captures, 'recorded_failures': failures,
            'input_fingerprint': checked['inspection']['fingerprint'], 'files': files,
            'source_images_modified': False, 'archives_created': False,
            'inspection': checked['inspection'], 'warnings': checked['warnings'],
            'note': '仅完成本轮原图与独立采集记录核验；通信日志可能跨轮共用，设备物理位移未独立验证。'}

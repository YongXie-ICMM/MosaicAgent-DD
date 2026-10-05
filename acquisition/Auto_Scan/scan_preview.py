"""Latest-frame preview reader; never touches Tk or moves the stage.

A camera driver may ignore its timeout and stay inside a native read. Stopping
the preview therefore does not join the worker: it invalidates its output and
lets that read finish. A replacement worker cannot start until the old worker
has exited, so reconnects cannot create competing camera readers.
"""
import threading


class PreviewPump:
    def __init__(self, camera_lock, interval_s=0.03):
        self._camera_lock = camera_lock
        self._interval_s = interval_s
        self._state_lock = threading.Lock()
        self._thread = None
        self._stop_event = None
        self._camera = None
        self._generation = 0
        self._mailbox = None

    def start(self, camera):
        """Start if idle; return False while a stopped native read drains.

        Repeated calls for the active camera are idempotent. Callers may retry
        this method on a GUI timer, but must not wait for a driver here.
        """
        if camera is None:
            self.stop()
            return False
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._camera is camera and not self._stop_event.is_set():
                    return True
                # A new camera invalidates the old result immediately, but the
                # old worker must finish before its replacement is launched.
                self._stop_event.set()
                self._generation += 1
                self._mailbox = None
                return False
            # Let the UI consume a failure before it decides whether to retry.
            if (self._camera is camera and self._mailbox is not None
                    and self._mailbox[0] == 'error'):
                return False
            self._generation += 1
            generation = self._generation
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._camera = camera
            self._mailbox = None
            self._thread = threading.Thread(
                target=self._read_frames, args=(camera, generation, stop_event),
                name='scan-live-preview', daemon=True)
            self._thread.start()
            return True

    def stop(self):
        """Invalidate queued and in-flight frames without waiting for a read."""
        with self._state_lock:
            self._generation += 1
            if self._stop_event is not None:
                self._stop_event.set()
            self._mailbox = None

    def poll(self):
        """Take the newest frame/error once, or None; never blocks on camera."""
        with self._state_lock:
            item = self._mailbox
            self._mailbox = None
            return item

    def _publish(self, item, generation, stop_event):
        with self._state_lock:
            if self._generation == generation and not stop_event.is_set():
                self._mailbox = item

    def _read_frames(self, camera, generation, stop_event):
        while not stop_event.is_set():
            # Acquisition owns this same lock. Preview must skip a busy camera,
            # not queue behind a capture or hold up the GUI thread.
            if self._camera_lock.acquire(blocking=False):
                try:
                    if stop_event.is_set():
                        return
                    frame = camera.grab_frame(timeout=0.5, flush=0)
                except Exception as exc:
                    self._publish(('error', str(exc)), generation, stop_event)
                    return
                finally:
                    self._camera_lock.release()
                if frame is not None:
                    self._publish(('frame', frame), generation, stop_event)
            stop_event.wait(self._interval_s)

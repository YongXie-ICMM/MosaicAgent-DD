"""Student entry: check dependencies and retain the complete console transcript."""
from datetime import datetime
from pathlib import Path
import importlib.util
import os
import platform
import subprocess
import sys


LOG_FAILURE_MESSAGE = (
    '控制台日志写入失败，采集程序可能仍运行；先在 GUI 停止/断开并正常关闭，'
    '不要开第二实例。已有日志请保留。'
)


def _console_print(message, *, end='\n', error=False):
    """A broken console must not abandon the instrument process either."""
    try:
        print(message, end=end, file=sys.stderr if error else sys.stdout, flush=True)
    except (OSError, ValueError):
        pass


def _wait_for_exit(process):
    # Do not terminate or restart a child which may still control the stage.
    warned = False
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            if not warned:
                _console_print('仍在等待采集程序退出；请在 GUI 停止/断开并正常关闭，不要开第二实例。',
                               error=True)
                warned = True


def _relay_console(process, transcript):
    """Keep draining and supervising the child after a transcript write fails."""
    log_failed = False

    def warn_log_failure(exc):
        nonlocal log_failed
        if not log_failed:
            _console_print(LOG_FAILURE_MESSAGE + '\n原因：' + str(exc), error=True)
        log_failed = True

    try:
        try:
            for line in process.stdout:
                _console_print(line, end='')
                if not log_failed:
                    try:
                        transcript.write(line)
                        transcript.flush()
                    except (OSError, ValueError) as exc:
                        warn_log_failure(exc)
        except KeyboardInterrupt:
            _console_print('已中断控制台读取；采集程序可能仍运行。请在 GUI 停止/断开并正常关闭，不要开第二实例。',
                           error=True)
        except (OSError, ValueError) as exc:
            _console_print('控制台读取失败，采集程序可能仍运行；请在 GUI 停止/断开并正常关闭，不要开第二实例。'
                           '\n原因：' + str(exc), error=True)
        return _wait_for_exit(process)
    finally:
        try:
            transcript.close()
        except (OSError, ValueError) as exc:
            warn_log_failure(exc)


def main():
    folder = Path(__file__).resolve().parent
    missing = [m for m in ('numpy', 'cv2', 'tkinter') if importlib.util.find_spec(m) is None]
    if missing:
        print('Missing dependencies: ' + ', '.join(missing))
        print('Run 01_setup.bat once. tkinter requires a Python installation with Tcl/Tk.')
        return 1
    if platform.system() != 'Windows' and '--sim' not in sys.argv:
        print('Real XY stage control requires Windows and the XIMC drivers.')
        print('No hardware was opened. --sim simulates the STAGE only; camera can still be real.')
        return 1
    logs = folder / 'history' / 'console_logs'
    path = logs / ('launch_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.txt')
    transcript = None
    try:
        logs.mkdir(parents=True, exist_ok=True)
        transcript = path.open('x', encoding='utf-8')
        transcript.write('Console history started: ' + datetime.now().isoformat() + '\n')
        transcript.flush()
    except (OSError, ValueError) as exc:
        if transcript is not None:
            try:
                transcript.close()
            except (OSError, ValueError):
                pass
        _console_print('无法建立控制台日志，采集程序尚未启动。请检查磁盘空间和文件夹写入权限后重试。'
                       '\n目录：' + str(logs) + '\n原因：' + str(exc), error=True)
        return 1
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-u', str(folder / '03Auto_Snake_Scan_Camera_v3.py'), *sys.argv[1:]]
    _console_print('Console history: ' + str(path))
    try:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    except OSError as exc:
        try:
            transcript.close()
        except (OSError, ValueError):
            pass
        _console_print('采集程序未能启动：' + str(exc), error=True)
        return 1
    return _relay_console(process, transcript)


if __name__ == '__main__':
    raise SystemExit(main())

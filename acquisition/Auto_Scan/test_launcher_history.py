"""Launcher supervision tests using fake streams/processes; no hardware opens."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import launch_scan


class FailingTranscript:
    def __init__(self, failure):
        self.failure = failure
        self.writes = []
        self.flushes = 0
        self.closed = False

    def write(self, line):
        self.writes.append(line)
        if self.failure == 'write':
            raise OSError('disk full')

    def flush(self):
        self.flushes += 1
        if self.failure == 'flush':
            raise OSError('flush failed')

    def close(self):
        self.closed = True
        if self.failure in ('write', 'flush', 'close'):
            raise OSError('close flush failed')


class LauncherHistoryTests(unittest.TestCase):
    def relay(self, transcript, lines=None, wait_results=None):
        process = Mock()
        delivered = []

        def output():
            for item in lines or ['first\n', 'second\n', 'third\n']:
                if isinstance(item, BaseException):
                    raise item
                delivered.append(item)
                yield item

        process.stdout = output()
        if wait_results is None:
            process.wait.return_value = 7
        else:
            process.wait.side_effect = wait_results
        visible, warnings = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(visible), contextlib.redirect_stderr(warnings):
            result = launch_scan._relay_console(process, transcript)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        return result, delivered, visible.getvalue(), warnings.getvalue(), process

    def test_write_or_flush_failure_drains_output_and_waits_without_repeated_warning(self):
        for failure in ('write', 'flush'):
            with self.subTest(failure=failure):
                transcript = FailingTranscript(failure)
                result, delivered, visible, warning, process = self.relay(transcript)
                self.assertEqual(result, 7)
                self.assertEqual(delivered, ['first\n', 'second\n', 'third\n'])
                self.assertEqual(visible, ''.join(delivered))
                self.assertEqual(len(transcript.writes), 1, 'stop writing the failed transcript')
                self.assertTrue(transcript.closed)
                process.wait.assert_called_once_with()
                self.assertEqual(warning.count(launch_scan.LOG_FAILURE_MESSAGE), 1)
                self.assertIn('不要开第二实例', warning)

    def test_close_failure_keeps_child_exit_code(self):
        result, delivered, _, warning, process = self.relay(FailingTranscript('close'))
        self.assertEqual(result, 7)
        self.assertEqual(len(delivered), 3)
        process.wait.assert_called_once_with()
        self.assertEqual(warning.count(launch_scan.LOG_FAILURE_MESSAGE), 1)

    def test_stdout_read_failure_still_waits_for_child_and_warns(self):
        transcript = FailingTranscript(None)
        result, delivered, _, warning, process = self.relay(
            transcript, ['first\n', OSError('pipe read error')])
        self.assertEqual(result, 7)
        self.assertEqual(delivered, ['first\n'])
        self.assertIn('控制台读取失败', warning)
        self.assertIn('不要开第二实例', warning)
        process.wait.assert_called_once_with()
        self.assertTrue(transcript.closed)

    def test_ctrl_c_does_not_abandon_or_kill_child(self):
        result, _, _, warning, process = self.relay(
            FailingTranscript(None), [KeyboardInterrupt()], [KeyboardInterrupt(), 0])
        self.assertEqual(result, 0)
        self.assertEqual(process.wait.call_count, 2)
        self.assertIn('采集程序可能仍运行', warning)

    def test_logging_initialization_failures_never_spawn_child(self):
        for failure in ('mkdir', 'open', 'write', 'flush'):
            with self.subTest(failure=failure):
                transcript = FailingTranscript(failure)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(launch_scan.importlib.util, 'find_spec', return_value=object()))
                    stack.enter_context(patch.object(launch_scan.platform, 'system', return_value='Windows'))
                    mkdir = stack.enter_context(patch.object(Path, 'mkdir'))
                    open_file = stack.enter_context(patch.object(Path, 'open', return_value=transcript))
                    spawn = stack.enter_context(patch.object(launch_scan.subprocess, 'Popen'))
                    if failure == 'mkdir':
                        mkdir.side_effect = OSError('no space for directory')
                    if failure == 'open':
                        open_file.side_effect = PermissionError('not writable')
                    warning = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    self.assertEqual(launch_scan.main(), 1)
                    spawn.assert_not_called()
                    self.assertIn('采集程序尚未启动', warning.getvalue())


if __name__ == '__main__':
    unittest.main()

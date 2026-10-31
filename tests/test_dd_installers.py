"""Execute installer branches with fake interpreters; never pip/venv/network.

Windows source invariants supplement executable POSIX branch tests because
cmd.exe is unavailable on macOS/Linux. Instrument acceptance is separate.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PINNED = 'requirements-student-py39.txt'

BASE_STUB = '''#!/bin/sh
printf '%s:%s\\n' "${0##*/}" "$*" >> "$INSTALL_CALLS"
case "$*" in
  'tools/dd_trial.py --check-python') exit "${BASE_CHECK_RC:-0}" ;;
  '-m venv .venv')
    if [ "${CREATE_RC:-0}" -ne 0 ]; then exit "$CREATE_RC"; fi
    mkdir -p .venv/bin
    cp "$VENV_STUB" .venv/bin/python
    chmod +x .venv/bin/python
    exit 0 ;;
  *) echo 'Unexpected base-interpreter invocation' >&2; exit 99 ;;
esac
'''
VENV_STUB = '''#!/bin/sh
printf 'venv:%s\\n' "$*" >> "$INSTALL_CALLS"
case "$*" in
  'tools/dd_trial.py --check-python') exit "${VENV_CHECK_RC:-0}" ;;
  '-m pip install -r requirements-analysis.txt'|'-m pip install -r requirements-student-py39.txt') exit "${PIP_RC:-0}" ;;
  'tools/dd_trial.py --check-environment') exit "${ENV_CHECK_RC:-0}" ;;
  *) echo 'Unexpected venv invocation' >&2; exit 99 ;;
esac
'''


def fixture(tmp_path, existing=False, trial=True, have39=True):
    kit = tmp_path / 'student kit with spaces'
    kit.mkdir()
    shutil.copy2(ROOT / '01_install.command', kit / '01_install.command')
    (kit / 'requirements-analysis.txt').write_text('# fixture only\n')
    (kit / PINNED).write_text('# pinned fixture only\n')
    if trial:
        (kit / 'trial_kit.json').write_text('{"kind":"dd_student_trial"}\n')
    (kit / 'tools').mkdir()
    (kit / 'tools/dd_trial.py').write_text('# Never executed by real Python\n')
    bin_path = tmp_path / 'fake-bin'
    bin_path.mkdir()
    for name in ['python3'] + (['python3.9'] if have39 else []):
        (bin_path / name).write_text(BASE_STUB)
        (bin_path / name).chmod(0o755)
    # Isolate PATH: an actual Python3.9 installed on the host must never run.
    for name in ['dirname', 'mkdir', 'cp', 'chmod']:
        (bin_path / name).symlink_to(shutil.which(name))
    venv_stub = tmp_path / 'fake-venv-python'
    venv_stub.write_text(VENV_STUB)
    venv_stub.chmod(0o755)
    if existing:
        (kit / '.venv/bin').mkdir(parents=True)
        shutil.copy2(venv_stub, kit / '.venv/bin/python')
        (kit / '.venv/keep.txt').write_text('preserve existing Python3.9 environment')
    calls = tmp_path / 'calls.txt'
    env = dict(os.environ, PATH=str(bin_path), INSTALL_CALLS=str(calls), VENV_STUB=str(venv_stub))
    return kit, env, calls


def run(kit, env):
    return subprocess.run(['/bin/sh', str(kit / '01_install.command')], cwd=kit.parent,
                          env=env, input='\n', text=True, capture_output=True, timeout=10)


def test_fresh_trial_prefers_python39_and_pins_all_dependencies_in_one_venv(tmp_path):
    kit, env, calls = fixture(tmp_path)
    result = run(kit, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.read_text().splitlines() == [
        'python3.9:tools/dd_trial.py --check-python', 'python3.9:-m venv .venv',
        'venv:tools/dd_trial.py --check-python', f'venv:-m pip install -r {PINNED}',
        'venv:tools/dd_trial.py --check-environment']
    assert '03_open_workbench.command' in result.stdout
    assert 'Optional: 02_run_layer_demo.command' in result.stdout
    assert 'acquisition requires Windows' in result.stdout


def test_existing_python39_environment_is_preserved_and_used_without_system_python(tmp_path):
    kit, env, calls = fixture(tmp_path, existing=True)
    result = run(kit, env)
    assert result.returncode == 0
    assert all(row.startswith('venv:') for row in calls.read_text().splitlines())
    assert (kit / '.venv/keep.txt').read_text() == 'preserve existing Python3.9 environment'


def test_python3_fallback_is_used_only_when_python39_is_unavailable(tmp_path):
    kit, env, calls = fixture(tmp_path, have39=False)
    result = run(kit, env)
    assert result.returncode == 0
    assert calls.read_text().splitlines()[0] == 'python3:tools/dd_trial.py --check-python'


def test_source_installation_without_trial_marker_retains_analysis_requirements(tmp_path):
    kit, env, calls = fixture(tmp_path, trial=False)
    result = run(kit, env)
    assert result.returncode == 0
    assert 'venv:-m pip install -r requirements-analysis.txt' in calls.read_text().splitlines()


def test_trial_missing_pinned_requirements_fails_before_interpreter_or_pip(tmp_path):
    kit, env, calls = fixture(tmp_path)
    (kit / PINNED).unlink()
    result = run(kit, env)
    assert result.returncode == 1
    assert not calls.exists()
    assert not (kit / '.venv').exists()
    assert PINNED in result.stdout
    assert 'Ready.' not in result.stdout


@pytest.mark.parametrize('variable,existing,last_call', [
    ('BASE_CHECK_RC', False, 'python3.9:tools/dd_trial.py --check-python'),
    ('CREATE_RC', False, 'python3.9:-m venv .venv'),
    ('VENV_CHECK_RC', True, 'venv:tools/dd_trial.py --check-python'),
    ('PIP_RC', True, f'venv:-m pip install -r {PINNED}'),
    ('ENV_CHECK_RC', True, 'venv:tools/dd_trial.py --check-environment'),
])
def test_failed_check_stops_without_ready_message_or_global_changes(tmp_path, variable, existing, last_call):
    kit, env, calls = fixture(tmp_path, existing=existing)
    env[variable] = '1'
    result = run(kit, env)
    assert result.returncode == 1
    assert calls.read_text().splitlines()[-1] == last_call
    assert 'Ready.' not in result.stdout
    if variable == 'BASE_CHECK_RC':
        assert not (kit / '.venv').exists()
    if existing:
        assert (kit / '.venv/keep.txt').is_file()


def test_windows_installer_prefers39_preserves_venv_and_never_changes_global_pip():
    raw = (ROOT / '01_install.bat').read_bytes()
    assert b'\r\n' in raw
    text = raw.decode('utf-8').replace('\r\n', '\n')
    assert text.index('py -3.9 --version') < text.index('py -3.9 "tools\\dd_trial.py" --check-python')
    assert 'py -3.9 "tools\\dd_trial.py" --check-python\nif errorlevel 1 goto python_failed\npy -3.9 -m venv ".venv"' in text
    assert 'python "tools\\dd_trial.py" --check-python\nif errorlevel 1 goto python_failed\npython -m venv ".venv"' in text
    assert text.index('if exist ".venv\\Scripts\\python.exe" goto check_environment') < text.index('py -3.9 --version')
    assert 'if not exist "trial_kit.json" goto select_python' in text
    assert 'if not exist "requirements-student-py39.txt" goto requirements_failed' in text
    assert 'set "DD_REQUIREMENTS=requirements-student-py39.txt"' in text
    section = text.split(':check_environment\n', 1)[1].split(':python_failed\n', 1)[0]
    assert section.index('--check-python') < section.index('-m pip install') < section.index('--check-environment')
    assert section.count('".venv\\Scripts\\python.exe"') == 4
    assert 'py -3 -m pip' not in text and '\npython -m pip' not in text
    assert '-m pip install -r "%DD_REQUIREMENTS%"' in text
    assert 'Run 03_open_workbench.bat' in text
    assert 'Python 3.9.18 is supported; no upgrade is required' in text
    assert '3.10' not in text


def test_posix_shell_syntax_and_no_forced_python_upgrade():
    subprocess.run(['/bin/sh', '-n', str(ROOT / '01_install.command')], check=True)
    text = (ROOT / '01_install.command').read_text()
    assert '3.10' not in text and '--upgrade' not in text

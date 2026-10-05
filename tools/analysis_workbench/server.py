#!/usr/bin/env python3
"""Local entry for existing acquisition and review tools; no model/hardware imports."""
from __future__ import annotations
import argparse
import errno
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit, quote
import urllib.request
import webbrowser
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
APP = 'mosaic-analysis-workbench'
MOSAIC_URL = 'https://github.com/YongXie-ICMM/MosaicAgent-DD'
SCAN_URL = MOSAIC_URL
IMAGE_RUN = '06_analysis/runs/20260924_0409_inference_examples'
MAX_JSON = 8 * 1024 * 1024
INSPECT_TTL_S = 30.0      # git subprocesses per scan_repo
HISTORY_TTL_S = 20.0      # full history re-parse while the scanner keeps appending


def words(zh, en):
    return {'zh': zh, 'en': en}


class WorkbenchError(ValueError):
    """A user-facing failure with a message in both interface languages."""

    def __init__(self, zh, en):
        super().__init__(en)
        self.message = words(zh, en)


def read_json(path, limit=MAX_JSON):
    if path.stat().st_size > limit:
        raise ValueError('JSON is too large to inspect safely')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + secrets.token_hex(5) + '.tmp')
    try:
        with temp.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class Workbench:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.runtime = self.config_path.parent / 'workbench_history'
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.artifacts = {}
        self.errors = []
        self._scanner_cache = {'repo': None, 'at': 0.0, 'info': None}
        self._history_cache = {'repo': None, 'signature': None, 'at': 0.0, 'rows': None}
        bundled_scan = REPO / 'acquisition'
        scan = bundled_scan if (bundled_scan / 'Auto_Scan/launch_scan.py').is_file() else REPO.parent / 'AmScope-Camera'
        self.config = {'project_root': '', 'scan_repo': str(scan), 'inference_results': ''}
        if self.config_path.exists():
            saved = read_json(self.config_path)
            self.config.update({k: str(saved[k]) for k in self.config if k in saved})
        else:
            atomic_json(self.config_path, self.config)
        # The 'opened' event is written by main() once this process actually serves the port.

    @property
    def project(self):
        value = self.config.get('project_root')
        return Path(value).expanduser().resolve() if value else None

    def relocate(self, value):
        """Resolve a recorded path. Manifests written on another computer carry absolute
        paths; when such a path does not exist here, retry the tail after the project
        folder's name under the configured project root. Returns a Path or None."""
        if not isinstance(value, (str, Path)) or not str(value):
            return None
        path = Path(str(value)).expanduser()
        if path.exists():
            return path
        root = self.project
        if root is None or not root.name:
            return None
        # Accept recorded Windows separators even while inspecting on macOS/Linux.
        parts = str(value).replace('\\', '/').split('/')
        if root.name not in parts:
            return None
        index = len(parts) - 1 - parts[::-1].index(root.name)
        candidate = root.joinpath(*parts[index + 1:])
        return candidate if candidate.exists() else None

    @property
    def inference_results(self):
        value = self.config.get('inference_results')
        return Path(value).expanduser().resolve() if value else (self.project / IMAGE_RUN if self.project else None)

    def roots(self):
        return {k: p for k, p in {'project': self.project, 'inference': self.inference_results, 'logs': self.runtime}.items() if p and p.is_dir()}

    def event(self, tool, kind, message, **extra):
        record = {'at': datetime.now(timezone.utc).isoformat(), 'tool': tool, 'kind': kind,
                  'message': message, 'artifact_ids': [], **extra}
        # Do not pretend that an unrecorded launch succeeded.
        with self.lock, (self.runtime / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        return record

    def events(self):
        path = self.runtime / 'events.jsonl'
        if not path.exists():
            return []
        with path.open('rb') as f:
            f.seek(0, 2)
            offset = max(0, f.tell() - 131072)
            f.seek(offset)
            if offset:
                f.readline()
            lines = f.read().decode('utf-8', errors='replace').splitlines()
        result = []
        for line in lines[-80:]:
            try:
                result.append(json.loads(line))
            except ValueError:
                result.append({'at': '', 'tool': 'workbench', 'kind': 'warning', 'message':
                               words('有一条历史记录无法读取，请保留历史文件。', 'An event could not be read; retain the history file.')})
        return result

    def register(self, path, tool, role, label, case_id='', expected_hash=None, member=None, *, case_label=None, image_size=None, source_run=None):
        path = self.relocate(path)
        if path is None:
            return None
        path = path.resolve()
        if not path.is_file() or not any(inside(path, root) for root in self.roots().values()):
            return None
        aid = hashlib.sha256((str(path) + (member or '') + tool + role).encode()).hexdigest()[:24]
        item = {'id': aid, 'tool': tool, 'role': role, 'label': label, 'case_id': case_id,
                'url': '/api/artifact/' + aid, 'path': str(path) + (' :: ' + member if member else ''),
                'mime': mimetypes.guess_type(member or str(path))[0] or 'application/octet-stream'}
        if case_label is not None:
            item['case_label'] = case_label
        if image_size is not None:
            item['image_size'] = image_size
        if source_run is not None:
            item['source_run'] = source_run
        self.artifacts[aid] = {'public': item, 'path': path, 'sha256': expected_hash, 'member': member}
        return item

    def collect(self):
        self.artifacts = {}
        self.errors = []
        self.inference_details = []
        launch_logs = self.runtime / 'scan_launches'
        if launch_logs.is_dir():
            for log in sorted(launch_logs.glob('*.log'))[-5:]:
                self.register(log, 'scan', 'data', words('扫描启动日志 ' + log.name, 'Scanner launch log ' + log.name))
        root = self.project
        if root and root.is_dir():
            for rel, label in [('06_analysis/Sample_growth_ledgers_20261004/Spectroscopy_Growth_Records_EN.xlsx', words('光谱样品与生长记录', 'Spectroscopy samples and growth records')),
                               ('06_analysis/runs/20261002_minimal_supervision/candidate_manifest.json', words('光谱与图像核对清单', 'Spectral and image review inventory')),
                               ('02_data/Growth_records/README.md', words('原始生长记录索引', 'Original growth record index'))]:
                self.register(root / rel, 'spectra', 'data', label)
            # Add evidence, never infer image/coverage verification from acquisition counts.
            for rel, label in [('06_analysis/runs/20260929_scan_v3_acceptance/00_先看结果.html', words('9月29日扫描验收记录', 'September 29 acquisition review')),
                               ('02_data/DD_MAPPING_20260930_v3/README.md', words('954点返还数据说明', '954-position returned-data record'))]:
                self.register(root / rel, 'scan', 'report', label)
            # These are archived statistical figures, not fresh predictions.
            self.register(root / '06_analysis/Figure3_current/fig3_recount_overview.png', 'images', 'report',
                          words('Figure 3 已存档统计概览', 'Archived Figure 3 statistics overview'))
            self.register(root / '06_analysis/Figure3_current/fig3_recount.json', 'images', 'data',
                          words('Figure 3 计数区域与分母记录', 'Figure 3 counting support and denominators'))
            self.register(root / '04_manuscript/TwistCVD_Paper/Spectra/dr_spectra_figure.png', 'spectra', 'report',
                          words('已存档光谱图（非新质检结果）', 'Archived spectra figure (not a new QC result)'))
        run = self.inference_results
        manifest_path = run / 'results/run_manifest.json' if run else None
        if manifest_path and manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
                source_run = manifest.get('source_run') or manifest.get('run_id') or run.name
                if not isinstance(source_run, str):
                    source_run = run.name
                source_run = source_run[:300]
                self.inference_details.append(words('当前识别运行：' + source_run, 'Current inference run: ' + source_run))
                weight = manifest.get('weights') or manifest.get('checkpoint')
                if isinstance(weight, dict) and isinstance(weight.get('path'), str):
                    weight_name = weight['path'].replace('\\', '/').rsplit('/', 1)[-1]
                    digest = weight.get('sha256')
                    fingerprint = ' · SHA-256 ' + digest[:12] if isinstance(digest, str) and len(digest) == 64 else ''
                    self.inference_details.append(words('清单记录的模型：' + weight_name + fingerprint, 'Model recorded in manifest: ' + weight_name + fingerprint))
                # This flag belongs to the saved review record. It is not a newly
                # computed image-quality verdict or evidence of model failure.
                if manifest.get('review_required') is True:
                    note = manifest.get('review_note')
                    note = note.strip()[:3000] if isinstance(note, str) else ''
                    self.errors.append(words(
                        '本次运行被标记为需要人工复核；这不是工作台自动判错。' + (' 清单备注：' + note if note else ''),
                        'This run is flagged for human review; this is not an automated error verdict.' + (' Manifest note: ' + note if note else '')))
                # Acquisition colour-balance check recorded by the demo (2026-10-05):
                # a saved comparison with the reference substrate colour, not a new verdict.
                colour = manifest.get('colour_check')
                if isinstance(colour, dict) and colour.get('verdict') not in (None, 'within_tolerance', 'no_reference', 'no_images'):
                    gains = [g for g in (colour.get('gains_rgb') or []) if isinstance(g, list)]
                    shown = '; '.join(', '.join(f'{float(v):.3f}' for v in g[:3]) for g in gains[:4] if len(g) == 3)
                    self.errors.append(words(
                        '清单记录的颜色检查：原图颜色/白平衡与参考衬底颜色不一致' + ('（校正到参考的每通道增益 ' + shown + '）' if shown else '') + '。预测需复核；新扫描前先在采集端校正白平衡。',
                        'Recorded colour check: the images\' colour balance differs from the reference substrate colour'
                        + (' (per-channel gain to reference ' + shown + ')' if shown else '') + '. Predictions need review; correct the camera white balance before new scans.'))
                recorded_sizes = set()
                for sample in manifest.get('samples', [])[:40]:
                    cid = sample['sample_id']
                    size = sample.get('source_image_size', sample.get('input_size_wh'))
                    if not (isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size)):
                        size = None
                    if size:
                        recorded_sizes.add(tuple(size))
                    label = cid + (f' [{size[0]} × {size[1]}]' if size else '')
                    metadata = {'case_label': words(label, label), 'image_size': size, 'source_run': source_run}
                    # Both review steps use the same saved inference evidence. No
                    # additional program or model is started by either action.
                    for tool in ('images', 'layers'):
                        self.register(sample['archive'], tool, 'original', words(label + ' 原图', label + ' original'), cid,
                                      sample.get('member_sha256'), sample['member'], **metadata)
                        for filename, role in [('mask_color.png', 'prediction'), ('overlay_full.jpg', 'overlay')]:
                            self.register(run / 'results' / cid / filename, tool, role,
                                          words(label + (' 层数预测' if role == 'prediction' else ' 叠加'), label + ' ' + role), cid,
                                          sample.get('outputs', {}).get(filename, {}).get('sha256'), **metadata)
                if recorded_sizes:
                    dimensions = ', '.join(f'{w} × {h}' for w, h in sorted(recorded_sizes))
                    self.inference_details.append(words('清单记录的原图尺寸：' + dimensions + ' 像素', 'Source image sizes recorded in manifest: ' + dimensions + ' pixels'))
                for tool in ('images', 'layers'):
                    self.register(manifest_path, tool, 'data', words('识别参数与来源记录', 'Inference parameters and provenance'))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.errors.append(words('层数识别清单读取失败：' + str(exc), 'Layer inference manifest could not be read: ' + str(exc)))
        if root and root.is_dir():
            self.register(root / '06_analysis/Figure3_current/fig3_recount_overview.png', 'layers', 'report',
                          words('Figure 3 已存档层数统计', 'Archived Figure 3 layer-number statistics'))
            self.register(root / '06_analysis/Figure3_current/fig3_recount.json', 'layers', 'data',
                          words('层数计数区域、分子与分母', 'Layer counting support, numerators and denominators'))
        return [v['public'] for v in self.artifacts.values()]

    def _inspect_scanner(self, repo):
        """inspect_scanner runs two git subprocesses; reuse its answer for INSPECT_TTL_S."""
        from scan_bridge import inspect_scanner
        cache = self._scanner_cache
        now = time.monotonic()
        if cache['info'] is not None and cache['repo'] == repo and now - cache['at'] < INSPECT_TTL_S:
            return cache['info']
        info = inspect_scanner(Path(repo).expanduser())
        cache.update(repo=repo, at=now, info=info)
        return info

    def _scanner_history(self, repo):
        """Re-parse scanner histories only when their files changed, and at most once per
        HISTORY_TTL_S while a scan keeps appending. A 954-point run is a 24.5 MB snapshot."""
        from scan_bridge import history_signature, scanner_history
        cache = self._history_cache
        now = time.monotonic()
        signature = history_signature(Path(repo).expanduser())
        if cache['rows'] is not None and cache['repo'] == repo and (
                signature == cache['signature'] or now - cache['at'] < HISTORY_TTL_S):
            return list(cache['rows'])
        rows = scanner_history(Path(repo).expanduser())
        cache.update(repo=repo, signature=signature, at=now, rows=list(rows))
        return rows

    def state(self):
        from scan_bridge import last_launch
        with self.lock:
            artifacts = self.collect()
            info = self._inspect_scanner(self.config['scan_repo'])
            available = {key: any(a['tool'] == key for a in artifacts) for key in ['images', 'layers', 'spectra']}
            scan_note = words('在 Windows 仪器电脑上启动；本机可查看代码与说明。', 'Launch on the Windows instrument computer; code and instructions remain accessible here.')
            scan_details = [info['version_note'], words('仅打开现有窗口；连接设备与开始扫描仍在原程序中操作。', 'Opens the existing window; connect devices and start acquisition there.')]
            if info.get('layout') == 'package':
                scan_details.append(words('识别为学生包（无 Git），版本来自 delivery_manifest.json。', 'Recognized as the student package (no Git); the revision comes from delivery_manifest.json.'))
            launch = last_launch(self.runtime / 'scan_launches')
            if launch:
                status = launch.get('status')
                code = launch.get('returncode')
                scan_details.append(words(f"最近一次启动：PID {launch.get('pid')}，状态 {status}" + (f"，退出码 {code}" if status == 'exited' else '') + '。',
                                          f"Last launch: PID {launch.get('pid')}, status {status}" + (f", exit code {code}" if status == 'exited' else '') + '.'))
            revision = info.get('revision') or ''
            if revision and info.get('revision_source') == 'delivery_manifest':
                revision = revision[:12] + ' (delivery manifest)'
            scan_blocked = bool(info.get('launch_blocked'))
            tools = [dict(id='scan', available=bool(info['exists']), launchable=bool(info['exists'] and info['platform_supported'] and not scan_blocked),
                          status='package_mismatch' if scan_blocked else 'ready' if info['exists'] and info['platform_supported'] else 'platform_unavailable' if info['exists'] else 'needs_setup',
                          status_note=words('学生包文件与交付清单不一致或无法核对；请重新解压完整原包后启动。', 'Student package files differ from the delivery manifest or could not be checked; re-extract the complete original package before launching.') if scan_blocked else scan_note if info['exists'] else words('请设置 AmScope-Camera 文件夹（Git 检出或解压后的学生包均可）。', 'Set the AmScope-Camera folder (a Git checkout or the extracted student package).'),
                          primary_label=words('启动扫描窗口', 'Open scanner'), github_url=SCAN_URL,
                          guide_url=SCAN_URL + '/blob/main/acquisition/README.md', revision=revision,
                          details=scan_details),
                     dict(id='images', available=available['images'], launchable=available['images'], status='review_available' if available['images'] else 'needs_setup',
                          status_note=words('查看已保存的原图、预测和来源；尚未运行新的拼接。', 'Review saved originals, predictions and provenance; no new stitching has run.'),
                          primary_label=words('查看图像结果', 'Review image results'), github_url=MOSAIC_URL,
                          guide_url=MOSAIC_URL + '/blob/main/flakepipeline/README.md', revision='', details=list(self.inference_details)),
                     dict(id='layers', available=available['layers'], launchable=available['layers'], status='review_available' if available['layers'] else 'needs_setup',
                          status_note=words('核对已保存的原图、层数预测和统计；此入口不运行新的识别。', 'Review saved originals, layer predictions and statistics; this entry does not run new inference.') if available['layers'] else words('未找到已保存的层数结果；请在设置中连接项目或层数识别结果文件夹。', 'No saved layer results found; connect a project or layer inference results folder in Setup.'),
                          primary_label=words('查看层数结果', 'Review layer-number results'), github_url=MOSAIC_URL,
                          guide_url=MOSAIC_URL + '/blob/main/tools/analysis_workbench/README.md', revision='',
                          details=list(self.inference_details) + [words('核对统计区域、有效像素与分母；预测图不代表独立验证。', 'Check counting support, valid pixels and denominators; a prediction is not independent validation.')]),
                     dict(id='spectra', available=available['spectra'], launchable=available['spectra'], status='review_available' if available['spectra'] else 'needs_setup',
                          status_note=words('核对已有光谱、样品和生长记录；采谱与位置复核仍需实测。', 'Check spectral/sample/growth records; acquisition and position verification require measurements.'),
                          primary_label=words('查看核对资料', 'Review spectral records'), github_url=MOSAIC_URL,
                          guide_url=MOSAIC_URL + '/blob/main/docs/SPECTRA.md', revision='', details=[])]
            events = self.events()
            try:
                events += self._scanner_history(self.config['scan_repo'])
            except (OSError, ValueError) as exc:
                self.errors.append(words('扫描历史暂不可读：' + str(exc), 'Scanner history is unavailable: ' + str(exc)))
            return {'app': APP, 'version': '0.3-dd-layers', 'platform': platform.system(), 'csrf_token': self.token,
                    'project': {'name': self.project.name if self.project else 'MosaicAgent', 'root': str(self.project or '')},
                    'tools': tools, 'artifacts': artifacts, 'events': sorted(events, key=lambda e: e.get('at', ''))[-80:],
                    'settings': self.config, 'counts': {'images': len({a['path'] for a in artifacts if a['role'] == 'original'}), 'logs': len(events)},
                    'warnings': self.errors, 'next_action': words('先选择要做的事情。扫描在仪器电脑执行；本机可查看拼接、层数和光谱记录。',
                                                                 'Choose a task. Acquire on the instrument computer; review stitching, layer-number and spectral records here.')}

    def configure(self, values):
        if not isinstance(values, dict) or any(k not in self.config for k in values):
            raise ValueError('Unsupported settings')
        changed = dict(self.config)
        for key, value in values.items():
            if not isinstance(value, str) or len(value) > 2048:
                raise ValueError('Expected a folder path')
            if value:
                path = Path(value).expanduser().resolve()
                if not path.is_dir():
                    raise ValueError('Folder does not exist: ' + str(path))
                value = str(path)
            changed[key] = value
        with self.lock:
            atomic_json(self.config_path, changed)
            self.config = changed
            self.event('workbench', 'settings_saved', words('已更新本机路径。', 'Local paths updated.'))
        return {'ok': True, 'message': words('设置已保存', 'Settings saved')}

    def action(self, tool, view=None):
        if tool not in {'scan', 'images', 'layers', 'spectra'}:
            raise ValueError('Unknown action')
        if view is not None:
            raise ValueError('Unknown tool view')
        with self.lock:
            status = next(t for t in self.state()['tools'] if t['id'] == tool)
            if not status['launchable']:
                raise WorkbenchError(status['status_note']['zh'], status['status_note']['en'])
            self.event(tool, 'requested', words('已请求打开工具或记录。', 'Opening the selected tool or record was requested.'))
            if tool == 'scan':
                from scan_bridge import launch_scanner
                launch = launch_scanner(Path(self.config['scan_repo']), self.runtime / 'scan_launches', sys.executable,
                                        on_exit=self._scanner_exited)
                self.event(tool, 'process_started', words(f"扫描窗口启动进程已创建（PID {launch.get('pid')}）；尚不代表开始采集。",
                                                          f"Scanner launch process created (PID {launch.get('pid')}); acquisition has not been verified."), process=launch)
                return {'ok': True, 'message': words('请在扫描窗口连接设备并操作。', 'Connect and operate devices in the scanner window.')}
            self.event(tool, 'review_opened', words('正在查看已保存的资料；未运行新的模型或测量。', 'Reviewing saved artifacts; no new inference or measurement was run.'))
            return {'ok': True, 'message': words('请在下方查看图像或记录。', 'Review images or records below.')}

    def _scanner_exited(self, record):
        """Called from the launch watcher thread when the scanner process ends. An exit
        is recorded as a fact with its code and log tail; it is never interpreted as a
        completed or successful acquisition."""
        code = record.get('returncode')
        tail = [line for line in record.get('log_tail') or [] if line.strip()][-3:]
        detail_zh = ('；日志末尾：' + ' | '.join(tail)) if tail else ''
        detail_en = ('; log tail: ' + ' | '.join(tail)) if tail else ''
        if code == 0:
            message = words(f"扫描窗口进程已退出（退出码 0）{detail_zh}。采集是否完成以扫描程序自身的历史为准。",
                            f"Scanner process exited (code 0){detail_en}. Whether acquisition happened is decided by the scanner's own history.")
            kind = 'process_exited'
        else:
            message = words(f"扫描窗口进程已退出，退出码 {code}{detail_zh}。请打开启动日志核对原因；不要重复点击启动。",
                            f"Scanner process exited with code {code}{detail_en}. Check the launch log before launching again.")
            kind = 'error'
        try:
            self.event('scan', kind, message, process={k: record.get(k) for k in ('pid', 'returncode', 'log_path', 'entrypoint', 'started_at_utc', 'ended_at_utc')})
        except OSError:
            pass

    def artifact(self, aid):
        with self.lock:
            if aid not in self.artifacts:
                self.collect()
            item = self.artifacts.get(aid)
            if not item:
                raise FileNotFoundError('Unknown artifact')
            path = item['path']
            if not any(inside(path, root) for root in self.roots().values()):
                raise ValueError('Artifact is outside configured folders')
            if item['member']:
                with zipfile.ZipFile(path) as z:
                    info = z.getinfo(item['member'])
                    if info.file_size > 32 * 1024 * 1024:
                        raise ValueError('Image exceeds preview limit')
                    data = z.read(info)
            else:
                if path.stat().st_size > 64 * 1024 * 1024:
                    raise ValueError('File exceeds preview limit; use its recorded local path')
                data = path.read_bytes()
            if item['sha256'] and hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Artifact checksum differs from its source manifest')
            return data, item['public']['mime'], path.name


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send(self, data, code=200, mime='application/json; charset=utf-8', attachment=None, artifact=False):
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self' data:; script-src " + ("'none'" if artifact else "'self'") + "; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'")
        if attachment:
            self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(attachment))
        self.end_headers()
        self.wfile.write(data)

    def local_request(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    def do_GET(self):
        if not self.local_request():
            return self.send({'error': 'Invalid host'}, 403)
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/api/state':
                return self.send(self.server.workbench.state())
            if path.startswith('/api/artifact/'):
                data, mime, name = self.server.workbench.artifact(path.rsplit('/', 1)[-1])
                attach = name if mime not in {'text/html', 'application/pdf'} and not mime.startswith('image/') else None
                return self.send(data, mime=mime, attachment=attach, artifact=True)
            setup_files = {'/stitch-setup/': 'index.html', '/stitch-setup/app.js': 'app.js',
                           '/stitch-setup/style.css': 'style.css'}
            if path in setup_files:
                target = REPO / 'tools/stitch_setup/static' / setup_files[path]
                return self.send(target.read_bytes(), mime=(mimetypes.guess_type(str(target))[0] or 'text/plain') + '; charset=utf-8')
            names = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            if path in names:
                target = HERE / 'static' / names[path]
                return self.send(target.read_bytes(), mime=(mimetypes.guess_type(str(target))[0] or 'text/plain') + '; charset=utf-8')
            raise FileNotFoundError(path)
        except FileNotFoundError:
            return self.send({'error': 'Not found'}, 404)
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
            return self.send({'error': str(exc)}, 400)

    def do_POST(self):
        origin = self.headers.get('Origin')
        expected = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        if not self.local_request() or (origin and origin not in expected) or not secrets.compare_digest(self.headers.get('X-Workbench-Token', ''), self.server.workbench.token):
            return self.send({'ok': False, 'message': words('请求来源无效', 'Invalid request origin')}, 403)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
            if self.path == '/api/config':
                result = self.server.workbench.configure(body)
            elif self.path == '/api/action':
                result = self.server.workbench.action(body.get('tool'), body.get('view'))
            elif self.path in {'/api/stitch/inspect', '/api/stitch/prepare'}:
                if str(REPO) not in sys.path:
                    sys.path.insert(0, str(REPO))
                try:
                    from tools.stitch_setup.service import dispatch
                    result = dispatch(self.path.rsplit('/', 1)[-1], body)
                except ImportError as exc:
                    raise ValueError('Stitch setup needs the MosaicAgent requirements: python -m pip install -r requirements.txt. ' + str(exc)) from exc
                if result.get('status') == 'profile_prepared':
                    self.server.workbench.event('images', 'profile_prepared',
                        words('拼接参数与输入检查已保存；尚未运行拼接。',
                              'Stitch profile and input checks saved; stitching has not run.'),
                        bundle_dir=result.get('bundle_dir'))
            else:
                return self.send({'ok': False, 'message': words('未知操作', 'Unknown action')}, 404)
            self.send(result)
        except WorkbenchError as exc:
            self.fail(words('未完成：' + exc.message['zh'], 'Not completed: ' + exc.message['en']))
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self.fail(words('未完成：' + str(exc), 'Not completed: ' + str(exc)))

    def fail(self, message):
        try:
            self.server.workbench.event('workbench', 'error', message)
        except OSError:
            pass
        self.send({'ok': False, 'message': message}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'workbench.local.json')
    parser.add_argument('--port', type=int, default=8792)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    workbench = Workbench(args.config)
    url = f'http://127.0.0.1:{args.port}/'
    try:
        httpd = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as bind_error:
        if bind_error.errno != errno.EADDRINUSE:
            parser.exit(1, f'Could not open the local workbench port: {bind_error}\n')
        try:
            with urllib.request.urlopen(url + 'api/state', timeout=2) as response:
                state = json.loads(response.read(MAX_JSON))
            if state.get('app') != APP or state.get('settings') != workbench.config:
                raise ValueError('A different task is already using this port')
        except Exception as exc:
            parser.exit(1, f'Port {args.port} is occupied; existing tools were retained. {exc}\n')
        print(url, flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return
    httpd.workbench = workbench
    workbench.event('workbench', 'opened', words('统一入口已打开；读取现有结果，尚未启动扫描。',
                                                 'Workbench opened; existing results are read without starting acquisition.'))
    print(url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()  # Never terminate independently launched acquisition processes.


if __name__ == '__main__':
    main()

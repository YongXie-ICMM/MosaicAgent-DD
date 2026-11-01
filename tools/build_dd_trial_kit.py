#!/usr/bin/env python3
"""Build a key-free local DD trial: analysis + unchanged C1 + one-page PDF.

This teacher's tool never installs dependencies, touches instruments or models,
or calls external services. It preserves all source and delivery artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_student_bundle as analysis

TOP_LEVEL = 'MosaicAgent-DD'
OUTER_MANIFEST = 'trial_kit_manifest.json'
QUICKSTART = '00_学生试跑极简说明.pdf'


class TrialBuildError(ValueError):
    pass


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _copy_file(source, target):
    if source.is_symlink() or not source.is_file():
        raise TrialBuildError(f'Only regular source files may be packed: {source.name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _record(root, path):
    return {'path': path.relative_to(root).as_posix(), 'sha256': analysis.sha256_file(path),
            'bytes': path.stat().st_size}


def _add_scanner(root, scanner):
    manifest = scanner / 'delivery_manifest.json'
    try:
        content = json.loads(manifest.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as error:
        raise TrialBuildError('Cannot read the original C1 delivery manifest') from error
    if content.get('release') != '954-1080p-C1-candidate':
        raise TrialBuildError('This kit requires the unchanged C1 communication-logging candidate')
    records = content.get('files')
    if not isinstance(records, list) or len(records) != 73:
        raise TrialBuildError('C1 original inventory must contain 73 files')
    seen = set()
    for record in records:
        relative = record.get('path', '')
        parts = PurePosixPath(relative).parts
        if (not parts or '\\' in relative or ':' in relative or PurePosixPath(relative).is_absolute()
                or any(part in {'', '.', '..'} for part in parts) or relative.casefold() in seen):
            raise TrialBuildError('Unsafe or duplicate path in C1 manifest')
        seen.add(relative.casefold())
        source = scanner / relative
        if (source.is_symlink() or not source.is_file() or source.stat().st_size != record['bytes']
                or analysis.sha256_file(source) != record['sha256']):
            raise TrialBuildError(f'C1 file does not match its original manifest: {relative}')
        _copy_file(source, root / 'scanner_C1' / relative)
    _copy_file(manifest, root / 'scanner_C1/delivery_manifest.json')
    return analysis.sha256_file(manifest)


def _start_text():
    return '''DD 学生完整试用包｜2026-11-01

1. 完整解压到本地文件夹，不要在 ZIP 里直接启动。先看同目录的“00_学生试跑极简说明.pdf”。
2. Windows 首次双击 01_install.bat。需 64 位 Python 3.9+（沿用仪器现有 3.9.18，含 Tcl/Tk），首次安装需联网。
3. 双击 03_open_workbench.bat；按网页箭头：扫描 → 日志交接 → 自动校色 → 拼接 → 结果检查 → 层数识别 → 空间图。
4. 先试 3×3。交接时选本次扫描的单独文件夹，不要选 ZIP 或包含多次扫描的父文件夹。
5. 自动校色已默认启用；不用选择颜色校正模式。遇到红色提示请保留画面及日志，交老师核查。
6. 把本次扫描原始文件夹、结果和整个 Memory 文件夹一起保留并交回。

仪器端：本包默认使用 scanner_C1 中的通信日志扫描程序；不要改用 acquisition 中的历史快照。
分析端和扫描端共用本包的 .venv，不必再运行 scanner_C1/01_setup.bat。
相机、位移台 USB 驱动及 VC++ 2013 x64 运行库由老师确认；它们不能由 pip 安装。
新电脑若提示 MSVCR120.dll / MSVCP120.dll 缺失，请把提示交老师处理。

识别前：老师需核对本批样品、0409模型适用性、拼接位移和图像尺度，准备本批的层数 JSON 配置。
configs/layer_workflow.teacher_template.json 是未确认模板，不是本批可直接运行的参数。
两张示例原图仅用于 02_run_layer_demo 的安装/推理试验；不能当作相邻扫描图拼接。
本包不携带 Kimi 密钥；默认可用离线流程。不需要学生填写 API Key。

本版是学生试用候选：C1已完成9张和1000张正常扫描；整合后的完整流程仍需本次Windows实机试跑。
不要把程序正常结束等同于结果已核验。各步查看实际图像、张数、覆盖和识别结果。
macOS可做分析；C1仪器扫描仅支持Windows x64。
'''


def build(repo: Path, scanner: Path, pdf: Path, out: Path) -> dict:
    repo, scanner, pdf, out = (Path(p).resolve() for p in (repo, scanner, pdf, out))
    if out.exists():
        raise TrialBuildError('Output already exists; choose a new filename to preserve the earlier kit')
    if not pdf.is_file() or pdf.is_symlink() or not pdf.read_bytes().startswith(b'%PDF-'):
        raise TrialBuildError('The verified one-page PDF is required')
    assets = analysis.verify_assets(repo)
    files, enumeration = analysis.select_files(repo)
    files = sorted(set(files) | {analysis.ASSET_MANIFEST} | {r['path'] for r in assets['files']})
    reserved = {'trial_kit.json', OUTER_MANIFEST, QUICKSTART, '00_开始这里.txt', 'bundle_manifest.json'}
    if reserved.intersection(files) or any(p.startswith('scanner_C1/') for p in files):
        raise TrialBuildError('Source checkout contains reserved kit output paths')
    manifest = analysis.build_manifest(repo, files, enumeration, assets)
    # Keep the historical full manual in docs/. Surface only the new short PDF.
    manifest['files'] = [r for r in manifest['files'] if not r.get('source')]
    manifest.update(file_count=len(manifest['files']),
                    total_bytes=sum(r['bytes'] for r in manifest['files']),
                    distribution='local_student_trial', bundle_name=out.name,
                    notes=['Analysis source inventory only. The outer trial_kit_manifest.json binds C1, PDF and all kit files.',
                           'No credentials, experiment data, histories or virtual environment included.',
                           'The original0409 model and demo archive are copied without modification.'])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dd_trial_build_') as temporary:
        root = Path(temporary) / TOP_LEVEL
        root.mkdir()
        for record in manifest['files']:
            _copy_file(repo / record['path'], root / record['path'])
            actual = _record(root, root / record['path'])
            if any(actual[key] != record[key] for key in ('sha256', 'bytes')):
                raise TrialBuildError('Analysis source changed while building; retry from a stable checkout')
        _json(root / 'bundle_manifest.json', manifest)
        c1sha = _add_scanner(root, scanner)
        _copy_file(pdf, root / QUICKSTART)
        (root / '00_开始这里.txt').write_text(_start_text(), encoding='utf-8-sig')
        _json(root / 'trial_kit.json', {
            'schema_version': 1, 'kind': 'dd_student_trial',
            'scanner': {'path': 'scanner_C1', 'manifest_sha256': c1sha},
            'source_commit': manifest['source_commit'], 'credentials_included': False,
            'distribution': 'local_student_trial', 'quickstart': QUICKSTART,
        })
        from tools.dd_trial import initial_config
        initial_config(root)  # Pure validation: no settings, SDK or model imports.
        outer_files = [_record(root, p) for p in sorted(root.rglob('*')) if p.is_file()]
        outer = {
            'schema_version': 1, 'kind': 'dd_student_trial_manifest', 'top_level_folder': TOP_LEVEL,
            'bundle_name': out.name, 'built_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_commit': manifest['source_commit'], 'source_branch': manifest['source_branch'],
            'working_tree_dirty': manifest['working_tree_dirty'], 'credentials_included': False,
            'distribution': 'local_student_trial', 'analysis_manifest': 'bundle_manifest.json',
            'scanner_manifest': 'scanner_C1/delivery_manifest.json', 'quickstart': QUICKSTART,
            'file_count': len(outer_files), 'total_bytes': sum(r['bytes'] for r in outer_files),
            'files': outer_files,
            'validation_scope': {
                'hardware_calls_during_build': 0, 'model_calls_during_build': 0,
                'external_model_calls': 0, 'windows_end_to_end': 'student_trial_pending',
                'C1_normal_scan_evidence': '2026-10-31: 9/9 and1000/1000; no injected motion failure',
                'C1_manifest': 'Original historical manifest preserved byte-for-byte; new scope is recorded here.',
            },
        }
        _json(root / OUTER_MANIFEST, outer)
        building = out.with_name(out.name + '.building')
        if building.exists():
            raise TrialBuildError('An unfinished .building file exists; inspect it before retrying')
        with zipfile.ZipFile(building, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob('*')):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(f'{TOP_LEVEL}/{relative}', datetime.now(timezone.utc).timetuple()[:6])
                info.compress_type = (zipfile.ZIP_STORED if path.suffix.lower() in analysis.STORED_SUFFIXES
                                      else zipfile.ZIP_DEFLATED)
                mode = 0o100755 if path.suffix in {'.command', '.sh'} else 0o100644
                info.external_attr = mode << 16
                with path.open('rb') as source, archive.open(info, 'w') as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        from tools.verify_dd_trial_kit import check_trial_kit
        report = check_trial_kit(building)
        os.replace(building, out)
    return {**report, 'path': str(out), 'archive_bytes': out.stat().st_size,
            'archive_sha256': analysis.sha256_file(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=analysis.REPO)
    parser.add_argument('--scanner', type=Path, required=True, help='Exact unchanged C1 delivered folder')
    parser.add_argument('--pdf', type=Path, required=True, help='Verified one-page student guide')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.repo, args.scanner, args.pdf, args.out), ensure_ascii=False, indent=2))
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        print(f'Trial kit not completed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

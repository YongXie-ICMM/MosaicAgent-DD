"""Small read-only history viewer; importing it opens no window or hardware."""
import json
import os
from pathlib import Path
import platform
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk

from shared_history import list_histories, read_history


LABELS = {
    'history_started': '程序历史开始', 'history_closed': '程序正常关闭',
    'stage_connect_requested': '请求连接位移台', 'stage_connected': '位移台已连接',
    'stage_connect_failed': '位移台连接失败', 'camera_connect_requested': '请求连接相机',
    'camera_connected': '相机已连接', 'camera_connect_failed': '相机连接失败',
    'devices_disconnected': '设备已断开', 'disconnect_failed': '断开失败',
    'manual_move_attempt': '手动移动：请求', 'manual_move_completed': '手动移动：计数核对完成',
    'manual_move_failed': '手动移动：失败', 'move_attempt': '扫描移动：请求',
    'move_completed': '扫描移动：计数核对完成', 'move_failed': '扫描移动：失败',
    'position_checked': '控制器位置核对', 'session_started': '扫描开始',
    'session_finished': '扫描结束', 'capture_attempt': '请求取图',
    'capture_failed': '取图失败', 'candidate_saved': '候选照片已保存',
    'human_review': '人工检查 / 取消', 'candidate_promotion_requested': '准备保存正式照片',
    'candidate_accepted': '正式照片已保存', 'capture_success': '采集记录完成',
    'emergency_stop_requested': '请求紧急停止', 'resume': '继续扫描',
    'resume_camera_readback': '继续前核对相机', 'manual_dwell': '仅停留，未拍照',
}


def _open_folder(path):
    path = str(Path(path).resolve())
    if platform.system() == 'Windows':
        os.startfile(path)
    else:
        subprocess.Popen(['open' if platform.system() == 'Darwin' else 'xdg-open', path])


def show_history(parent, root_dir, current_path=None):
    window = tk.Toplevel(parent)
    window.title('移动与保存历史 · 只读')
    window.geometry('1020x640')
    window.minsize(760, 460)
    window.columnconfigure(0, weight=1)
    window.rowconfigure(2, weight=2)
    window.rowconfigure(4, weight=1)
    toolbar = ttk.Frame(window, padding=10)
    toolbar.grid(row=0, column=0, sticky='ew')
    toolbar.columnconfigure(0, weight=1)
    selection = ttk.Combobox(toolbar, state='readonly')
    selection.grid(row=0, column=0, sticky='ew', padx=(0, 8))
    status = tk.StringVar(value='连接设备或操作后自动生成历史。')
    ttk.Label(window, textvariable=status, wraplength=960, padding=(10, 0, 10, 8)).grid(
        row=1, column=0, sticky='ew')
    box = ttk.Frame(window, padding=(10, 0))
    box.grid(row=2, column=0, sticky='nsew')
    box.columnconfigure(0, weight=1); box.rowconfigure(0, weight=1)
    table = ttk.Treeview(box, columns=('seq', 'time', 'action', 'result', 'file'), show='headings')
    for key, text, width in (('seq', '序号', 50), ('time', '时间（UTC）', 120),
                              ('action', '操作', 215), ('result', '结果 / 位置', 270),
                              ('file', '照片文件', 230)):
        table.heading(key, text=text); table.column(key, width=width, stretch=key in ('action', 'result', 'file'))
    table.grid(row=0, column=0, sticky='nsew')
    scroll = ttk.Scrollbar(box, orient='vertical', command=table.yview)
    scroll.grid(row=0, column=1, sticky='ns'); table.configure(yscrollcommand=scroll.set)
    ttk.Label(window, text='点击一行查看完整 JSON。移动完成表示控制器计数已核对；实际画面需结合图像与人工确认。',
              padding=(10, 8)).grid(row=3, column=0, sticky='ew')
    details = tk.Text(window, wrap='word', height=9, font=('Menlo', 10), state='disabled')
    details.grid(row=4, column=0, sticky='nsew', padx=10, pady=(0, 10))
    paths, records = [], {}

    def selected_path():
        i = selection.current()
        return paths[i] if 0 <= i < len(paths) else None

    def load_selected(*_):
        records.clear()
        for item in table.get_children():
            table.delete(item)
        path = selected_path()
        if path is None:
            status.set('尚无历史。连接设备、手动移动和扫描时会自动记录。')
            return
        try:
            history = read_history(path, recover=True)
            for event in history['events']:
                action = event['event']
                result = event.get('error') or event.get('decision') or event.get('status') or ''
                result = {'accept': '人工选择接受', 'retake': '原位重拍', 'stop': '停止 / 取消',
                          'completed': '完成', 'failed': '失败', 'aborted': '已中止'}.get(result, result)
                if not result and 'dx_steps' in event:
                    result = '请求步数 ΔX=%s  ΔY=%s' % (event['dx_steps'], event.get('dy_steps', '?'))
                if not result and 'x_steps' in event:
                    result = '逻辑步数 X=%s  Y=%s' % (event['x_steps'], event.get('y_steps', '?'))
                key = event['event_id']; records[key] = event
                table.insert('', 'end', iid=key, values=(event['sequence'], event['timestamp_utc'][11:19],
                    LABELS.get(action, action), result, Path(event.get('filepath', '')).name))
            diagnostic = history['read_diagnostics']
            suffix = ' · 有未完成日志，请检查' if diagnostic.get('warning') or diagnostic.get('snapshot_error') else ''
            simulated = ' · 位移台仿真' if history['metadata'].get('stage_simulated') else ''
            status.set('%s 条记录%s · %s · %s%s' % (history['event_count'], simulated,
                '程序已关闭' if history['is_closed'] else '程序运行中或尚未正常关闭', path, suffix))
            if table.get_children():
                table.see(table.get_children()[-1])
        except Exception as exc:
            status.set('无法读取，请保留原文件：' + str(exc))

    def refresh():
        previous = selected_path() or (str(current_path) if current_path else None)
        try:
            entries = list_histories(root_dir)
            paths[:] = [entry['history_path'] for entry in reversed(entries)]
            selection['values'] = [Path(path).parent.name for path in paths]
            if paths:
                selection.current(paths.index(previous) if previous in paths else 0)
            load_selected()
        except Exception as exc:
            status.set('历史目录读取失败：' + str(exc))

    def inspect(*_):
        selected = table.selection()
        value = records.get(selected[0]) if selected else None
        details.configure(state='normal'); details.delete('1.0', tk.END)
        if value is not None:
            details.insert('1.0', json.dumps(value, ensure_ascii=False, indent=2))
        details.configure(state='disabled')

    def open_selected():
        path = selected_path()
        if path:
            try:
                _open_folder(Path(path).parent)
            except Exception as exc:
                messagebox.showerror('打开文件夹失败', str(exc), parent=window)

    ttk.Button(toolbar, text='刷新', command=refresh).grid(row=0, column=1, padx=(0, 8))
    ttk.Button(toolbar, text='打开 JSON 所在文件夹', command=open_selected).grid(row=0, column=2)
    selection.bind('<<ComboboxSelected>>', load_selected)
    table.bind('<<TreeviewSelect>>', inspect)
    refresh()
    return window

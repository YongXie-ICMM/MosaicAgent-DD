"""Display-only Chinese/English adapter; acquisition guards are defined in the scan core.

Underlying Tk variables, commands, parameters, raw JSON and instrument calls are
never translated or rewritten. Only displayed strings and dialog copy change.
Typography comes from gui_theme; a styling failure keeps the original look.
"""
import json
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk

try:
    from gui_theme import apply_theme
except Exception as exc:  # display styling must never stop the acquisition program
    apply_theme = None
    print('[ui] Display theme unavailable; original fonts kept: ' + repr(exc))


def _styled(step, *args):
    try:
        return step(*args)
    except Exception as exc:
        print('[ui] Display theme step skipped; original fonts kept: ' + repr(exc))
        return None


CATALOG = json.loads(Path(__file__).with_name('gui_language_catalog.json').read_text(encoding='utf-8'))
ZH_TITLES = {
    '显微扫描工作台 v3.4 / 自动走位 + 人工确认 + 共享历史': '显微扫描工作台 / 954-1080p-A1 · 1920 × 1080',
    'v3.4 · 看图确认后再走下一点': '954-1080p-A1 · 1920 × 1080',
}


def _patterns():
    patterns = []
    for parts in CATALOG['templates']:
        literals = sum(len(p) for p in parts if isinstance(p, str))
        regex = ''.join(re.escape(p) if isinstance(p, str) else '(.*?)' for p in parts)
        patterns.append((literals, re.compile(regex, re.S), parts))
    for source, target in CATALOG['en'].items():
        if '%s' in source:
            parts = re.split('(%s)', source)
            tokens = [{'field': 'formatted'} if p == '%s' else p for p in parts]
            english = re.split('(%s)', target)
            # These whole-format translations preserve the order of fields.
            tokens = [(a, b) if isinstance(a, str) else a for a, b in zip(tokens, english)]
            regex = ''.join(re.escape(p[0]) if isinstance(p, tuple) else '(.*?)' for p in tokens)
            patterns.append((len(source), re.compile(regex, re.S), tokens))
    return sorted(patterns, key=lambda p: p[0], reverse=True)


PATTERNS = _patterns()


def translate(text, language='en', depth=0):
    text = str(text)
    if language == 'zh':
        return ZH_TITLES.get(text, text)
    if text in CATALOG['en']:
        return CATALOG['en'][text]
    if depth >= 4:
        return text
    for _, pattern, parts in PATTERNS:
        match = pattern.fullmatch(text)
        if not match:
            continue
        fields = iter(match.groups())
        out = []
        for part in parts:
            if isinstance(part, tuple):
                out.append(part[1])
            elif isinstance(part, str):
                out.append(CATALOG['en'].get(part, part))
            else:
                value = next(fields)
                name = part['field']
                if any(word in name for word in ('error', 'exc', 'message', 'summary', 'formatted')):
                    value = translate(value, language, depth + 1)
                out.append(value)
        return ''.join(out)
    return text


class Dialogs:
    """Proxy messages only; original dialog functions decide their results."""
    def __init__(self, owner, original):
        self.owner, self.original = owner, original

    def __getattr__(self, name):
        function = getattr(self.original, name)
        if not callable(function):
            return function

        def display(*args, **kwargs):
            args = list(args)
            for i in range(min(2, len(args))):
                args[i] = self.owner.text(args[i])
            for key in ('title', 'message', 'detail'):
                if key in kwargs:
                    kwargs[key] = self.owner.text(kwargs[key])
            return function(*args, **kwargs)
        return display


class LanguageUI:
    def __init__(self, root, theme=None):
        self.root, self.language, self.theme = root, 'zh', theme
        self.widgets, self.titles, self.trees = {}, {}, {}
        self.button = None
        self.attach(root)
        header = root.btn_auto.master
        self.button = ttk.Button(header, text='English', command=self.toggle)
        self.button.pack(side=tk.RIGHT, padx=(8, 0))
        # Limit dialog/viewer adaptation to the scan module; no global Tk patches.
        namespace = root.on_show_history.__func__.__globals__
        namespace['messagebox'] = Dialogs(self, namespace['messagebox'])
        original_history = namespace['show_history']
        history_namespace = original_history.__globals__
        history_namespace['messagebox'] = Dialogs(self, history_namespace['messagebox'])

        def show_history(*args, **kwargs):
            window = original_history(*args, **kwargs)
            if self.theme is not None:
                _styled(self.theme.style_history, window)
            self.attach(window)
            return window
        namespace['show_history'] = show_history

    def text(self, value):
        return translate(value, self.language)

    def toggle(self):
        self.set_language('en' if self.language == 'zh' else 'zh')

    def set_language(self, language):
        if language not in ('zh', 'en'):
            raise ValueError('Language must be zh or en')
        self.language = language
        for widget, state in list(self.widgets.items()):
            if not widget.winfo_exists():
                self.widgets.pop(widget, None)
                continue
            if 'raw_variable' in state:
                state['display_variable'].set(self.text(state['raw_variable'].get()))
            elif 'raw_text' in state:
                state['configure'](text=self.text(state['raw_text']))
        for widget, (title, raw) in list(self.titles.items()):
            if widget.winfo_exists():
                title(self.text(raw))
        for tree, state in list(self.trees.items()):
            if not tree.winfo_exists():
                continue
            for column, raw in state['headings'].items():
                state['heading'](column, text=self.text(raw))
            for item, raw in list(state['values'].items()):
                if tree.exists(item):
                    state['item'](item, values=self._tree_values(raw))
                else:
                    state['values'].pop(item, None)
        self.button.configure(text='中文' if language == 'en' else 'English')

    def attach(self, widget):
        if widget in self.widgets or widget is self.button:
            return
        state = {'configure': widget.configure}
        self.widgets[widget] = state
        if isinstance(widget, (tk.Tk, tk.Toplevel)):
            original_title, raw = widget.title, widget.title()
            self.titles[widget] = (original_title, raw)
            original_title(self.text(raw))
        keys = widget.keys()
        # Entry/combobox contents are user data, not interface copy.
        if not isinstance(widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            raw_name = str(widget.cget('textvariable')) if 'textvariable' in keys else ''
            if raw_name:
                raw = tk.StringVar(master=widget, name=raw_name)
                display = tk.StringVar(master=widget, value=self.text(raw.get()))
                state.update(raw_variable=raw, display_variable=display)
                state['configure'](textvariable=display)

                def update(*_, raw=raw, display=display):
                    display.set(self.text(raw.get()))
                token = raw.trace_add('write', update)
                state['trace'] = token
                widget.bind('<Destroy>', lambda event, w=widget, r=raw, t=token:
                            r.trace_remove('write', t) if event.widget == w else None, add='+')
            elif 'text' in keys:
                state['raw_text'] = str(widget.cget('text'))
                state['configure'](text=self.text(state['raw_text']))

            def configure(cnf=None, **kwargs):
                options = dict(cnf) if isinstance(cnf, dict) else {}
                options.update(kwargs)
                if 'text' in options:
                    state['raw_text'] = options['text']
                    options['text'] = self.text(options['text'])
                if options:
                    return state['configure'](**options)
                return state['configure'](cnf) if cnf is not None else state['configure']()
            widget.configure = widget.config = configure
        if isinstance(widget, ttk.Treeview):
            self._attach_tree(widget)
        for child in widget.winfo_children():
            self.attach(child)

    def _tree_values(self, values):
        values = list(values)
        # History action/result columns only; timestamps and filenames unchanged.
        for i in (2, 3):
            if i < len(values):
                values[i] = self.text(values[i])
        return values

    def _attach_tree(self, tree):
        state = {'heading': tree.heading, 'insert': tree.insert, 'item': tree.item,
                 'headings': {}, 'values': {}}
        self.trees[tree] = state
        for column in tree['columns']:
            raw = tree.heading(column, 'text')
            state['headings'][column] = raw
            state['heading'](column, text=self.text(raw))
        for item in tree.get_children():
            raw = tree.item(item, 'values')
            state['values'][item] = raw
            state['item'](item, values=self._tree_values(raw))

        def insert(parent, index, iid=None, **kwargs):
            raw = kwargs.get('values')
            if raw is not None:
                kwargs['values'] = self._tree_values(raw)
            item = state['insert'](parent, index, iid, **kwargs)
            if raw is not None:
                state['values'][item] = raw
            return item
        tree.insert = insert


def install_language_switch(root):
    # Fonts first, while widgets still carry their raw text and variables.
    theme = _styled(apply_theme, root) if apply_theme is not None else None
    ui = LanguageUI(root, theme)
    if theme is not None:
        _styled(theme.style_language_button, ui.button)
    return ui

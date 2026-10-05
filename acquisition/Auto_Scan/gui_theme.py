"""Display-only typography for the 954-base scan window (954-base-L1 + type pass).

gui_language.install_language_switch() calls apply_theme(root) once, after the
unchanged scan core has built its widgets and before translations attach, then
style_language_button() and, for every history viewer, style_history(window).
Only fonts, text colours, ttk paddings and wrap widths change. Commands, button
states, Tk variables, image pixels, instruments and files are never touched.
"""
import tkinter as tk
from tkinter import font as tkfont, ttk

# The first installed family wins (checked with tkinter.font.families()).
# Fonts are never downloaded or bundled. Microsoft YaHei UI carries Segoe-style
# Latin glyphs, so Chinese, English and digits share one design on Windows.
UI_FAMILIES = {
    'win32': ('Microsoft YaHei UI', 'Microsoft YaHei', 'Segoe UI'),
    'aqua': ('PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'Helvetica Neue'),
    'x11': ('Noto Sans CJK SC', 'Source Han Sans SC', 'Source Han Sans CN',
            'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans'),
}
MONO_FAMILIES = {
    'win32': ('Consolas', 'Cascadia Mono', 'Lucida Console', 'Courier New'),
    'aqua': ('Menlo', 'SF Mono', 'Monaco', 'Courier'),
    'x11': ('DejaVu Sans Mono', 'Noto Sans Mono', 'Liberation Mono', 'FreeMono'),
}
# Pixel sizes, so the pixel-based layout renders alike on every platform.
# 13 px equals 10 pt at Windows 100 % (96 dpi).
ROLES = {
    'title': (20, 'bold'),          # window title, one per window
    'section': (13, 'bold'),        # group and image-panel headings, table headings
    'body': (13, 'normal'),         # labels, buttons, entries, radio buttons, status
    'strong': (13, 'bold'),         # emergency stop only
    'small': (12, 'normal'),        # version, live state, connection, position readouts, hints
    'numeric': (14, 'bold'),        # scan progress counter
    'placeholder': (14, 'normal'),  # text inside the dark image panels
    'mono': (12, 'normal'),         # raw JSON in the history viewer
}
STANDARD_FONTS = {
    'TkDefaultFont': 'body', 'TkTextFont': 'body', 'TkMenuFont': 'body',
    'TkHeadingFont': 'section', 'TkCaptionFont': 'section',
    'TkSmallCaptionFont': 'small', 'TkTooltipFont': 'small', 'TkIconFont': 'small',
    'TkFixedFont': 'mono',
}
TEXT = '#243c49'           # core label colour, now shared by buttons and entries
MUTED = '#5b6f79'          # core used #647a84 (4.0:1 on #edf2f5); this is 4.6:1
CORE_MUTED = '#647a84'
DISABLED_TEXT = '#6e7c83'  # clam default #999999 was 2.0:1 on the button face; now 3.1:1
LABEL_ROLES = {'var_progress': 'numeric', 'var_live_status': 'small', 'var_connection': 'small',
               'var_pos_steps': 'small', 'var_pos_um': 'small', 'var_status': None, 'var_review': None}


def choose(candidates, installed, fallback):
    """First candidate that is installed; otherwise the platform's own default family."""
    return next((family for family in candidates if family in installed), fallback)


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def _option(widget, name):
    try:
        return str(widget.cget(name))
    except tk.TclError:
        return ''


def _pair(value):
    """pack -padx/-pady as (before, after)."""
    items = [int(float(str(v))) for v in (value if isinstance(value, (tuple, list)) else str(value).split())]
    return (items[0], items[-1]) if items else (0, 0)


def _box(widget):
    """A widget's ttk padding as (left, top, right, bottom), following ttk's 1-4 value rule."""
    raw = widget.cget('padding')
    n = [int(float(str(v))) for v in (widget.tk.splitlist(raw) if raw not in ('', None) else ())] or [0]
    left = n[0]
    top = n[1] if len(n) > 1 else left
    right = n[2] if len(n) > 2 else left
    bottom = n[3] if len(n) > 3 else top
    return left, top, right, bottom


class Theme:
    def __init__(self, root):
        self.root = root
        system = str(root.tk.call('tk', 'windowingsystem'))
        installed = set(tkfont.families(root))
        default = tkfont.nametofont('TkDefaultFont', root=root).actual('family')
        fixed = tkfont.nametofont('TkFixedFont', root=root).actual('family')
        self.family = choose(UI_FAMILIES.get(system, UI_FAMILIES['x11']), installed, default)
        self.mono_family = choose(MONO_FAMILIES.get(system, MONO_FAMILIES['x11']), installed, fixed)
        self.fonts = {}      # role -> named Font; these references keep the fonts alive
        self.roles = {}      # widget -> role, for widgets that carry their own font
        for role, (pixels, weight) in ROLES.items():
            family = self.mono_family if role == 'mono' else self.family
            self.fonts[role] = self._named('ScanUI%sFont' % role.capitalize(), family=family,
                                           size=-pixels, weight=weight)

    def _named(self, name, **options):
        try:
            font = tkfont.Font(root=self.root, name=name, exists=True)
            font.configure(**options)
        except tk.TclError:
            font = tkfont.Font(root=self.root, name=name, **options)
        return font

    def _assign(self, widget, role, **extra):
        widget.configure(font=self.fonts[role], **extra)
        self.roles[widget] = role

    def configure_fonts(self):
        """Tk's standard fonts: entries, dialogs, menus and lists follow the same scale."""
        for name, role in STANDARD_FONTS.items():
            try:
                tkfont.nametofont(name, root=self.root).configure(**self.fonts[role].configure())
            except tk.TclError:
                pass
        self.root.option_add('*TCombobox*Listbox.font', self.fonts['body'])

    def configure_styles(self):
        style, fonts = ttk.Style(self.root), self.fonts
        style.configure('TLabel', font=fonts['body'])
        style.configure('Header.TLabel', font=fonts['title'])
        style.configure('TLabelframe.Label', font=fonts['section'])
        style.configure('TLabelframe', labelmargins=(0, 0, 0, 3))
        style.configure('TRadiobutton', font=fonts['body'], foreground=TEXT, padding=1)
        style.configure('TEntry', foreground=TEXT)
        style.configure('TCombobox', foreground=TEXT)
        # One vertical padding for every button style, so mixed rows line up.
        style.configure('TButton', font=fonts['body'], foreground=TEXT, padding=(8, 2))
        style.configure('Accent.TButton', padding=(12, 2))
        style.configure('Stop.TButton', font=fonts['strong'], padding=(12, 2))
        style.map('TButton', foreground=[('disabled', DISABLED_TEXT)])
        style.configure('Treeview', font=fonts['body'], foreground=TEXT,
                        rowheight=fonts['body'].metrics('linespace') + 4)
        style.configure('Treeview.Heading', font=fonts['section'], foreground=TEXT)

    def style_main_window(self):
        root = self.root
        header = root.btn_auto.master
        variables = {str(getattr(root, name)): name for name in LABEL_ROLES if hasattr(root, name)}
        labels = {}
        for widget in _walk(root):
            if widget.winfo_class() != 'TLabel':
                continue
            variable = variables.get(_option(widget, 'textvariable'))
            version_line = (variable is None and widget.master is header
                            and not _option(widget, 'style'))
            if variable is not None:
                labels[variable] = widget
                if LABEL_ROLES[variable]:
                    self._assign(widget, LABEL_ROLES[variable])
            elif _option(widget, 'font'):
                self._assign(widget, 'section')        # live and review image headings
            elif version_line:
                self._assign(widget, 'small')
            if version_line or _option(widget, 'foreground') == CORE_MUTED:
                widget.configure(foreground=MUTED)
        for name in ('_live_panel', '_reference_label', '_candidate_label'):
            if getattr(root, name, None) is not None:
                self._assign(getattr(root, name), 'placeholder')
        if 'var_connection' in labels:
            labels['var_connection'].configure(wraplength=262)     # core: 246
        for name in ('var_pos_steps', 'var_pos_um'):
            if name in labels:
                labels[name].configure(wraplength=284)            # core: 276; column is 286
        self._tighten(header, labels.get('var_status'))
        self._align_header_baseline(header)
        self._keep_header_controls(header)
        if 'var_status' in labels:
            self._fit_footer(labels['var_status'])

    def _tighten(self, header, status_label):
        """Trim a few pixels so the left column fits 1366x768 and long English fits.

        Header and footer adopt the body's 12 px side margin (core: 14), so the
        title, group headings and status line share one left edge.
        """
        body = header.master.grid_slaves(row=1, column=0)
        side = _box(body[0])[0] if body else 12
        _, top, _, bottom = _box(header)
        header.configure(padding=(side, max(0, top - 2), side, max(0, bottom - 1)))
        if status_label is not None:
            _, top, _, bottom = _box(status_label.master)
            status_label.master.configure(padding=(side, max(0, top - 1), side, max(0, bottom - 2)))
        for widget in _walk(self.root):
            if widget.winfo_class() == 'TLabelframe':
                left, top, right, bottom = _box(widget)
                widget.configure(padding=(left, max(0, top - 2), right, max(0, bottom - 2)))
                if widget.winfo_manager() == 'grid':
                    widget.grid_configure(pady=(0, 6))            # core: (0, 8)
            elif (widget.winfo_class() == 'TButton'
                  and _option(widget, 'command').endswith('on_apply_overlap')):
                widget.configure(width=-6)   # ttk's -11 minimum overflowed the 286 px column
        for widget in header.pack_slaves():
            if widget is getattr(self.root, 'btn_resume', None):
                widget.pack_configure(padx=6)                     # core: 8
            elif widget.winfo_class() == 'TLabel' and not _option(widget, 'style'):
                widget.pack_configure(padx=10)                    # core: 12

    def _align_header_baseline(self, header):
        """Pack centres labels vertically; move the version line onto the title baseline."""
        title, small = self.fonts['title'].metrics(), self.fonts['small'].metrics()
        shift = ((title['ascent'] - title['descent']) - (small['ascent'] - small['descent'])) // 2
        for widget in header.pack_slaves():
            if self.roles.get(widget) == 'small' and shift > 0:
                widget.pack_configure(pady=(2 * shift, 0))

    def _keep_header_controls(self, header):
        """In narrow windows the version line yields first (it is also in the window
        title), so the language toggle and Start scan keep their full width."""
        version = next((w for w in header.pack_slaves() if self.roles.get(w) == 'small'), None)
        title = next((w for w in header.pack_slaves() if _option(w, 'style') == 'Header.TLabel'), None)
        if version is None or title is None:
            return
        info = {('in_' if k == 'in' else k): v for k, v in version.pack_info().items()}

        def refit(event=None):
            if not header.winfo_ismapped():
                return
            # Width needed with the version line shown; independent of the current state.
            need = title.winfo_reqwidth() + version.winfo_reqwidth() + sum(_pair(info['padx']))
            need += sum(w.winfo_reqwidth() + sum(_pair(w.pack_info()['padx']))
                        for w in header.pack_slaves() if w not in (title, version))
            left, _, right, _ = _box(header)
            fits = need <= header.winfo_width() - left - right
            shown = version in header.pack_slaves()
            if fits and not shown:
                version.pack(after=title, **info)
                title.pack_configure(padx=0)
            elif not fits and shown:
                version.pack_forget()
                title.pack_configure(padx=(0, _pair(info['padx'])[0]))
        # Header resizes and caption changes (language switch) both reach these bindings.
        for widget in [header] + header.pack_slaves():
            widget.bind('<Configure>', refit, add='+')
        self.refit_header = refit

    def _fit_footer(self, status_label):
        """Wrap long status text before it can push the progress counter out of view."""
        footer = status_label.master
        left, _, right, _ = _box(footer)
        reserve = self.fonts['numeric'].measure('0000/0000') + 24

        def fit(event):
            if event.widget is footer:
                status_label.configure(wraplength=max(240, event.width - left - right - reserve))
        footer.bind('<Configure>', fit, add='+')

    def style_language_button(self, button):
        """English/中文 toggle: one stable width for both captions, set 10 px apart from
        the scan buttons (L1 packed it flush against Start scan)."""
        button.configure(width=-8)
        if button.winfo_manager() == 'pack':
            button.pack_configure(padx=(0, 10))

    def style_history(self, window):
        """History viewer: monospace raw JSON; status and hint lines as wrapping small text."""
        for widget in _walk(window):
            if isinstance(widget, tk.Text):
                self._assign(widget, 'mono')
            elif widget.winfo_class() == 'TLabel':
                self._assign(widget, 'small', foreground=MUTED)

                def fit(event, label=widget):
                    if event.widget is window:
                        label.configure(wraplength=max(320, event.width - 40))
                window.bind('<Configure>', fit, add='+')


def apply_theme(root):
    theme = Theme(root)
    theme.configure_fonts()
    theme.configure_styles()
    theme.style_main_window()
    return theme

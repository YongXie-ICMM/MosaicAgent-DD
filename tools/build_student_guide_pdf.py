#!/usr/bin/env python3
"""Render docs/STUDENT_GUIDE_zh.md to docs/STUDENT_GUIDE_zh.pdf (A4, Chinese fonts).

The Markdown file is the single source of the student manual; the PDF is what students
open. Rendering: python-markdown -> HTML with the style sheet below -> headless Chromium
(Playwright) -> PDF. Both are optional developer dependencies (``pip install markdown
playwright`` then ``playwright install chromium``); when Playwright is missing the script
writes the HTML next to the PDF instead, so it can be printed to PDF from any browser.

    python tools/build_student_guide_pdf.py            # writes docs/STUDENT_GUIDE_zh.pdf
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "STUDENT_GUIDE_zh.md"
TARGET = REPO / "docs" / "STUDENT_GUIDE_zh.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
html { font-size: 11.5pt; }
body { font-family: "Noto Sans CJK SC", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", "Source Han Sans SC", sans-serif;
       color: #1d1d1b; line-height: 1.65; margin: 0; }
h1 { font-size: 24pt; margin: 0 0 4pt 0; letter-spacing: 0.02em; }
h1 + p { color: #555; margin-top: 0; font-size: 11pt; }
h2 { font-size: 17pt; margin: 22pt 0 8pt 0; padding-bottom: 4pt; border-bottom: 2px solid #14747b; color: #14747b; }
h3 { font-size: 13.5pt; margin: 16pt 0 6pt 0; color: #222; }
p { margin: 5pt 0; }
ol, ul { margin: 4pt 0 6pt 0; padding-left: 1.6em; }
li { margin: 2.5pt 0; }
li > p { margin: 0; }
code { font-family: "DejaVu Sans Mono", "Menlo", "Consolas", monospace; font-size: 9.6pt; background: #f1f3f4;
       border-radius: 3px; padding: 0.5pt 3.5pt; white-space: nowrap; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt 0; font-size: 10.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9ced3; padding: 5pt 7pt; vertical-align: top; text-align: left; }
th { background: #e8f1f2; font-weight: 600; }
th:first-child { width: 36%; }
td code, th code { white-space: normal; }
blockquote { margin: 8pt 0; padding: 7pt 12pt; background: #fff7e6; border-left: 4px solid #e0a52a; color: #5a4300; }
blockquote p { margin: 2pt 0; }
hr { border: 0; height: 0; margin: 0; page-break-after: always; }
strong { font-weight: 700; }
h2, h3 { page-break-after: avoid; }
tr { page-break-inside: avoid; }
"""

FOOTER = ("<div style=\"font-family:'Noto Sans CJK SC',sans-serif;font-size:8.5pt;color:#777;width:100%;"
          "text-align:center;\">MosaicAgent-DD 学生操作说明 · {label} · 第 <span class='pageNumber'></span> / "
          "<span class='totalPages'></span> 页</div>")


def render_html(markdown_text: str) -> str:
    import markdown  # python-markdown
    body = markdown.markdown(markdown_text, extensions=["tables", "sane_lists"], output_format="html5")
    return ("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
            "<title>学生操作说明 · MosaicAgent-DD</title><style>" + CSS + "</style></head><body>" + body + "</body></html>")


def version_label(markdown_text: str) -> str:
    """The '2026-10-05 版' label from the manual's subtitle, or today's date."""
    match = re.search(r"\d{4}-\d{2}-\d{2} 版", markdown_text)
    return match.group(0) if match else date.today().isoformat()


def write_pdf(html: str, target: Path, label: str) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            page.emulate_media(media="print")
            page.pdf(path=str(target), format="A4", print_background=True, prefer_css_page_size=True,
                     display_header_footer=True, header_template="<span></span>",
                     footer_template=FOOTER.format(label=label),
                     margin={"top": "18mm", "right": "16mm", "bottom": "20mm", "left": "16mm"})
        finally:
            browser.close()
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args(argv)
    markdown_text = args.source.read_text(encoding="utf-8")
    try:
        html = render_html(markdown_text)
    except ImportError:
        print("python-markdown is required: pip install markdown", file=sys.stderr)
        return 1
    if write_pdf(html, args.target, version_label(markdown_text)):
        print(f"Wrote {args.target} ({args.target.stat().st_size:,} bytes)")
        return 0
    html_path = args.target.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    print(f"Playwright is not installed; wrote {html_path} instead. Open it in a browser and print to PDF, "
          "or: pip install playwright && playwright install chromium", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

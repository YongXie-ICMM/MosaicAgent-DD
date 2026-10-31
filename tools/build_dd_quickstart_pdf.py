#!/usr/bin/env python3
"""Build the one-page Chinese trial guide; no device/model/network calls."""
from pathlib import Path
import argparse
from html import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-11-01'
BLUE = HexColor('#285d82')
INK = HexColor('#23384a')
MUTED = HexColor('#596e7d')
LINE = HexColor('#dbe4eb')

STEPS = [
    ('01', '显微扫描', [
        '点「打开扫描程序」，在原窗口连接相机、位移台。',
        '先扫 3 × 3，确认照片清楚、方向正确、保存了 9 张。']),
    ('02', '日志交接', [
        '填原图所在文件夹路径，点「检查交接数据与日志」。',
        '选本轮子文件夹，内有照片、session.json、events.jsonl。',
        '看到「记录已核验」再继续；不要填 ZIP 文件路径。']),
    ('03', '自动校色 / 准备拼接', [
        '点「检查原图并准备拼接」→「检查文件夹」。',
        '标定项由老师填写，再点「保存配置并生成命令」。']),
    ('04', '大面积拼接', [
        '点「开始自动校色与拼接」，等待「已核验输出」。',
        '校色默认开启，不用选择；运行中不要关闭启动窗口。']),
    ('05', '图像复核', [
        '点「打开拼接结果」，放大检查缺图、错位和色带。',
        '检查正常后，再进入层数识别。']),
    ('06', '层数识别', [
        '填老师给的本批识别配置 JSON 路径，点「运行层数识别」。',
        '等待「已核验输出」，把层数图与原图对照。']),
    ('07', 'Figure 4 空间图（可选）', [
        '核对本批原生 mask 和有效区，点「生成 Figure 4 空间分布图」。',
        '不用彩色预览图；排除区不确定时，请老师确认。']),
]


def build(output, font):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont('CN', str(font)))
    body = ParagraphStyle('body', fontName='CN', fontSize=10.6, leading=15.5,
                          textColor=INK, wordWrap='CJK', spaceAfter=0)
    small = ParagraphStyle('small', parent=body, fontSize=9.7, leading=14.5, textColor=MUTED)
    c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    c.setTitle('DD 学生试跑极简说明：扫描、拼接与层数识别')
    c.setAuthor('Yong Xie')
    c.setSubject('One-page Chinese trial guide for the local DD workflow')
    w, h = A4
    x, right = 35, w - 35

    def text(value, x0, y, size=11, colour=INK, font_name='CN'):
        c.setFillColor(colour)
        c.setFont(font_name, size)
        c.drawString(x0, y, value)

    def para(value, x0, top, width, style=body):
        p = Paragraph(escape(value), style)
        _, height = p.wrap(width, h)
        p.drawOn(c, x0, top - height)
        return height

    text('DD 学生试跑极简说明', x, h - 47, 21, BLUE)
    text('扫描 → 拼接 → 层数识别', x, h - 71, 12.2)
    text(f'{DATE}  |  统一流程试用版  |  先用 3 × 3 走通，再扩大扫描', x, h - 91, 9.3, MUTED)

    top = h - 106
    c.setFillColor(HexColor('#edf4f9'))
    c.roundRect(x, top - 74, right - x, 74, 7, stroke=0, fill=1)
    text('开始前', x + 12, top - 18, 11.2, BLUE)
    para('老师先备好：运行环境、通信日志版扫描路径、拼接标定和本批识别配置。',
         x + 12, top - 25, right - x - 24, small)
    para('Windows 双击 03_open_workbench.bat；保持启动窗口开启。',
         x + 12, top - 41, right - x - 24, small)
    para('页面地址以本机启动窗口显示为准。沿顶部箭头或「下一步」操作。',
         x + 12, top - 57, right - x - 24, small)

    y = top - 91
    for number, title, lines in STEPS:
        c.setFillColor(BLUE)
        c.roundRect(x, y - 23, 29, 25, 5, stroke=0, fill=1)
        text(number, x + 6, y - 15.4, 12.1, white, 'Helvetica-Bold')
        tx = x + 41
        text(title, tx, y - 12, 11.9, BLUE)
        cursor = y - 20
        for line in lines:
            cursor -= para(line, tx, cursor, right - tx)
        if number == '06':
            cursor -= 4
            lx = tx
            for label, colour in [('1L 单层', '#008000'), ('2L 双层', '#800000'), ('TL 更厚层', '#808000')]:
                c.setFillColor(HexColor(colour))
                c.rect(lx, cursor - 8, 12, 6, stroke=0, fill=1)
                text(label, lx + 17, cursor - 9.5, 9.2, MUTED)
                lx += 112
            cursor -= 16
        c.setStrokeColor(LINE)
        c.setLineWidth(.45)
        c.line(tx, cursor - 7, right, cursor - 7)
        y = cursor - 19

    bottom_top = y + 3
    if bottom_top < 111:
        raise ValueError(f'One-page layout overflow: feedback starts at {bottom_top:.1f}pt')
    c.setFillColor(HexColor('#f4f6f7'))
    c.roundRect(x, bottom_top - 63, right - x, 63, 7, stroke=0, fill=1)
    text('试完交回什么', x + 12, bottom_top - 16, 11, BLUE)
    para('拼接图 + 层数图 + 完成页面截图；原图文件夹和整个 Memory/ 都保留。',
         x + 12, bottom_top - 23, right - x - 24, small)
    para('红色报错：停在本步，点「查看运行日志」，把日志和截图交老师。',
         x + 12, bottom_top - 40, right - x - 24, small)
    text('本地试用；新流程仍待 Windows 仪器电脑完整验证。', x, 25, 8.8, MUTED)
    text('1 / 1', right - 24, 25, 8.4, MUTED, 'Helvetica')
    c.showPage()
    c.save()
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'output/pdf/DD_学生试跑极简说明_20261101.pdf')
    parser.add_argument('--font', type=Path,
                        default=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    args = parser.parse_args()
    build(args.output, args.font)


if __name__ == '__main__':
    main()

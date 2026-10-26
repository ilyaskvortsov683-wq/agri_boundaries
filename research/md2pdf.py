"""Компактный рендер Markdown -> PDF (pandoc JSON AST -> reportlab).

Использование:  python3 research/md2pdf.py report.md report.pdf
Нужны: pandoc, reportlab, шрифты DejaVu (кириллица).
Chromium-печать даёт PDF в 3-4 раза больше, поэтому здесь свой рендер.
"""
import json, re, subprocess, sys
from xml.sax.saxutils import escape
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle, HRFlowable)

rl_config.useA85 = 0  # без ASCII85: файл на четверть меньше
FD = '/usr/share/fonts/truetype/dejavu/'
pdfmetrics.registerFont(TTFont('DV', FD + 'DejaVuSans.ttf'))
# один пропорциональный шрифт + моноширинный только для блоков кода: каждый встроенный шрифт добавляет ~20 КБ;
# жирный заменён цветом и подчёркиванием
pdfmetrics.registerFont(TTFont('DVM', FD + 'DejaVuSansMono.ttf'))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV', italic='DV', boldItalic='DV')

GREEN = colors.HexColor('#1d4e2c')
LINK = '#1a5fa8'
base = ParagraphStyle('base', fontName='DV', fontSize=9.2, leading=12.6, alignment=TA_LEFT,
                      spaceAfter=4, splitLongWords=1)
S = {
    'p': base,
    'h1': ParagraphStyle('h1', base, keepWithNext=1, fontName='DV', fontSize=16, leading=20, textColor=GREEN,
                         spaceBefore=4, spaceAfter=8),
    'h2': ParagraphStyle('h2', base, keepWithNext=1, fontName='DV', fontSize=12.5, leading=16, textColor=GREEN,
                         spaceBefore=12, spaceAfter=5),
    'h3': ParagraphStyle('h3', base, keepWithNext=1, fontName='DV', fontSize=10.5, leading=14,
                         textColor=colors.HexColor('#2b5d3a'), spaceBefore=8, spaceAfter=3),
    'h4': ParagraphStyle('h4', base, keepWithNext=1, fontName='DV', fontSize=9.6, textColor=GREEN, leading=13, spaceBefore=6),
    'cell': ParagraphStyle('cell', base, fontSize=7.4, leading=9.2, spaceAfter=0),
    'hcell': ParagraphStyle('hcell', base, fontName='DV', fontSize=7.4, textColor=GREEN, leading=9.2, spaceAfter=0),
    'code': ParagraphStyle('code', base, fontName='DVM', fontSize=6.9, leading=8.8,
                           backColor=colors.HexColor('#f3f3f3'), borderPadding=4,
                           spaceBefore=3, spaceAfter=7),
    'quote': ParagraphStyle('quote', base, leftIndent=10, textColor=colors.HexColor('#333333'),
                            backColor=colors.HexColor('#f4f8f4'), borderPadding=3),
}


def inl(xs):
    out = []
    for x in xs:
        t, c = x['t'], x.get('c')
        if t == 'Str':
            out.append(escape(c))
        elif t in ('Space', 'SoftBreak'):
            out.append(' ')
        elif t == 'LineBreak':
            out.append('<br/>')
        elif t == 'Strong':
            out.append('<font color="#0b3d1c"><u>' + inl(c) + '</u></font>')
        elif t in ('Emph', 'Underline', 'SmallCaps'):
            out.append(inl(c))
        elif t == 'Strikeout':
            out.append('<strike>' + inl(c) + '</strike>')
        elif t in ('Superscript', 'Subscript'):
            tag = 'super' if t == 'Superscript' else 'sub'
            out.append(f'<{tag}>' + inl(c) + f'</{tag}>')
        elif t == 'Code':
            out.append('<font color="#8a3b12">' + escape(c[1]) + '</font>')
        elif t == 'Link':
            url = escape(c[2][0], {'"': '&quot;'})
            out.append(f'<a href="{url}" color="{LINK}">' + inl(c[1]) + '</a>')
        elif t == 'Quoted':
            q = ('«', '»') if c[0]['t'] == 'DoubleQuote' else ('‘', '’')
            out.append(q[0] + inl(c[1]) + q[1])
        elif t in ('Span', 'Cite'):
            out.append(inl(c[1] if t == 'Span' else c[1]))
        elif t == 'Math':
            out.append(escape(c[1]))
        elif t == 'RawInline':
            out.append(escape(c[1]))
        elif t == 'Note':
            pass
        elif t == 'Image':
            out.append(inl(c[1]))
    return ''.join(out)


def plain_len(blocks):
    s = 0
    for b in blocks:
        if b['t'] in ('Para', 'Plain'):
            s += len(inl(b['c']))
    return s


def cell_text(blocks):
    parts = []
    for b in blocks:
        if b['t'] in ('Para', 'Plain'):
            parts.append(inl(b['c']))
        elif b['t'] in ('BulletList', 'OrderedList'):
            items = b['c'] if b['t'] == 'BulletList' else b['c'][1]
            parts += ['• ' + cell_text(it) for it in items]
    return '<br/>'.join(parts)


def table(c, width):
    head, bodies = c[3], c[4]
    rows, hdr = [], 0
    for r in head[1]:
        rows.append([cell_text(cl[4]) for cl in r[1]]); hdr += 1
    for bd in bodies:
        for r in bd[2] + bd[3]:
            rows.append([cell_text(cl[4]) for cl in r[1]])
    n = max(len(r) for r in rows)
    rows = [r + [''] * (n - len(r)) for r in rows]
    # ширины колонок по объёму текста (sqrt, чтобы не задавить короткие)
    w = []
    for j in range(n):
        lens = [len(r[j]) for r in rows]
        w.append(max(4.0, (sum(lens) / len(lens)) ** 0.5 + 0.25 * min(max(lens), 40) ** 0.5))
    tot = sum(w)
    cw = [width * x / tot for x in w]
    data = [[Paragraph(x, S['hcell'] if i < hdr else S['cell']) for x in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=cw, repeatRows=hdr)
    st = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#b9c7bc')),
          ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3),
          ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    if hdr:
        st.append(('BACKGROUND', (0, 0), (-1, hdr - 1), colors.HexColor('#dbe9de')))
    for i in range(hdr, len(rows)):
        if (i - hdr) % 2:
            st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f6f9f6')))
    t.setStyle(TableStyle(st))
    return [Spacer(1, 3), t, Spacer(1, 7)]


def blocks(bs, width, indent=0):
    out = []
    for b in bs:
        t, c = b['t'], b.get('c')
        if t == 'Header':
            lvl = min(c[0], 4)
            out.append(Paragraph(inl(c[2]), S[f'h{lvl}']))
        elif t in ('Para', 'Plain'):
            st = S['p'] if not indent else ParagraphStyle('pi', S['p'], leftIndent=indent)
            out.append(Paragraph(inl(c), st))
        elif t in ('BulletList', 'OrderedList'):
            items = c if t == 'BulletList' else c[1]
            start = 1 if t == 'BulletList' else c[0][0]
            for k, it in enumerate(items):
                mark = '•' if t == 'BulletList' else f'{start + k}.'
                st = ParagraphStyle('li', S['p'], leftIndent=indent + 12, bulletIndent=indent + 2,
                                    spaceAfter=2)
                first, rest = (it[0], it[1:]) if it and it[0]['t'] in ('Para', 'Plain') else (None, it)
                if first is not None:
                    out.append(Paragraph(inl(first['c']), st, bulletText=mark))
                out += blocks(rest, width - 12, indent + 12)
        elif t == 'CodeBlock':
            out.append(Preformatted(c[1], S['code'], maxLineLength=118, newLineChars='  '))
        elif t == 'BlockQuote':
            for sub in c:
                if sub['t'] in ('Para', 'Plain'):
                    out.append(Paragraph(inl(sub['c']), S['quote']))
                else:
                    out += blocks([sub], width, indent + 10)
        elif t == 'Table':
            out += table(c, width - indent)
        elif t == 'HorizontalRule':
            out.append(HRFlowable(width='100%', color=colors.HexColor('#c9d8cc'), spaceBefore=4,
                                  spaceAfter=6))
        elif t == 'Div':
            out += blocks(c[1], width, indent)
        elif t == 'LineBlock':
            for ln in c:
                out.append(Paragraph(inl(ln), S['p']))
    return out


def page_no(canv, doc):
    canv.saveState()
    canv.setFont('DV', 7.5)
    canv.setFillColor(colors.HexColor('#777777'))
    canv.drawRightString(A4[0] - 14 * mm, 8 * mm, str(doc.page))
    canv.restoreState()


def main(src, dst):
    ast = json.loads(subprocess.check_output(['pandoc', src, '-f', 'gfm', '-t', 'json']))
    h1 = next((b for b in ast['blocks'] if b['t'] == 'Header' and b['c'][0] == 1), None)   # заголовок PDF = первый «# …»
    title = re.sub(r'<[^>]+>', '', inl(h1['c'][2])) if h1 else 'agri_boundaries'
    doc = SimpleDocTemplate(dst, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title=title, author='agri_boundaries',
                            pageCompression=1)
    story = blocks(ast['blocks'], doc.width)
    doc.build(story, onFirstPage=page_no, onLaterPages=page_no)
    try:  # необязательно: object streams сжимают аннотации-ссылки (~25 % размера)
        import os, pikepdf
        tmp = dst + '.tmp'
        with pikepdf.open(dst) as pdf:
            pdf.save(tmp, object_stream_mode=pikepdf.ObjectStreamMode.generate,
                     recompress_flate=True)
        os.replace(tmp, dst)
    except ImportError:
        pass


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

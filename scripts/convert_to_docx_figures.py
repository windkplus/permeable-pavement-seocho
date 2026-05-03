"""
convert_to_docx_figures.py
draft_v1.md → draft_v1_figures.docx 변환 (참조용)
Figure Captions 섹션에 실제 이미지 삽입 버전
"""

import os
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH    = os.path.join(BASE_DIR, 'paper/draft_v1.md')
OUT_PATH   = os.path.join(BASE_DIR, 'paper/draft_v1_figures.docx')
FIG_DIR    = os.path.join(BASE_DIR, 'results/figures')

# Figure Captions 섹션에서 파일명 추출: *(File: xxx.png)*
FIG_FILE_RE = re.compile(r'\*\(File:\s*([^)]+\.png)\)\*')

# ── 문서 생성 및 기본 설정 ─────────────────────────────────────────────────────
doc = Document()

for section in doc.sections:
    section.top_margin    = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin   = Cm(2.54)
    section.right_margin  = Cm(2.54)

# ── 스타일 설정 함수 ───────────────────────────────────────────────────────────
def set_font(run, size=12, bold=False, italic=False, color=None):
    run.font.name        = 'Times New Roman'
    run.font.size        = Pt(size)
    run.font.bold        = bold
    run.font.italic      = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    rpr = run._r.get_or_add_rPr()
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'),    'Times New Roman')
    rFonts.set(qn('w:hAnsi'),    'Times New Roman')
    rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    rpr.insert(0, rFonts)

def set_para_spacing(para, before=0, after=6, line_spacing=2.0):
    pPr = para._p.get_or_add_pPr()
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:before'),   str(int(before * 20)))
    spacing.set(qn('w:after'),    str(int(after  * 20)))
    spacing.set(qn('w:line'),     str(int(line_spacing * 240)))
    spacing.set(qn('w:lineRule'), 'auto')
    pPr.append(spacing)

def add_heading(doc, text, level):
    sizes = {1: 16, 2: 14, 3: 13}
    para  = doc.add_paragraph()
    run   = para.add_run(text)
    set_font(run, size=sizes.get(level, 12), bold=True)
    set_para_spacing(para, before=12, after=6, line_spacing=1.5)
    return para

def add_body(doc, text, italic=False):
    para    = doc.add_paragraph()
    pattern = re.compile(r'\*\*(.+?)\*\*|\*(.+?)\*')
    last    = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            run = para.add_run(text[last:m.start()])
            set_font(run, italic=italic)
        if m.group(1):
            run = para.add_run(m.group(1))
            set_font(run, bold=True)
        elif m.group(2):
            run = para.add_run(m.group(2))
            set_font(run, italic=True)
        last = m.end()
    if last < len(text):
        run = para.add_run(text[last:])
        set_font(run, italic=italic)
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_para_spacing(para, before=0, after=6, line_spacing=2.0)
    return para

def add_table_from_md(doc, rows):
    data_rows = [r for r in rows if not re.match(r'^\|[-| :]+\|$', r.strip())]
    if not data_rows:
        return
    cols   = [c.strip() for c in data_rows[0].strip('|').split('|')]
    n_cols = len(cols)
    table  = doc.add_table(rows=len(data_rows), cols=n_cols)
    table.style     = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row_str in enumerate(data_rows):
        cells = [c.strip() for c in row_str.strip('|').split('|')]
        for j, cell_text in enumerate(cells[:n_cols]):
            cell = table.cell(i, j)
            para = cell.paragraphs[0]
            run  = para.add_run(cell_text)
            set_font(run, size=10, bold=(i == 0))
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

def insert_figure(doc, filename):
    """그림 파일을 문서에 삽입 (중앙 정렬, 최대 너비 14cm)"""
    fig_path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(fig_path):
        para = doc.add_paragraph()
        run  = para.add_run(f'[이미지 없음: {filename}]')
        set_font(run, italic=True, color=(200, 0, 0))
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run  = para.add_run()
    run.add_picture(fig_path, width=Cm(14))
    set_para_spacing(para, before=6, after=12, line_spacing=1.0)

# ── MD 파싱 및 변환 ────────────────────────────────────────────────────────────
with open(MD_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()

i         = 0
table_buf = []
in_fig_captions = False   # Figure Captions 섹션 내부 여부

while i < len(lines):
    line = lines[i].rstrip('\n')

    # 테이블 수집
    if line.strip().startswith('|'):
        table_buf.append(line.strip())
        i += 1
        continue
    else:
        if table_buf:
            add_table_from_md(doc, table_buf)
            table_buf = []

    # 빈 줄
    if not line.strip():
        i += 1
        continue

    # 구분선
    if line.strip().startswith('---'):
        doc.add_paragraph()
        i += 1
        continue

    # H2 섹션 감지 → Figure Captions 여부 추적
    if line.startswith('## '):
        section_title = line[3:].strip()
        in_fig_captions = ('Figure Caption' in section_title)
        add_heading(doc, section_title, level=2)
        i += 1
        continue

    # H1
    if line.startswith('# ') and not line.startswith('## '):
        para = doc.add_paragraph()
        run  = para.add_run(line[2:].strip())
        set_font(run, size=18, bold=True)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(para, before=0, after=12, line_spacing=1.5)

    # 저자/소속
    elif line.startswith('**') and line.endswith('**') and len(line) < 80:
        para = doc.add_paragraph()
        run  = para.add_run(line.strip('*'))
        set_font(run, size=12, bold=True)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(para, before=0, after=4, line_spacing=1.2)

    # H3
    elif line.startswith('### '):
        add_heading(doc, line[4:].strip(), level=3)

    # Keywords
    elif line.startswith('**Keywords'):
        para     = doc.add_paragraph()
        kw_text  = line.replace('**Keywords:**', '').strip()
        run_label = para.add_run('Keywords: ')
        set_font(run_label, bold=True)
        run_kw   = para.add_run(kw_text)
        set_font(run_kw, italic=True)
        set_para_spacing(para, before=0, after=6, line_spacing=2.0)

    # 이탤릭 단락
    elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
        add_body(doc, line.strip('*'), italic=True)

    # 글머리 항목
    elif line.startswith('- ') or re.match(r'^\d+\.\s', line):
        text = re.sub(r'^[-\d]+[.)]\s*', '', line)
        para = doc.add_paragraph(style='List Bullet')
        run  = para.add_run(text)
        set_font(run)
        set_para_spacing(para, before=0, after=3, line_spacing=1.5)

    # 일반 본문 (Figure Captions 섹션이면 캡션 뒤에 이미지 삽입)
    else:
        if line.strip():
            # Figure Captions 섹션에서 *(File: xxx.png)* 탐지 → 그림 먼저, 캡션 나중
            if in_fig_captions:
                m = FIG_FILE_RE.search(line)
                if m:
                    insert_figure(doc, m.group(1).strip())
            add_body(doc, line.strip())

    i += 1

# 남은 테이블
if table_buf:
    add_table_from_md(doc, table_buf)

# ── 저장 ──────────────────────────────────────────────────────────────────────
doc.save(OUT_PATH)
print(f"변환 완료 → {OUT_PATH}")

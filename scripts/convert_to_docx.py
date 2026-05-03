"""
convert_to_docx.py
draft_v1.md → draft_v1.docx 변환
논문 형식 스타일 적용 (Times New Roman, 12pt, 더블스페이싱)
"""

import os
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH   = os.path.join(BASE_DIR, 'paper/draft_v1.md')
OUT_PATH  = os.path.join(BASE_DIR, 'paper/draft_v1.docx')

# ── 문서 생성 및 기본 설정 ─────────────────────────────────────────────────────
doc = Document()

# 페이지 여백 (상하좌우 2.54cm = 1인치)
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
    # 동아시아 폰트도 설정
    rpr = run._r.get_or_add_rPr()
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'),    'Times New Roman')
    rFonts.set(qn('w:hAnsi'),    'Times New Roman')
    rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    rpr.insert(0, rFonts)

def set_para_spacing(para, before=0, after=6, line_spacing=2.0):
    """더블 스페이싱 설정"""
    from docx.shared import Pt
    from docx.oxml import OxmlElement
    pPr = para._p.get_or_add_pPr()
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:before'), str(int(before * 20)))
    spacing.set(qn('w:after'),  str(int(after  * 20)))
    spacing.set(qn('w:line'),   str(int(line_spacing * 240)))
    spacing.set(qn('w:lineRule'), 'auto')
    pPr.append(spacing)

def add_heading(doc, text, level):
    sizes   = {1: 16, 2: 14, 3: 13}
    para    = doc.add_paragraph()
    run     = para.add_run(text)
    set_font(run, size=sizes.get(level, 12), bold=True)
    set_para_spacing(para, before=12, after=6, line_spacing=1.5)
    return para

def add_body(doc, text, italic=False):
    para = doc.add_paragraph()
    # **bold** 와 *italic* 처리
    pattern = re.compile(r'\*\*(.+?)\*\*|\*(.+?)\*')
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            run = para.add_run(text[last:m.start()])
            set_font(run, italic=italic)
        if m.group(1):   # **bold**
            run = para.add_run(m.group(1))
            set_font(run, bold=True)
        elif m.group(2): # *italic*
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
    """마크다운 테이블 → Word 테이블 변환"""
    data_rows = [r for r in rows if not re.match(r'^\|[-| :]+\|$', r.strip())]
    if not data_rows:
        return
    cols = [c.strip() for c in data_rows[0].strip('|').split('|')]
    n_cols = len(cols)
    table = doc.add_table(rows=len(data_rows), cols=n_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, row_str in enumerate(data_rows):
        cells = [c.strip() for c in row_str.strip('|').split('|')]
        for j, cell_text in enumerate(cells[:n_cols]):
            cell = table.cell(i, j)
            para = cell.paragraphs[0]
            run  = para.add_run(cell_text)
            set_font(run, size=10, bold=(i == 0))
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()   # 테이블 뒤 공백

# ── MD 파싱 및 변환 ────────────────────────────────────────────────────────────
with open(MD_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()

i = 0
table_buf = []

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

    # 제목 (H1)
    if line.startswith('# ') and not line.startswith('## '):
        para = doc.add_paragraph()
        run  = para.add_run(line[2:].strip())
        set_font(run, size=18, bold=True)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(para, before=0, after=12, line_spacing=1.5)

    # 저자/소속 (# 다음 ** 줄들)
    elif line.startswith('**') and line.endswith('**') and len(line) < 80:
        para = doc.add_paragraph()
        run  = para.add_run(line.strip('*'))
        set_font(run, size=12, bold=True)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para_spacing(para, before=0, after=4, line_spacing=1.2)

    # H2
    elif line.startswith('## '):
        add_heading(doc, line[3:].strip(), level=2)

    # H3
    elif line.startswith('### '):
        add_heading(doc, line[4:].strip(), level=3)

    # Keywords / 이탤릭 단락
    elif line.startswith('**Keywords'):
        para = doc.add_paragraph()
        kw_text = line.replace('**Keywords:**', '').strip()
        run_label = para.add_run('Keywords: ')
        set_font(run_label, bold=True)
        run_kw = para.add_run(kw_text)
        set_font(run_kw, italic=True)
        set_para_spacing(para, before=0, after=6, line_spacing=2.0)

    # 이탤릭 단락 (*Manuscript...*)
    elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
        add_body(doc, line.strip('*'), italic=True)

    # 글머리 항목
    elif line.startswith('- ') or re.match(r'^\d+\.\s', line):
        text = re.sub(r'^[-\d]+[.)]\s*', '', line)
        para = doc.add_paragraph(style='List Bullet')
        run  = para.add_run(text)
        set_font(run)
        set_para_spacing(para, before=0, after=3, line_spacing=1.5)

    # 일반 본문
    else:
        if line.strip():
            add_body(doc, line.strip())

    i += 1

# 남은 테이블
if table_buf:
    add_table_from_md(doc, table_buf)

# ── 저장 ──────────────────────────────────────────────────────────────────────
doc.save(OUT_PATH)
print(f"변환 완료 → {OUT_PATH}")

# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown>=3.6", "python-docx>=1.1"]
# ///
"""Build report/report.docx and report/report.pdf from report/REPORT_DRAFT.md.

    uv run report/build.py            # both (PDF needs Google Chrome or Chromium)
    uv run report/build.py --docx     # Word only

The Markdown file is the single source; images are referenced relative to report/.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (kept for manual tweaks)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC = HERE / "REPORT_DRAFT.md"
CHROME = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chromium-browser"]

CSS = """
@page { size: A4; margin: 14mm 14mm 14mm 14mm; }
body { font: 9.6pt/1.36 "Times New Roman", Times, serif; color: #111; }
h1 { font-size: 17pt; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 14pt 0 4pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; }
p, li { margin: 3pt 0; text-align: justify; }
ul, ol { margin: 3pt 0 3pt 16pt; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 7pt; font-size: 7.8pt; line-height: 1.25; page-break-inside: avoid; }
th, td { border: 0.6pt solid #888; padding: 2pt 3pt; text-align: left; vertical-align: top; }
th { background: #e9eef0; }
img { display: block; max-width: 100%; max-height: 64mm; margin: 6pt auto 2pt; }
p.cap { text-align: center; font-size: 8.5pt; color: #333; margin: 0 0 8pt; }
code { font: 8.5pt Menlo, monospace; }
"""


def md_text() -> str:
    return SRC.read_text(encoding="utf-8")


# ------------------------------------------------------------------ PDF (HTML -> Chrome)
def build_pdf(out: Path) -> None:
    html = markdown.markdown(md_text(), extensions=["tables"])
    # image alt text becomes a caption under the figure
    html = re.sub(r'<p>(<img alt="([^"]*)" src="([^"]*)"\s*/?>)</p>',
                  lambda m: f'<p>{m.group(1)}</p><p class="cap">{m.group(2)}</p>', html)
    doc = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{html}</body></html>'
    chrome = next((c for c in CHROME if Path(c).exists() or shutil.which(c)), None)
    if chrome is None:
        sys.exit("Chrome/Chromium not found - build the PDF by exporting report.docx from Word.")
    with tempfile.TemporaryDirectory() as tmp:
        page = HERE / "_report_print.html"           # in report/ so relative image paths resolve
        page.write_text(doc, encoding="utf-8")
        try:
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            f"--user-data-dir={tmp}", f"--print-to-pdf={out}", page.as_uri()],
                           check=True, capture_output=True, timeout=120)
        finally:
            page.unlink(missing_ok=True)
    print(f"wrote {out}")


# ------------------------------------------------------------------ DOCX
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)")


def add_runs(par, text: str, size: float | None = None, bold: bool = False) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            run = par.add_run(part[2:-2]); run.bold = True
        elif part.startswith("`"):
            run = par.add_run(part[1:-1]); run.font.name = "Menlo"; run.font.size = Pt(8.5)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = par.add_run(part[1:-1]); run.italic = True
        else:
            run = par.add_run(part)
        if bold:
            run.bold = True
        if size and not part.startswith("`"):
            run.font.size = Pt(size)


def shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc: Document, rows: list[list[str]]) -> None:
    header, body = rows[0], rows[2:]                       # rows[1] is the --- separator
    t = doc.add_table(rows=1 + len(body), cols=len(header))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    size = 7.5 if len(header) > 7 else 8.5
    for r, cells in enumerate([header, *body]):
        for c, txt in enumerate(cells[: len(header)]):
            cell = t.cell(r, c)
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            add_runs(cell.paragraphs[0], txt.strip(), size=size, bold=(r == 0))
            if r == 0:
                shade(cell, "E9EEF0")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build_docx(out: Path) -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.8); sec.top_margin = sec.bottom_margin = Cm(1.7)
    normal = doc.styles["Normal"]; normal.font.name = "Times New Roman"; normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(3)
    for lvl, sz in ((1, 17), (2, 13), (3, 11)):
        st = doc.styles[f"Heading {lvl}"]; st.font.size = Pt(sz); st.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
        st.font.name = "Times New Roman"

    lines = md_text().splitlines()
    i, para = 0, []

    def flush() -> None:
        if para:
            p = doc.add_paragraph(); add_runs(p, " ".join(s.strip() for s in para))
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para.clear()

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            flush(); i += 1; continue
        if m := re.match(r"^(#{1,3}) (.*)", ln):
            flush(); doc.add_heading(m.group(2).strip(), level=len(m.group(1))); i += 1; continue
        if ln.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i].strip().strip("|").split("|")); i += 1
            add_table(doc, rows); continue
        if m := re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", ln):
            flush()
            doc.add_picture(str((HERE / m.group(2)).resolve()), width=Cm(16.5 if "fig1" in m.group(2) or "viewer" in m.group(2) else 14.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(); add_runs(cap, m.group(1), size=8.5); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1; continue
        if m := re.match(r"^(\s*)([-*]|\d+\.) (.*)", ln):
            flush()
            text = m.group(3); i += 1
            while i < len(lines) and lines[i].startswith("   ") and not re.match(r"^\s*([-*]|\d+\.) ", lines[i]):
                text += " " + lines[i].strip(); i += 1
            style = "List Number" if m.group(2)[0].isdigit() else "List Bullet"
            p = doc.add_paragraph(style=style); add_runs(p, text)
            continue
        para.append(ln); i += 1
    flush()
    doc.save(out)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", action="store_true", help="only build the Word file")
    a = ap.parse_args()
    build_docx(HERE / "report.docx")
    if not a.docx:
        build_pdf(HERE / "report.pdf")


if __name__ == "__main__":
    main()

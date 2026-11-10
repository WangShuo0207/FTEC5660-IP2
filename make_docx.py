"""Convert report.md to a clean .docx for Blackboard submission."""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches

src = Path("report.md").read_text(encoding="utf-8")
doc = Document()

# Set base font
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

for line in src.splitlines():
    line = line.rstrip()
    if not line:
        doc.add_paragraph("")
        continue
    if line.startswith("# "):
        doc.add_heading(line[2:], level=1)
    elif line.startswith("## "):
        doc.add_heading(line[3:], level=2)
    elif line.startswith("### "):
        doc.add_heading(line[4:], level=3)
    elif line.startswith("- "):
        p = doc.add_paragraph(line[2:], style="List Bullet")
    elif re.match(r"^\d+\.\s", line):
        doc.add_paragraph(re.sub(r"^\d+\.\s", "", line), style="List Number")
    elif line.startswith("|"):
        # table row - collect
        # We'll just add as a paragraph for simplicity (tables are minor)
        p = doc.add_paragraph(line)
        p.paragraph_format.left_indent = Inches(0.2)
    elif line.startswith("---"):
        doc.add_paragraph("")
    else:
        # strip markdown bold/italic
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        text = re.sub(r"`(.+?)`", r"\1", text)
        doc.add_paragraph(text)

out = "WANGShuo_IP2_Reproducibility.docx"
doc.save(out)
print(f"wrote {out}")

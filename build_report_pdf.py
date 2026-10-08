from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).parent
SOURCE = ROOT / "Wine_Quality_Regression_Report.md"
OUTPUT = ROOT / "Wine_Quality_Regression_Report.pdf"
FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CJKTitle", parent=styles["Title"], fontName=FONT, fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="CJKHeading", parent=styles["Heading2"], fontName=FONT, fontSize=13, leading=18, textColor=colors.HexColor("#1f4e79"), spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name="CJKBody", parent=styles["BodyText"], fontName=FONT, fontSize=9.2, leading=14, spaceAfter=5))
styles.add(ParagraphStyle(name="CJKSmall", parent=styles["BodyText"], fontName=FONT, fontSize=7.5, leading=10))


def clean_inline(value: str) -> str:
    value = re.sub(r"!\[[^]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = value.replace("**", "").replace("`", "")
    return html.escape(value.strip())


def table_flow(rows):
    data = [[Paragraph(clean_inline(cell), styles["CJKSmall"]) for cell in row] for row in rows]
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


lines = SOURCE.read_text(encoding="utf-8").splitlines()
story = []
in_code = False
table_rows = []
for line in lines:
    if line.strip() == "<!-- PAGEBREAK -->":
        if table_rows:
            story.append(table_flow(table_rows))
            story.append(Spacer(1, 0.25 * cm))
            table_rows = []
        story.append(PageBreak())
        continue
    if line.strip().startswith("```"):
        in_code = not in_code
        continue
    if in_code:
        continue
    if line.startswith("|"):
        if "---" not in line:
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
        continue
    if table_rows:
        story.append(table_flow(table_rows))
        story.append(Spacer(1, 0.25 * cm))
        table_rows = []
    if line.startswith("# "):
        story.append(Paragraph(clean_inline(line[2:]), styles["CJKTitle"]))
    elif line.startswith("## "):
        story.append(Paragraph(clean_inline(line[3:]), styles["CJKHeading"]))
    elif line.startswith("### "):
        story.append(Paragraph(clean_inline(line[4:]), styles["CJKHeading"]))
    elif line.startswith(">"):
        story.append(Paragraph(clean_inline(line.lstrip("> ")), styles["CJKSmall"]))
    elif line.startswith("!["):
        match = re.search(r"\(([^)]+)\)", line)
        if match:
            image_path = ROOT / match.group(1)
            if image_path.exists():
                image = Image(str(image_path))
                image._restrictSize(17 * cm, 10 * cm)
                story.append(image)
                story.append(Spacer(1, 0.2 * cm))
    elif line.startswith("- ") or re.match(r"\d+\. ", line):
        story.append(Paragraph("．" + clean_inline(re.sub(r"^(?:- |\d+\. )", "", line)), styles["CJKBody"]))
    elif line.strip():
        story.append(Paragraph(clean_inline(line), styles["CJKBody"]))
if table_rows:
    story.append(table_flow(table_rows))

doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Wine Quality Regression Report")
doc.build(story)
print(f"Created: {OUTPUT}")

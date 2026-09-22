"""최종 Markdown을 제출 가능한 PDF로 변환한다."""

from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path


class PdfExportError(RuntimeError):
    """PDF 생성 환경 또는 출력에 문제가 있을 때 발생한다."""


_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf"),
)


def _font_path() -> Path:
    configured = os.environ.get("REPORT_PDF_FONT")
    candidates = (Path(configured),) + _FONT_CANDIDATES if configured else _FONT_CANDIDATES
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise PdfExportError(
        "한글 PDF 폰트를 찾지 못했습니다. REPORT_PDF_FONT에 TTF/TTC 경로를 지정하세요."
    )


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _paragraph(text: str, style):
    from reportlab.platypus import Paragraph

    # Markdown 링크는 PDF에서 URL이 보이도록 단순화하고, 나머지는 XML 안전 문자열로 만든다.
    text = re.sub(r"\[([^]]+)]\(([^)]+)\)", r"\1 (\2)", text)
    return Paragraph(escape(text), style)


def _markdown_story(markdown: str, styles: dict, page_width: float) -> list:
    from reportlab.lib import colors
    from reportlab.platypus import KeepTogether, Spacer, Table, TableStyle

    story: list = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index].rstrip()
        if not raw:
            story.append(Spacer(1, 4))
            index += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", raw)
        if heading:
            level = len(heading.group(1))
            story.append(_paragraph(heading.group(2), styles[f"h{level}"]))
            index += 1
            continue

        if raw.startswith("|") and index + 1 < len(lines) and _is_table_separator(lines[index + 1]):
            table_lines = [raw]
            index += 2
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = [
                [_paragraph(cell, styles["table_header"] if row_index == 0 else styles["table"])
                 for cell in _table_cells(line)]
                for row_index, line in enumerate(table_lines)
            ]
            column_count = max(len(row) for row in rows)
            for row in rows:
                row.extend([_paragraph("", styles["table"])] * (column_count - len(row)))
            table = Table(
                rows,
                colWidths=[page_width / column_count] * column_count,
                repeatRows=1,
                hAlign="LEFT",
            )
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEB9C8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(KeepTogether([table, Spacer(1, 9)]))
            continue

        bullet = re.match(r"^\s*-\s+(.+)$", raw)
        if bullet:
            story.append(_paragraph("• " + bullet.group(1), styles["bullet"]))
        else:
            story.append(_paragraph(raw, styles["body"]))
        index += 1
    return story


def export_markdown_pdf(
    markdown: str,
    output_path: str | Path,
    *,
    title: str = "기술 비교 평가 보고서",
    subtitle: str = "",
) -> Path:
    """Markdown 보고서를 A4 PDF로 저장하고 절대 경로를 반환한다."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise PdfExportError(
            "PDF 출력에는 reportlab이 필요합니다. "
            "agents/report/requirements.txt를 설치하세요."
        ) from exc

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    font_name = "ReportKorean"
    try:
        pdfmetrics.getFont(font_name)
    except KeyError:
        pdfmetrics.registerFont(TTFont(font_name, str(_font_path())))

    navy = colors.HexColor("#17365D")
    muted = colors.HexColor("#5D6B7A")
    styles = {
        "cover": ParagraphStyle(
            "ReportCover", fontName=font_name, fontSize=24, leading=34,
            textColor=navy, alignment=TA_CENTER, spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", fontName=font_name, fontSize=11, leading=18,
            textColor=muted, alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "ReportH1", fontName=font_name, fontSize=16, leading=23,
            textColor=navy, spaceBefore=16, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "ReportH2", fontName=font_name, fontSize=13, leading=19,
            textColor=colors.HexColor("#28507A"), spaceBefore=12, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "ReportH3", fontName=font_name, fontSize=11, leading=17,
            textColor=navy, spaceBefore=9, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "ReportBody", fontName=font_name, fontSize=9.2, leading=15,
            textColor=colors.HexColor("#202833"), wordWrap="CJK", spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet", fontName=font_name, fontSize=9.2, leading=15,
            textColor=colors.HexColor("#202833"), wordWrap="CJK",
            leftIndent=11, firstLineIndent=-8, spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "ReportTable", fontName=font_name, fontSize=7.4, leading=10.5,
            textColor=colors.HexColor("#202833"), wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "ReportTableHeader", fontName=font_name, fontSize=7.7, leading=11,
            textColor=navy, wordWrap="CJK",
        ),
    }

    document = SimpleDocTemplate(
        str(target), pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=19 * mm, bottomMargin=18 * mm,
        title=title, author="Report Agent",
    )

    def first_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(navy)
        canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, fill=1, stroke=0)
        canvas.restoreState()

    def later_pages(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 7.5)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, A4[1] - 11 * mm, title[:54])
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
        canvas.setStrokeColor(colors.HexColor("#D8DEE8"))
        canvas.line(18 * mm, A4[1] - 13 * mm, A4[0] - 18 * mm, A4[1] - 13 * mm)
        canvas.restoreState()

    story = [Spacer(1, 54 * mm), _paragraph(title, styles["cover"])]
    if subtitle:
        story.append(_paragraph(subtitle, styles["subtitle"]))
    story.extend([
        Spacer(1, 12 * mm),
        _paragraph("구조화된 평가 결과와 확인된 근거를 바탕으로 작성", styles["subtitle"]),
        PageBreak(),
    ])
    story.extend(_markdown_story(markdown, styles, document.width))
    document.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    if not target.is_file() or target.stat().st_size == 0:
        raise PdfExportError(f"PDF 파일이 생성되지 않았습니다: {target}")
    return target

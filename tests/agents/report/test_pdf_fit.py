"""보고서 PDF 분량 조정 테스트: 최대 10장 (과제 요건)."""
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("reportlab")

from agents.report.pdf import MAX_REPORT_PAGES, export_markdown_pdf_fit  # noqa: E402


def markdown(paragraphs: int) -> str:
    body = "\n\n".join(f"문단 {i}. " + "KV cache 최적화 기술의 관점별 평가 내용을 설명하는 문장이다. " * 6 for i in range(paragraphs))
    return f"# SUMMARY\n\n요약\n\n# 1. 본문\n\n{body}\n\n# REFERENCE\n\n- 출처\n"


def test_short_report_keeps_cover_and_default_size():
    with tempfile.TemporaryDirectory() as tmp:
        result = export_markdown_pdf_fit(markdown(10), Path(tmp) / "r.pdf")
        assert result.within_limit and result.cover_page and result.scale == 1.0
        assert result.pages <= MAX_REPORT_PAGES


def test_long_report_is_compacted_to_fit_without_cutting_content():
    with tempfile.TemporaryDirectory() as tmp:
        base = export_markdown_pdf_fit(markdown(110), Path(tmp) / "a.pdf", max_pages=999)
        assert base.pages > MAX_REPORT_PAGES  # 기본 모양으로는 초과하는 분량
        fit = export_markdown_pdf_fit(markdown(110), Path(tmp) / "b.pdf")
        assert fit.within_limit and fit.pages <= MAX_REPORT_PAGES
        assert not fit.cover_page  # 표지를 없애거나 글자를 줄여서 맞췄다


def test_too_long_report_is_reported_not_silently_truncated():
    with tempfile.TemporaryDirectory() as tmp:
        result = export_markdown_pdf_fit(markdown(300), Path(tmp) / "c.pdf")
        assert not result.within_limit and result.pages > MAX_REPORT_PAGES
        assert result.scale == 0.88  # 가장 작은 단계까지 시도한 뒤 초과로 보고

"""원문 본문 수집 (Pool B — 런타임 수집 코퍼스).

웹 검색 스니펫만으로는 채택·생태계·반대 근거를 제대로 판단하기 어렵다. 짧은 문서는
통째로 근거 후보가 되고, 긴 문서는 index.py에서 청킹·색인한 뒤 질의에 맞는 조각만
근거 후보로 올린다.

도메인 에이전트(agents/domain/rag/fetch.py)와 같은 설계를 시장 에이전트에 맞춰 다시
구현한다("다른 에이전트 폴더를 import하지 않는다"는 팀 규칙 때문에 공유하지 않는다).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agents.market.rag.evidence import normalize_text

# 이 길이 이하면 통째로 써도 프롬프트에 부담이 없다. 넘으면 청킹·색인 대상으로 보낸다.
SHORT_DOCUMENT_CHARS = 8000
# 한 페이지 분량의 근사치. 웹 문서는 페이지 개념이 없어 200p 한도 계산에 이 값을 쓴다.
CHARS_PER_PAGE = 3000
PAGE_BUDGET = 200

_SCRIPT_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")
# pdfplumber 기본값(3)은 arXiv 논문에서 단어를 붙여 버린다(도메인 에이전트가 겪은 문제와 동일).
PDF_X_TOLERANCE = 1.5


@dataclass
class DocumentPart:
    text: str
    locator: str  # "p.7" 또는 "chars:0-3000"


@dataclass
class FetchedDocument:
    url: str
    title: str
    parts: list[DocumentPart] = field(default_factory=list)
    page_count: int = 0
    is_long: bool = False
    error: str | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.parts)


def _extract_html_text(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_SCRIPT_TAGS):
        tag.decompose()
    title = normalize_text(soup.title.get_text()) if soup.title else ""
    blocks = [normalize_text(el.get_text(" ")) for el in soup.find_all(["p", "li", "h1", "h2", "h3"])]
    text = "\n".join(b for b in blocks if len(b) > 30)
    return title, text or normalize_text(soup.get_text(" "))


def _split_by_chars(text: str, size: int = CHARS_PER_PAGE) -> list[DocumentPart]:
    return [
        DocumentPart(text=text[i : i + size], locator=f"chars:{i}-{min(i + size, len(text))}")
        for i in range(0, len(text), size)
    ]


def _extract_pdf_parts(data: bytes) -> tuple[list[DocumentPart], int]:
    import pdfplumber

    parts: list[DocumentPart] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        total = len(pdf.pages)
        for index, page in enumerate(pdf.pages, start=1):
            text = normalize_text(page.extract_text(x_tolerance=PDF_X_TOLERANCE) or "")
            if text:
                parts.append(DocumentPart(text=text, locator=f"p.{index}"))
    return parts, total


def fetch_document(url: str, *, timeout_seconds: float = 20.0, cache_dir: Path | None = None) -> FetchedDocument:
    """URL 하나를 가져와 본문을 파싱한다. 실패해도 예외 대신 error에 담아 돌려준다."""
    cache_path = None
    if cache_dir is not None:
        from agents.market.rag.evidence import content_checksum

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (content_checksum(url)[:16] + ".bin")

    try:
        if cache_path is not None and cache_path.exists():
            payload, content_type = cache_path.read_bytes(), ""
        else:
            with httpx.Client(
                timeout=timeout_seconds, follow_redirects=True,
                headers={"User-Agent": "kvcache-market-agent/1.0 (research)"},
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.content
                content_type = response.headers.get("content-type", "")
            if cache_path is not None:
                cache_path.write_bytes(payload)
    except Exception as exc:
        return FetchedDocument(url=url, title=url, error=f"{type(exc).__name__}: {exc}")

    try:
        is_pdf = payload[:5] == b"%PDF-" or "application/pdf" in content_type.lower()
        if is_pdf:
            parts, page_count = _extract_pdf_parts(payload)
            title = url.rsplit("/", 1)[-1]
        else:
            title, text = _extract_html_text(payload.decode("utf-8", errors="replace"))
            parts = _split_by_chars(text)
            page_count = max(1, (len(text) + CHARS_PER_PAGE - 1) // CHARS_PER_PAGE)
    except Exception as exc:
        return FetchedDocument(url=url, title=url, error=f"파싱 실패: {type(exc).__name__}: {exc}")

    if not parts:
        return FetchedDocument(url=url, title=title or url, error="본문을 추출하지 못함")

    full = "\n\n".join(p.text for p in parts)
    return FetchedDocument(
        url=url, title=title or url, parts=parts, page_count=page_count,
        is_long=len(full) > SHORT_DOCUMENT_CHARS,
    )

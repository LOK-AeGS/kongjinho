"""원문 수집: 검색 스니펫을 넘어 실제 본문을 가져온다.

스니펫만으로는 도메인 평가가 성립하지 않는다. "처리량 1.80배"는 스니펫에 나오지만
그 수치가 어떤 모델·문맥 길이·배치에서 측정됐는지는 본문에만 있다.
적용 조건을 못 읽으면 실험 환경과 데이터센터 환경의 차이를 판단할 수 없다.

길이에 따라 처리가 갈린다. 짧은 문서는 통째로 근거가 되고, 긴 문서(논문 등)는
청킹·색인을 거쳐 질문에 맞는 부분만 근거로 승격한다. 임베딩이 필요한 지점은 후자뿐이다.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from agents.domain.tools.evidence import content_checksum, normalize_text

# 이 길이 이하면 통째로 써도 프롬프트에 부담이 없다. 넘으면 색인 대상으로 보낸다.
SHORT_DOCUMENT_CHARS = 8000
# 한 페이지 분량의 근사치. 웹 문서는 페이지 개념이 없어 200p 한도 계산에 이 값을 쓴다.
CHARS_PER_PAGE = 3000

_SCRIPT_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")


@dataclass
class DocumentPart:
    """본문 조각 1개. locator 는 근거 검증에서 위치를 특정하는 데 쓰인다."""

    text: str
    locator: str  # "p.7" 또는 "chars:0-3000"


@dataclass
class FetchedDocument:
    url: str
    title: str
    parts: list[DocumentPart] = field(default_factory=list)
    content_sha256: str = ""
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
    # 문단 경계를 살려야 인용문이 문장 중간에서 잘리지 않는다.
    blocks = [normalize_text(el.get_text(" ")) for el in soup.find_all(["p", "li", "h1", "h2", "h3"])]
    text = "\n".join(b for b in blocks if len(b) > 30)
    return title, text or normalize_text(soup.get_text(" "))


def _split_by_chars(text: str, size: int = CHARS_PER_PAGE) -> list[DocumentPart]:
    return [
        DocumentPart(text=text[i : i + size], locator=f"chars:{i}-{min(i + size, len(text))}")
        for i in range(0, len(text), size)
    ]


# pdfplumber 기본값(3)은 arXiv 논문에서 단어를 붙여 버린다.
# 실측: 기본값은 "Inthepastfewyears,LargeLanguageModels" 로 추출되어 BM25 토크나이저가
# 통째로 한 토큰으로 잡는다. 1.5 로 낮추면 정상 분리되고 과분할도 관찰되지 않았다.
PDF_X_TOLERANCE = 1.5


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


def fetch_document(
    url: str, *, timeout_seconds: float = 25.0, cache_dir: Path | None = None
) -> FetchedDocument:
    """URL 하나를 가져와 본문을 파싱한다. 실패해도 예외 대신 error 를 담아 돌려준다."""
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (content_checksum(url)[:16] + ".bin")

    try:
        if cache_path is not None and cache_path.exists():
            payload, content_type = cache_path.read_bytes(), ""
        else:
            with httpx.Client(
                timeout=timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": "kvcache-domain-agent/1.0 (research)"},
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
        url=url,
        title=title or url,
        parts=parts,
        content_sha256=content_checksum(full),
        page_count=page_count,
        is_long=len(full) > SHORT_DOCUMENT_CHARS,
    )


def pick_quote(text: str, max_chars: int = 400) -> str:
    """근거로 쓸 인용문을 문장 경계에서 자른다.

    문장 중간에서 자르면 guard 의 원문 대조는 통과하더라도 읽는 사람이 맥락을 잃는다.
    """
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind("다. "))
    return cut[: boundary + 1] if boundary > max_chars * 0.5 else cut


_NUM_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|x|배|GB|TB|MB|KB|ms|us|ns|s초|초|W|kW|MW|TFLOPS|GB/s|TB/s|token|tokens)",
    re.IGNORECASE,
)


def has_numeric_with_unit(text: str) -> bool:
    """단위 없는 숫자만 있는 문장은 도메인 평가 근거로 약하다."""
    return bool(_NUM_UNIT.search(text or ""))

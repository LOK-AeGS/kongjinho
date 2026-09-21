"""근거 저장소: 결정적 ID와 멱등 병합.

리스트에 operator.add로 쌓으면 재시도할 때마다 같은 근거가 중복으로 들어간다.
그래서 evidence_store는 evidence_id를 키로 하는 dict이고, ID는 출처 신원에서 해시로
유도해 같은 자료를 다시 가져와도 같은 ID가 나오게 한다(idempotent).

quote와 content_sha256을 함께 보관하는 이유는 검증 때문이다. deterministic guard가
"주장이 인용한 문장이 실제 원문에 있는가"를 문자열 대조로 확인하는데,
원문 스냅샷 없이는 이 검사를 할 수 없다.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

SourceType = Literal["paper", "patent", "official_web", "standard", "news", "vendor", "other"]
RetrievedVia = Literal["web_search", "fetched_document", "document_index"]

ID_PREFIX = "domain:ev"
_WS = re.compile(r"\s+")
# 추적용 쿼리 파라미터는 같은 문서를 다른 URL로 보이게 만들어 중복 ID를 유발한다.
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref=", "mc_cid", "mc_eid")


def normalize_url(url: str) -> str:
    """같은 문서가 다른 ID를 받지 않도록 URL을 정규화한다."""
    parts = urlsplit(url.strip())
    host = parts.hostname.lower() if parts.hostname else ""
    host = host[4:] if host.startswith("www.") else host
    query = "&".join(
        q for q in parts.query.split("&") if q and not any(t in q.lower() for t in _TRACKING_PARAMS)
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def normalize_text(text: str) -> str:
    """공백·개행 차이로 같은 인용문이 달라 보이지 않게 한다. PDF 추출본에서 특히 잦다."""
    return _WS.sub(" ", (text or "").replace("­", "")).strip()


def make_evidence_id(url: str, locator: str, quote: str) -> str:
    """출처 신원(정규화 URL + 위치 + 인용문)에서 ID를 유도한다.

    재시도·재실행에서 같은 근거가 같은 ID를 받아야 evidence_store 병합이 멱등해진다.
    """
    raw = f"{normalize_url(url)}|{locator}|{normalize_text(quote)}"
    return f"{ID_PREFIX}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def content_checksum(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass
class Evidence:
    """근거 1건. locator와 quote는 검증에 쓰이므로 비어 있으면 안 된다."""

    evidence_id: str
    source_type: SourceType
    title: str
    author_or_organization: str
    url: str
    quote: str  # 원문에서 그대로 옮긴 문장
    locator: str  # "p.7" / "section:Evaluation" / "chars:1200-1800"
    retrieved_via: RetrievedVia
    accessed_date: str
    published_date: str | None = None
    content_sha256: str | None = None  # 인용을 대조할 원문 스냅샷의 체크섬
    search_query: str | None = None  # 어떤 질의로 찾았는지(검색 로그 추적)

    @classmethod
    def build(
        cls,
        *,
        source_type: SourceType,
        title: str,
        author_or_organization: str,
        url: str,
        quote: str,
        locator: str,
        retrieved_via: RetrievedVia,
        published_date: str | None = None,
        content_sha256: str | None = None,
        search_query: str | None = None,
        accessed_date: str | None = None,
    ) -> "Evidence":
        quote = normalize_text(quote)
        return cls(
            evidence_id=make_evidence_id(url, locator, quote),
            source_type=source_type,
            title=normalize_text(title) or url,
            author_or_organization=author_or_organization,
            url=url,
            quote=quote,
            locator=locator,
            retrieved_via=retrieved_via,
            accessed_date=accessed_date or date.today().isoformat(),
            published_date=published_date,
            content_sha256=content_sha256,
            search_query=search_query,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def merge_evidence(left: dict[str, dict], right: dict[str, dict]) -> dict[str, dict]:
    """evidence_store 리듀서. 같은 ID는 덮어쓰지 않고 비어 있던 필드만 채운다.

    덮어쓰기로 두면 재시도에서 published_date 같은 필드가 None으로 되돌아갈 수 있다.
    병합 결과가 입력 순서에 좌우되지 않아야 재현성이 유지된다.
    """
    merged = dict(left or {})
    for key, incoming in (right or {}).items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = incoming
            continue
        filled = dict(existing)
        for field_name, value in incoming.items():
            if filled.get(field_name) in (None, "", []) and value not in (None, "", []):
                filled[field_name] = value
        merged[key] = filled
    return merged


@dataclass
class SearchLogEntry:
    """검색 로그. 어떤 질의가 어떤 출처를 데려왔는지 남겨야 재현·감사가 가능하다."""

    query: str
    provider: str
    requested_at: str
    result_urls: list[str] = field(default_factory=list)
    accepted_urls: list[str] = field(default_factory=list)  # 출처 정책 통과분
    rejected: list[dict] = field(default_factory=list)  # {"url":..., "reason":...}
    from_cache: bool = False
    error: str | None = None

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

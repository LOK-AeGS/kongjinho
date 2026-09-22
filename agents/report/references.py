"""실제 인용 근거만 참고문헌으로 변환한다."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REFERENCE_CATEGORY_ORDER = ("patent", "paper", "other")
REFERENCE_CATEGORY_LABELS = {
    "patent": "특허",
    "paper": "논문",
    "other": "기타",
}


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query = urlencode([
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ])
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def source_identity(evidence: dict) -> tuple[str, str]:
    document_id = evidence.get("document_id")
    if document_id:
        return "document_id", str(document_id)
    url = normalize_url(str(evidence.get("url") or ""))
    if url:
        return "url", url
    fallback = "|".join(
        str(evidence.get(field) or "").strip().casefold()
        for field in ("title", "author_or_organization", "published_date")
    )
    return "metadata", fallback


def reference_category(evidence: dict) -> str:
    """공통 Evidence의 source_type을 특허·논문·기타로 분류한다."""
    source_type = str(evidence.get("source_type") or "other").strip().casefold()
    if source_type in {"patent", "patents"}:
        return "patent"
    if source_type in {"paper", "academic", "academic_paper", "journal", "conference"}:
        return "paper"
    return "other"


def _creator(author: str, published: str, *, year_only: bool) -> str:
    displayed_date = published[:4] if year_only and len(published) >= 4 else published
    if author:
        return author + (f"({displayed_date})" if displayed_date else "")
    return f"({displayed_date})" if displayed_date else ""


def _arxiv_id(url: str) -> str:
    parts = urlsplit(url)
    if not (parts.hostname or "").casefold().endswith("arxiv.org"):
        return ""
    match = re.search(r"/(?:abs|pdf)/([^/?#]+)", parts.path)
    return match.group(1).removesuffix(".pdf") if match else ""


def _site_name(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").casefold().removeprefix("www.")
    if host == "research.google" and "/blog" in parts.path.casefold():
        return "Google Research Blog"
    return host


def _sentence(parts: list[str]) -> str:
    values = [part.strip().rstrip(".") for part in parts if part.strip()]
    return ". ".join(values) + ("." if values else "")


def format_reference(evidence: dict) -> str:
    """자료 유형별 서지 형식을 적용하며 없는 메타데이터는 채우지 않는다."""
    author = str(evidence.get("author_or_organization") or "").strip()
    published = str(evidence.get("published_date") or "").strip()
    title = str(evidence.get("title") or "").strip()
    url = str(evidence.get("url") or "").strip()
    locator = str(evidence.get("locator") or "").strip()
    category = reference_category(evidence)

    if category == "patent":
        parts = [_creator(author, published, year_only=True)]
        if title:
            parts.append(f"*{title}*")
        parts.extend(value for value in (locator, url) if value)
    elif category == "paper":
        parts = [_creator(author, published, year_only=True)]
        if title:
            parts.append(title)
        arxiv_id = _arxiv_id(url)
        if arxiv_id:
            parts.append(f"*arXiv*, {arxiv_id}")
        else:
            parts.extend(value for value in (locator, url) if value)
    else:
        parts = [_creator(author, published, year_only=False)]
        if title:
            parts.append(f"*{title}*")
        site_name = _site_name(url)
        if site_name:
            parts.append(site_name)
        if locator and not url:
            parts.append(locator)
        if url:
            parts.append(url)

    sentence = _sentence(parts)
    return f"{REFERENCE_CATEGORY_LABELS[category]} : {sentence}" if sentence else ""


def collect_references(
    used_evidence_ids: list[str], evidence_store: dict[str, dict]
) -> tuple[list[str], dict[str, dict]]:
    """source 단위로 중복 제거한 뒤 특허 → 논문 → 기타 순서로 묶는다."""
    seen: set[tuple[str, str]] = set()
    grouped_lines: dict[str, list[str]] = {
        category: [] for category in REFERENCE_CATEGORY_ORDER
    }
    records: dict[str, dict] = {}
    for evidence_id in used_evidence_ids:
        evidence = evidence_store.get(evidence_id)
        if not evidence:
            continue
        identity = source_identity(evidence)
        if identity in seen:
            continue
        seen.add(identity)
        line = format_reference(evidence)
        if not line:
            continue
        grouped_lines[reference_category(evidence)].append(line)
        records[evidence_id] = {
            "evidence_id": evidence_id,
            "title": evidence.get("title") or "",
            "author_or_org": evidence.get("author_or_organization") or "",
            "published_at": evidence.get("published_date"),
            "url": evidence.get("url") or None,
            "locator": evidence.get("locator") or "",
        }
    lines = [
        line
        for category in REFERENCE_CATEGORY_ORDER
        for line in grouped_lines[category]
    ]
    return lines, records

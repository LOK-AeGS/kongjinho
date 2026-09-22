"""실제 인용 근거만 참고문헌으로 변환한다."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


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


def format_reference(evidence: dict) -> str:
    """없는 메타데이터를 n.d. 같은 값으로 채우지 않고 가진 값만 조합한다."""
    author = str(evidence.get("author_or_organization") or "").strip()
    published = str(evidence.get("published_date") or "").strip()
    title = str(evidence.get("title") or "").strip()
    url = str(evidence.get("url") or "").strip()
    source_type = str(evidence.get("source_type") or "other")

    parts: list[str] = []
    if author:
        parts.append(author + (f"({published})" if published else ""))
    elif published:
        parts.append(f"({published})")
    if title:
        parts.append(f"*{title}*")
    if source_type == "paper":
        locator = str(evidence.get("locator") or "").strip()
        if locator:
            parts.append(locator)
    if url:
        parts.append(url)
    return ". ".join(part.rstrip(".") for part in parts if part) + "."


def collect_references(
    used_evidence_ids: list[str], evidence_store: dict[str, dict]
) -> tuple[list[str], dict[str, dict]]:
    """본문 사용 순서를 유지하면서 source 단위로 중복 제거한다."""
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
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
        lines.append(line)
        records[evidence_id] = {
            "evidence_id": evidence_id,
            "title": evidence.get("title") or "",
            "author_or_org": evidence.get("author_or_organization") or "",
            "published_at": evidence.get("published_date"),
            "url": evidence.get("url") or None,
            "locator": evidence.get("locator") or "",
        }
    return lines, records

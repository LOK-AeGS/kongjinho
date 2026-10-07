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


def _strip_tracking(url: str) -> str:
    """서지 표기용: utm_* 같은 추적 파라미터만 지우고 나머지 URL은 그대로 둔다."""
    if not url or "?" not in url:
        return url
    parts = urlsplit(url)
    query = urlencode([
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def source_identity(evidence: dict) -> tuple[str, str]:
    # 같은 arXiv 논문은 판본(v2·v4)·형식(abs·pdf·html)·고정 코퍼스 여부와 무관하게 한 출처로 본다.
    arxiv_id = _arxiv_base_id(str(evidence.get("url") or ""))
    if arxiv_id:
        return "arxiv", arxiv_id
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
    match = re.search(r"/(?:abs|pdf|html)/([^/?#]+)", parts.path)
    return match.group(1).removesuffix(".pdf") if match else ""


def _arxiv_base_id(url: str) -> str:
    return re.sub(r"v\d+$", "", _arxiv_id(url))


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
    url = _strip_tracking(str(evidence.get("url") or "").strip())
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
            parts.append(f"*arXiv*, {_arxiv_base_id(url)}")
        elif url:
            parts.append(url)
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


def _reference_entries(
    used_evidence_ids: list[str], evidence_store: dict[str, dict]
) -> tuple[list[tuple[tuple[str, str], str]], dict[tuple[str, str], list[str]], dict[str, dict]]:
    """source 단위로 중복 제거한 뒤 특허 → 논문 → 기타 순서로 정렬한 (identity, 서지) 목록."""
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[tuple[tuple[str, str], str]]] = {
        category: [] for category in REFERENCE_CATEGORY_ORDER
    }
    members: dict[tuple[str, str], list[str]] = {}
    records: dict[str, dict] = {}
    for evidence_id in used_evidence_ids:
        evidence = evidence_store.get(evidence_id)
        if not evidence:
            continue
        identity = source_identity(evidence)
        members.setdefault(identity, []).append(evidence_id)
        if identity in seen:
            continue
        seen.add(identity)
        line = format_reference(evidence)
        if not line:
            continue
        grouped[reference_category(evidence)].append((identity, line))
        records[evidence_id] = {
            "evidence_id": evidence_id,
            "title": evidence.get("title") or "",
            "author_or_org": evidence.get("author_or_organization") or "",
            "published_at": evidence.get("published_date"),
            "url": evidence.get("url") or None,
            "locator": evidence.get("locator") or "",
        }
    entries = [entry for category in REFERENCE_CATEGORY_ORDER for entry in grouped[category]]
    return entries, members, records


def collect_references(
    used_evidence_ids: list[str], evidence_store: dict[str, dict]
) -> tuple[list[str], dict[str, dict]]:
    """source 단위로 중복 제거한 뒤 특허 → 논문 → 기타 순서로 묶는다."""
    entries, _, records = _reference_entries(used_evidence_ids, evidence_store)
    return [line for _, line in entries], records


def reference_numbers(
    used_evidence_ids: list[str], evidence_store: dict[str, dict]
) -> dict[str, int]:
    """evidence_id → REFERENCE 번호(1부터). 같은 출처의 근거는 같은 번호를 받는다.

    번호는 코드가 REFERENCE 순서에서 직접 계산하므로, 본문 번호와 실제 출처가 어긋날 수 없다.
    """
    entries, members, _ = _reference_entries(used_evidence_ids, evidence_store)
    numbers: dict[str, int] = {}
    for number, (identity, _) in enumerate(entries, 1):
        for evidence_id in members.get(identity, []):
            numbers[evidence_id] = number
    return numbers


_CITATION_MARK = re.compile(r"〔근거:\s*([^〕]+)〕")


def number_citations(markdown: str, numbers: dict[str, int]) -> str:
    """제출용 표기: `〔근거: id, id〕`를 `[1, 3]`으로, REFERENCE 항목 앞에 `[n]`을 붙인다.

    검증·추적은 ID 표기 원본으로 하고, 이 변환은 PDF·report.md 출력에만 쓴다.
    번호가 없는 ID는 그대로 남겨 누락을 숨기지 않는다.
    """

    def replace(match: re.Match) -> str:
        ids = [value.strip() for value in match.group(1).split(",") if value.strip()]
        known = sorted({numbers[value] for value in ids if value in numbers})
        unknown = [value for value in ids if value not in numbers]
        parts = [str(number) for number in known] + unknown
        return f"[{', '.join(parts)}]" if parts else ""

    head, marker, tail = markdown.partition("# REFERENCE")
    head = _CITATION_MARK.sub(replace, head)
    if not marker:
        return head
    lines = tail.splitlines()
    counter = 0
    for index, line in enumerate(lines):
        if line.startswith("- "):
            counter += 1
            lines[index] = f"- [{counter}] {line[2:]}"
    return head + marker + "\n".join(lines)

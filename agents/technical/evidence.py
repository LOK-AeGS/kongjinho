"""기술조사 근거 ID, 인용 검증, 부모 Evidence 변환."""

from __future__ import annotations

import hashlib
from datetime import date

from .corpus import normalize_text


def make_evidence_id(source_identity: str, locator: str, quote: str) -> str:
    raw = f"{source_identity}|{locator}|{normalize_text(quote)}"
    return "technical:ev:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_claim_id(technology: str, text: str, evidence_ids: list[str]) -> str:
    raw = f"{technology}|{normalize_text(text)}|{'|'.join(sorted(evidence_ids))}"
    return "technical:claim:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _candidate_containing(quote: str, candidates: dict[str, dict]) -> dict | None:
    """인용문을 글자 그대로 포함하는 다른 후보를 찾는다. 없으면 None."""
    for candidate in candidates.values():
        if quote in normalize_text(candidate.get("text", "")):
            return candidate
    return None


def validate_reference(reference: dict, candidates: dict[str, dict]) -> tuple[dict | None, str | None]:
    quote = normalize_text(reference.get("quote", ""))
    candidate = candidates.get(reference.get("candidate_id", ""))
    if not quote:
        chunk_id = candidate["chunk_id"] if candidate else reference.get("candidate_id")
        return None, f"빈 인용문: {chunk_id}"
    if candidate is not None and quote in normalize_text(candidate.get("text", "")):
        return candidate, None
    # 인용문이 원문 그대로 다른 후보에 있으면 출처를 그쪽으로 바로잡는다. 모델이 붙인
    # candidate_id 만 틀린 경우가 실측 실패의 다수였고(28건 중 12건이 인용한 chunk 와
    # 일치율 30% 미만), 이 경로도 완전 일치를 요구하므로 지어낸 인용문은 여전히
    # 통과하지 못한다.
    relocated = _candidate_containing(quote, candidates)
    if relocated is not None:
        return relocated, None
    if candidate is None:
        return None, f"존재하지 않는 candidate_id: {reference.get('candidate_id')}"
    return None, f"원문에 없는 인용문: {candidate['chunk_id']}"


def to_parent_evidence(candidate: dict, quote: str, claim_id: str, *, accessed_at: str | None = None) -> dict:
    clean_quote = normalize_text(quote)
    source_identity = candidate.get("url") or candidate.get("doc_id")
    if not source_identity:
        raise ValueError("근거 후보에 URL 또는 doc_id가 필요합니다.")
    evidence_id = make_evidence_id(source_identity, candidate["locator"], clean_quote)
    primary_or_secondary = candidate.get("primary_or_secondary")
    if primary_or_secondary not in {"primary", "secondary"}:
        primary_or_secondary = "primary"
    direct_or_proxy = candidate.get("direct_or_proxy")
    if direct_or_proxy not in {"direct", "proxy"}:
        direct_or_proxy = "direct" if candidate.get("role") == "primary" else "proxy"
    return {
        "id": evidence_id,
        "claim_id": claim_id,
        "doc_id": candidate["doc_id"],
        "title": candidate["title"],
        "author_or_org": candidate["author_or_org"],
        "source_type": candidate.get("source_type", "paper"),
        "primary_or_secondary": primary_or_secondary,
        "direct_or_proxy": direct_or_proxy,
        "url": candidate.get("url"),
        "published_at": candidate.get("published_at"),
        "accessed_at": accessed_at or date.today().isoformat(),
        "page_or_locator": candidate["locator"],
        "quote": clean_quote,
        "stance": "neutral",
        "evidence_level": "unknown",
        "metric_tag": None,
        "perspective": "technical",
        "content_hash": candidate["content_hash"],
    }


def merge_evidence(left: dict[str, dict], right: dict[str, dict]) -> dict[str, dict]:
    merged = dict(left or {})
    for evidence_id, incoming in (right or {}).items():
        if evidence_id not in merged:
            merged[evidence_id] = incoming
            continue
        current = dict(merged[evidence_id])
        for key, value in incoming.items():
            if current.get(key) in (None, "", [], {}):
                current[key] = value
        level_rank = {"unknown": 0, "forecast": 1, "announcement": 2, "pilot": 3, "production": 4}
        if level_rank.get(incoming.get("evidence_level", "unknown"), 0) > level_rank.get(current.get("evidence_level", "unknown"), 0):
            current["evidence_level"] = incoming["evidence_level"]
        merged[evidence_id] = current
    return merged

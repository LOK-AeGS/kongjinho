"""시장 에이전트 내부 State와 부모 State(graph/state.py의 AppState) 변환.

내부에서는 stance/scope/observed_at을 Claim의 실제 필드로 다룬다(subgraph.py는 바꾸지 않는다).
부모 State는 AppState 하나로 확정됐다(ISSUE.md 1-1, 2026-09-22 "Replace parent State with team
AppState" 커밋). to_perspective_findings()가 이 내부 표현을 AppState.market_findings
(PerspectiveFindings)와 AppState.evidence_store(Evidence)로 옮긴다.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from agents.market.rag.evidence import content_checksum, make_evidence_id

TechnologyID = Literal["sw", "hw"]
Aspect = Literal["size", "adoption", "ecosystem", "counter"]
EvidenceLevel = Literal["forecast", "announcement", "pilot", "production", "unknown"]
Stance = Literal["support", "counter", "neutral"]
Scope = Literal["direct", "class"]


class SearchResult(TypedDict, total=False):
    title: str
    url: str
    content: str
    organization: str
    published_date: str | None
    source_type: str
    document_id: str  # RAG 청크일 때만
    page: int
    section: str | None


class InternalEvidence(TypedDict):
    """정규화 전 원시 근거. url+page/locator+quote가 신원을 결정한다."""

    id: str
    url: str
    locator: str  # 논문: "p.21", 웹: "url" 자체
    title: str
    organization: str
    source_type: str
    published_at: str | None
    accessed_at: str
    quote: str
    doc_id: str | None
    scope: Scope


class Claim(TypedDict):
    claim_id: str
    technology_id: TechnologyID
    aspect: str  # size/adoption/ecosystem/counter, 또는 추론 주제("채택 동인" 등)
    basis: Literal["direct_evidence", "inference"]
    evidence_level: EvidenceLevel | None
    statement: str
    evidence_ids: list[str]
    stance: Stance
    scope: Scope
    observed_at: str | None
    uncertainty: str
    is_stub: bool


class Completion(TypedDict):
    status: Literal["complete", "partial", "failed"]
    search_rounds_used: int
    revision_rounds_used: int
    gaps: list[str]
    errors: list[str]


class MarketLocal(TypedDict, total=False):
    technologies: dict[str, str]
    tech_desc: dict[str, str]
    domain: str
    as_of_date: str
    technical_summary: str
    max_search_rounds: int
    query_bank: dict[str, list[str]]
    pending: list[dict]
    raw: list[dict]
    seen: list[str]
    quote_keys: list[str]
    evidence: list[InternalEvidence]
    claims: list[Claim]
    verdicts: list[dict]
    search_rounds_used: int
    errors: list[str]
    next_action: str
    completion: Completion
    prompt_version: str
    pages_used: int  # Pool B 본문 수집 누적 페이지(agents/market/rag/fetch.py 의 PAGE_BUDGET 기준)
    search_log: list[dict]  # {"message": str}. 출처 등급 필터·예산 초과 등 검색 단계 로그


def new_evidence(*, url: str, locator: str, title: str, organization: str, source_type: str,
                  published_at: str | None, accessed_at: str, quote: str, doc_id: str | None,
                  scope: Scope) -> InternalEvidence:
    return {
        "id": make_evidence_id(url, locator, quote), "url": url, "locator": locator, "title": title,
        "organization": organization, "source_type": source_type, "published_at": published_at,
        "accessed_at": accessed_at, "quote": quote, "doc_id": doc_id, "scope": scope,
    }


# ---------------- graph/state.py (AppState) 로 변환 ----------------
#
# 몇몇 매핑은 설계서에 표가 없어 이 함수에서 직접 정한 것이다:
# - assessment(market_signal) 값 도출: evidence_level → adopted/announced/projected/none
# - primary_or_secondary: §1.4 출처 우선순위(1차/2차 출처) 기준으로 source_type을 분류
# - scope 'mixed': 같은 기준에 direct·class 근거가 섞여 있을 때만 쓴다
#   (quality/rubric.py의 verdict.scope 필드는 direct/class/'-' 3값만 만들어서, 여기서는
#   raw claim.scope 집합을 따로 봐서 'mixed'를 판정한다)
# - 근거가 없는 칸의 scope: Scope 타입에 '해당 없음' 값이 없어 'direct'를 기본값으로 두고,
#   basis=unknown이 근거 부재 사실을 대신 나타낸다
# - '반대·한계 근거'는 VerdictRecord로 만들지 않는다: quality/rubric.py가 이 기준에는
#   충분/부분/부족 강도를 정의하지 않고(counter_missing으로 존재 여부만 판정) 있어서,
#   있는 로직만 그대로 gaps로 옮긴다.

_LEVEL_RANK = {"production": 4, "pilot": 3, "announcement": 2, "forecast": 1, "unknown": 0}
_LEVEL_TO_SIGNAL: dict[str, str] = {
    "production": "adopted", "announcement": "announced", "pilot": "projected",
    "forecast": "projected", "unknown": "none",
}
_BASIS_OF_VERDICT = {"충분": "direct", "부분": "inferred", "부족": "unknown"}
_PRIMARY_SOURCE_TYPES = {"paper", "patent", "official_web"}  # §1.4: 논문·표준·공식 문서 = 1차 출처


def _v08_highest_level(levels: list[str]) -> str:
    return max(levels, key=lambda lv: _LEVEL_RANK.get(lv, 0)) if levels else "unknown"


def _v08_evidence(evidence: list[InternalEvidence], claims: list[Claim]):
    """InternalEvidence 목록을 graph.state.Evidence로 옮긴다. id는 재해시하지 않고 그대로 쓴다."""
    citing_by_ev: dict[str, Claim] = {}
    for c in claims:
        for eid in c["evidence_ids"]:
            citing_by_ev.setdefault(eid, c)  # 같은 근거를 여러 주장이 인용해도 첫 인용만 남긴다.

    out = []
    for ev in evidence:
        citing = citing_by_ev.get(ev["id"])
        out.append({
            "id": ev["id"],
            "claim_id": citing["claim_id"] if citing else "",
            "doc_id": ev["doc_id"],
            "title": ev["title"],
            "author_or_org": ev["organization"],
            "source_type": ev["source_type"],
            "primary_or_secondary": "primary" if ev["source_type"] in _PRIMARY_SOURCE_TYPES else "secondary",
            "direct_or_proxy": "direct" if ev["scope"] == "direct" else "proxy",
            "url": ev["url"],
            "published_at": ev["published_at"],
            "accessed_at": ev["accessed_at"],
            "page_or_locator": ev["locator"],
            "quote": ev["quote"],
            "stance": citing["stance"] if citing else "not_found",
            "evidence_level": (citing["evidence_level"] if citing else None) or "unknown",
            "metric_tag": None,  # TODO: 수치 상충 자동 감지는 이번 범위 밖.
            "perspective": "market",
            "content_hash": content_checksum(ev["quote"]),
        })
    return out


def _v08_records(claims: list[Claim], verdicts: list[dict]):
    from agents.market.quality import rubric

    key_of = {v: k for k, v in rubric.CRITERIA.items()}
    out = []
    for v in verdicts:
        aspect_key = key_of[v["criterion"]]
        matching = [c for c in claims if c["basis"] == "direct_evidence" and c["technology_id"] == v["technology_id"] and c["aspect"] == aspect_key]
        levels = [c["evidence_level"] for c in matching if c["evidence_level"]]
        scopes = {c["scope"] for c in matching}
        if len(scopes) > 1:
            scope = "mixed"
        elif scopes:
            scope = "class" if "class" in scopes else "direct"
        else:
            scope = "direct"  # 판정할 근거가 없음(basis=unknown이 이 사실을 나타낸다).
        level = _v08_highest_level(levels)
        findings_text = "; ".join(c["statement"].strip() for c in matching)[:240]
        if not findings_text:
            findings_text = f"근거 부족: {v['criterion']}"
        out.append({
            "technology": v["technology_id"],
            "perspective": "market",
            "criterion": v["criterion"],
            "basis": _BASIS_OF_VERDICT[v["verdict"]],
            "evidence_level": level,
            "scope": scope,
            "stance_counts": dict(v["stance_counts"]),
            "evidence_ids": list(v["evidence_ids"]),
            "assessment": _LEVEL_TO_SIGNAL[level],
            "assessment_vocab": "market_signal",
            "value": None,  # 시장 관점엔 TRL처럼 관점 고유 수치 범위가 없다.
            "findings": findings_text,
            "limitations": list(v["limitations"]),
        })
    return out


def _v08_gaps(claims: list[Claim], verdicts: list[dict]):
    """graph.state.Gap: technology, perspective, criterion, reason, missing_evidence."""
    from agents.market.quality import rubric

    gaps = [
        {
            "technology": v["technology_id"], "perspective": "market", "criterion": v["criterion"],
            "reason": "직접 근거 부족(basis=unknown)",
            "missing_evidence": [],  # 없다는 사실만 판정하지, 기대했던 근거 ID 목록은 추적하지 않는다.
        }
        for v in verdicts if v["verdict"] == "부족"
    ]
    gaps += [
        {
            "technology": tid, "perspective": "market", "criterion": "반대·한계 근거",
            "reason": "반대·한계 근거 없음", "missing_evidence": [],
        }
        for tid in rubric.counter_missing(claims)
    ]
    return gaps


def _v08_claims(claims: list[Claim], id_map: dict[str, str]):
    """graph.state.Claim: claim_id, technology, perspective, text, evidence_ids, conditions, limitations.

    내부 Claim의 uncertainty(자유 서술 caveat)를 claim별 limitations 한 줄로 옮긴다.
    """
    return [
        {
            "claim_id": c["claim_id"],
            "technology": c["technology_id"],
            "perspective": "market",
            "text": c["statement"].strip()[:240],
            "evidence_ids": [id_map[e] for e in c["evidence_ids"] if e in id_map],
            "conditions": [],  # 내부 Claim은 stance/scope/observed_at을 실제 필드로 다루므로 태그가 없다.
            "limitations": [c["uncertainty"]] if c["uncertainty"].strip() else [],
        }
        for c in claims
    ]


def to_perspective_findings(s: MarketLocal, completion: Completion) -> dict:
    """market_agent 실행 결과를 부모 State(graph/state.py AppState)의 market_findings·evidence_store로 옮긴다.

    finalize()가 이미 계산해 둔 s['verdicts']를 그대로 쓴다(quality/rubric.py 재계산 없음).
    id_map은 InternalEvidence.id를 그대로 쓰므로 항등 사상이지만, 재해시가 필요해지면 이 함수만
    고치면 되도록 남겨 둔다.
    """
    from agents.market.quality import rubric

    claims, evidence, names = s["claims"], s["evidence"], s["technologies"]
    verdicts = s["verdicts"]
    id_map = {e["id"]: e["id"] for e in evidence}

    limitations: list[str] = []
    note = rubric.imbalance_note(claims, names)  # 기술 간 비교라 개별 VerdictRecord엔 못 담는다.
    if note:
        limitations.append(note)
    limitations += [f"오류: {e}" for e in completion["errors"]]  # PerspectiveFindings엔 errors 필드가 없다.

    findings = {
        "perspective": "market",
        "status": completion["status"],
        "records": _v08_records(claims, verdicts),
        "claims": _v08_claims(claims, id_map),
        "gaps": _v08_gaps(claims, verdicts),
        "limitations": limitations,
        "input_evidence_ids": [],  # TODO: 시장 에이전트가 technical_findings를 아직 안 읽는다.
    }
    # evidence_store는 market_findings 밖, 부모 State의 공유 키다(AppState.evidence_store).
    return {"market_findings": findings, "evidence_store": {e["id"]: e for e in _v08_evidence(evidence, claims)}}

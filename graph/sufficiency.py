"""근거 충분성 판단과 재작업 예산. Supervisor 가 State 만 보고 계산하는 순수 함수 모음.

충분성 기준 (market·stakeholder·domain): findings 가 있고 status 가 failed 가 아니며,
그 관점의 근거가 MIN_EVIDENCE 건 이상이고 서로 다른 출처가 MIN_SOURCES 곳 이상.
기술 조사(technical)는 고정 코퍼스(Pool A) 기반이라 근거 수 기준을 적용하지 않고 실패 여부만 본다.
not_found(반증을 찾지 못함)는 부족이 아니라 정상 결과로 취급한다.
"""

from __future__ import annotations

from urllib.parse import urlparse

PERSPECTIVES = ("technical", "market", "stakeholder", "domain")
MIN_EVIDENCE = 3
MIN_SOURCES = 2
PER_PERSPECTIVE_REWORK = 2
GLOBAL_REWORK = 4
MAX_QUALITY_LOOPS = 2
MAX_FOCUS = 6


def source_key(evidence: dict) -> str:
    url = evidence.get("url")
    if url:
        return urlparse(url).netloc or url
    return str(evidence.get("doc_id") or evidence.get("title") or evidence.get("id"))


def perspective_evidence(state: dict, perspective: str) -> list[dict]:
    store = state.get("evidence_store") or {}
    return [ev for ev in store.values() if ev.get("perspective") == perspective]


def attempts(state: dict, name: str) -> int:
    return int(((state.get("node_status") or {}).get(name) or {}).get("attempts", 0))


def is_stale(state: dict, name: str) -> bool:
    return ((state.get("node_status") or {}).get(name) or {}).get("status") == "stale"


def rework_used(state: dict) -> int:
    return sum(max(attempts(state, p) - 1, 0) for p in PERSPECTIVES)


def budget_left(state: dict, perspective: str) -> bool:
    return max(attempts(state, perspective) - 1, 0) < PER_PERSPECTIVE_REWORK and rework_used(state) < GLOBAL_REWORK


def assess(state: dict, perspective: str) -> dict:
    """한 관점의 충분성. present / failed / sufficient / reasons 를 돌려준다."""
    findings = state.get(f"{perspective}_findings")
    if findings is None:
        return {"present": False, "failed": False, "sufficient": False, "reasons": ["결과 없음"], "n_evidence": 0, "n_sources": 0}
    evidence = perspective_evidence(state, perspective)
    sources = {source_key(ev) for ev in evidence}
    reasons = []
    failed = findings.get("status") == "failed"
    if failed:
        reasons.append("status=failed")
    if perspective != "technical":
        if len(evidence) < MIN_EVIDENCE:
            reasons.append(f"근거 {len(evidence)}건 < {MIN_EVIDENCE}건")
        if len(sources) < MIN_SOURCES:
            reasons.append(f"출처 {len(sources)}곳 < {MIN_SOURCES}곳")
    return {"present": True, "failed": failed, "sufficient": not reasons, "reasons": reasons,
            "n_evidence": len(evidence), "n_sources": len(sources)}


def rework_candidates(state: dict) -> dict[str, dict]:
    """다시 돌려야 하는 관점 → {reason, requested_by}. 최초 실행(결과 없음)도 포함한다."""
    synthesis = state.get("synthesis")
    synthesis_fresh = synthesis is not None and not is_stale(state, "synthesis")
    asked_by_synthesis = {r.get("perspective") for r in (synthesis or {}).get("retry_requests") or []} if synthesis_fresh else set()
    pending = state.get("rework_requests") or {}
    result = {}
    for p in PERSPECTIVES:
        a = assess(state, p)
        if not a["present"]:
            result[p] = {"reason": "결과 없음(최초 실행)", "requested_by": None}
        elif pending.get(p):
            result[p] = {"reason": "재작업 요청 대기", "requested_by": pending[p]["requested_by"]}
        elif not a["sufficient"] and budget_left(state, p):
            result[p] = {"reason": ", ".join(a["reasons"]), "requested_by": "supervisor-sufficiency"}
        elif p in asked_by_synthesis and budget_left(state, p):
            result[p] = {"reason": "평가 종합의 retry_requests", "requested_by": "synthesis"}
    return result


def make_rework_request(state: dict, perspective: str, requested_by: str, extra_gaps: list[dict] | None = None) -> dict:
    findings = state.get(f"{perspective}_findings") or {}
    gaps = [*(findings.get("gaps") or []), *(extra_gaps or [])][:MAX_FOCUS]
    focus = [str(g.get("criterion")) for g in gaps if g.get("criterion") and g.get("criterion") != "(전체)"]
    return {"perspective": perspective, "gaps": gaps, "focus_queries": list(dict.fromkeys(focus))[:MAX_FOCUS],
            "extra_rounds": 1, "requested_by": requested_by}


def exhausted_perspectives(state: dict) -> list[str]:
    """예산을 다 써도 충분하지 못한 관점 (보고서에 한계로 명시해야 한다)."""
    return [p for p in PERSPECTIVES if not assess(state, p)["sufficient"] and not budget_left(state, p)]

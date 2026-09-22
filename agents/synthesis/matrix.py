"""① 결과 완결성 검사 + 비교 매트릭스. LLM 없이 코드로만 계산한다.

설계서 §8.6 "결과 완결성 검사 → 비교 매트릭스 생성 → 근거 강도·불확실성 반영"을 한 단계로 합쳤다.
근거 강도(basis)·증거 수준·근거 수·stance 는 매트릭스 셀 자체에 들어가므로 별도 단계가 필요 없다.
완결성 검사는 결과를 거르지 않고 gaps / limitations / retry_requests 에 기록만 한다 (partial 도 정상 산출물).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

from agents.synthesis.rules import IMBALANCE_RATIO, PERSPECTIVES, TECHNOLOGIES

REQUIRED_RECORD_FIELDS = (
    "technology", "perspective", "criterion", "basis", "evidence_level",
    "scope", "stance_counts", "evidence_ids", "assessment", "assessment_vocab",
)
# §6.4 시간 규칙: 이 관점들의 근거는 발행일/접근일이 있어야 한다.
TIME_SENSITIVE = ("technical", "market", "stakeholder")


def record_ref(record: dict) -> str:
    """CrossFinding.record_refs 형식: "market/hw/상용화·채택 현황"."""
    return f"{record['perspective']}/{record['technology']}/{record['criterion']}"


def content_key(evidence_id: str, store: dict) -> str:
    """중복 판정 키. content_hash 가 없으면 evidence_id 로 대신한다."""
    ev = store.get(evidence_id) or {}
    return ev.get("content_hash") or evidence_id


def _parse_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        try:
            return date.fromisoformat(text[:7] + "-01")
        except ValueError:
            return None


def input_hash(findings: dict, store: dict) -> str:
    """같은 입력이면 같은 해시. 재실행·캐시 판단용 (SynthesisResult.input_hash)."""
    payload = {
        "findings": {p: findings.get(p) for p in PERSPECTIVES},
        "evidence_ids": sorted(store),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_matrix(state: dict) -> dict:
    findings: dict = state.get("findings") or {}
    store: dict = state.get("evidence_store") or {}
    as_of = _parse_date(state.get("as_of"))

    gaps: list[dict] = []
    limitations: list[str] = []
    retry_requests: list[dict] = []
    matrix: list[dict] = []
    present = 0

    for perspective in PERSPECTIVES:
        result = findings.get(perspective)
        if result is None:
            gaps.append({"technology": "both", "perspective": perspective, "criterion": "(전체)",
                         "reason": "관점 결과 없음 (미실행)", "missing_evidence": []})
            retry_requests.append({"perspective": perspective, "reason": "결과 없음"})
            continue
        status = result.get("status")
        if status == "failed":
            retry_requests.append({"perspective": perspective, "reason": "status=failed"})
            limitations.append(f"{perspective}: 실행 실패 결과(status=failed)를 한계로 반영")
        elif status == "partial":
            limitations.append(f"{perspective}: 부분 결과(status=partial). 판단 보류 칸이 있음")
        gaps.extend(result.get("gaps") or [])
        limitations.extend(f"{perspective}: {text}" for text in result.get("limitations") or [])

        records = result.get("records") or []
        if records:
            present += 1
        for record in records:
            missing = [f for f in REQUIRED_RECORD_FIELDS if f not in record]
            if missing:
                limitations.append(f"{perspective}: 필수 필드 누락 record 제외 ({', '.join(missing)})")
                continue
            ref = record_ref(record)
            known = [eid for eid in record["evidence_ids"] if eid in store]
            unknown = [eid for eid in record["evidence_ids"] if eid not in store]
            if unknown:
                limitations.append(f"{ref}: evidence_store 에 없는 근거 ID {len(unknown)}건 제외")
            if record["basis"] == "direct" and not known:
                limitations.append(f"{ref}: basis=direct 인데 유효한 근거가 없음")
            for eid in known:
                ev = store[eid]
                published = _parse_date(ev.get("published_at"))
                if as_of and published and published > as_of:
                    limitations.append(f"{ref}: 기준일 이후 근거 포함 ({eid})")
                if perspective in TIME_SENSITIVE and not ev.get("accessed_at"):
                    limitations.append(f"{ref}: 접근일(accessed_at) 없는 근거 ({eid})")
            matrix.append({
                "technology": record["technology"],
                "perspective": record["perspective"],
                "criterion": record["criterion"],
                "basis": record["basis"],
                "assessment": record["assessment"],
                "assessment_vocab": record["assessment_vocab"],
                "value": record.get("value"),
                "n_evidence": len({content_key(eid, store) for eid in known}),
                "evidence_level": record["evidence_level"],
                "stance_counts": dict(record["stance_counts"]),
                "scope": record["scope"],
            })

    # 제시 순서 편향 방지: 입력 순서와 무관하게 같은 순서로 정렬한다.
    matrix.sort(key=lambda c: (TECHNOLOGIES.index(c["technology"]) if c["technology"] in TECHNOLOGIES else 9,
                               PERSPECTIVES.index(c["perspective"]), c["criterion"]))

    imbalance = _imbalance(findings, store)
    if imbalance.get("flagged"):
        limitations.append(imbalance["note"])

    if present == 0:
        status = "failed"
    elif gaps or retry_requests or any((findings.get(p) or {}).get("status") != "complete" for p in PERSPECTIVES):
        status = "partial"
    else:
        status = "complete"

    return {
        "status": status,
        "input_hash": input_hash(findings, store),
        "matrix": matrix,
        "gaps": gaps,
        "imbalance": imbalance,
        "retry_requests": retry_requests,
        "limitations": list(dict.fromkeys(limitations)),
    }


def _imbalance(findings: dict, store: dict) -> dict:
    """기술별 근거 수(content_hash 중복 제거)와 basis 분포. not_applicable 은 제외한다."""
    evidence = {t: set() for t in TECHNOLOGIES}
    basis = {t: {} for t in TECHNOLOGIES}
    for result in findings.values():
        for record in (result or {}).get("records") or []:
            tech = record.get("technology")
            if tech not in evidence or record.get("basis") == "not_applicable":
                continue
            evidence[tech] |= {content_key(e, store) for e in record.get("evidence_ids", []) if e in store}
            basis[tech][record["basis"]] = basis[tech].get(record["basis"], 0) + 1
    counts = {t: len(v) for t, v in evidence.items()}
    low, high = min(counts.values()), max(counts.values())
    flagged = high > 0 and (low == 0 or high / low >= IMBALANCE_RATIO)
    note = f"기술별 근거 수 불균형: sw {counts['sw']}건 / hw {counts['hw']}건 (중복 제거, not_applicable 제외)"
    return {"evidence_counts": counts, "basis_counts": basis, "flagged": flagged, "note": note}

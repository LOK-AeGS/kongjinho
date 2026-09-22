"""④ 중립성 검토 (C1~C7, 설계서 §8.6.2). 검사는 코드, 재생성만 LLM.

설계서 그림의 "중립성 검토 → 통과? → 결론 수정 → 다시 검토" 루프를 이 단계 안에서 끝낸다.
규칙이 "위반한 문장만 1회 재생성 → 그래도 위반이면 제거하고 dropped_sentences 에 기록"이라
문장 단위·최대 1회이므로 그래프 루프가 필요 없다.
"""

from __future__ import annotations

import re

from agents.synthesis.rules import (
    OUT_OF_DOMAIN_WORDS,
    PERSPECTIVES,
    RANKING_WORDS,
    RULE_THRESHOLD_NUMBERS,
    condition_violations,
    is_trl_record,
    number_present,
    numbers_in,
    trl_bounds,
)

MAX_CLAIM_CHARS = 240
_TRL_MENTION = re.compile(r"TRL\s*[:=]?\s*(\d(?:\s*[–\-~]\s*\d)?)")
_RULE_CODE = re.compile(r"\b(?:C[1-7]|SX[1-7])\b")


def review_context(state: dict) -> dict:
    """검사에 필요한 색인: 관점이 실제 인용한 근거 집합, 근거별 record, 기술별 TRL 값."""
    store = state.get("evidence_store") or {}
    cited, records_by_evidence, trl, claim_texts = set(), {}, {}, {}
    for perspective in PERSPECTIVES:
        result = (state.get("findings") or {}).get(perspective) or {}
        for r in result.get("records") or []:
            for eid in r.get("evidence_ids", []):
                cited.add(eid)
                records_by_evidence.setdefault(eid, []).append(r)
            if is_trl_record(r) and r.get("value"):
                trl.setdefault(r["technology"], set()).add(trl_bounds(r["value"]))
        for c in result.get("claims") or []:
            cited |= set(c.get("evidence_ids", []))
            claim_texts[f"{perspective}/claim/{c.get('claim_id')}"] = " ".join([c.get("text", "")] + list(c.get("conditions") or []))
    return {"store": store, "cited": cited, "records_by_evidence": records_by_evidence, "trl": trl, "claim_texts": claim_texts}


def check_claim(claim: dict, ctx: dict, extra_haystack: str = "") -> list[str]:
    text = claim.get("text", "")
    tech = claim.get("technology")
    ids = claim.get("evidence_ids") or []
    store, v = ctx["store"], []

    if len(text) > MAX_CLAIM_CHARS:
        v.append(f"길이 초과: {len(text)}자 > {MAX_CLAIM_CHARS}자")
    # C1 근거 ID 존재
    if not ids:
        v.append("C1 근거 ID 없음")
    missing = [e for e in ids if e not in store]
    if missing:
        v.append(f"C1 evidence_store 에 없는 근거: {', '.join(missing)}")
    # C2 네 관점이 실제 인용한 근거 안에 있는지
    outside = [e for e in ids if e in store and e not in ctx["cited"]]
    if outside:
        v.append(f"C2 관점들이 인용하지 않은 근거: {', '.join(outside)}")
    # C3 숫자가 인용 quote 또는 record findings/value 에 있는지
    haystack = " ".join(
        [store[e].get("quote", "") for e in ids if e in store]
        + [f"{r.get('findings', '')} {r.get('value') or ''}" for e in ids for r in ctx["records_by_evidence"].get(e, [])]
        + [extra_haystack]
    )
    for number in numbers_in(text):
        if not number_present(number, haystack):
            v.append(f"C3 근거에 없는 수치: {number}")
    # C4 조건 필수 수치
    v.extend(condition_violations(text, tech))
    # 검사 규칙 코드가 본문에 새어 들어감 (재생성 때 위반 사유를 따라 쓴 경우)
    if _RULE_CODE.search(text):
        v.append("검사 규칙 코드(C1~C7·SX1~SX7)가 문장에 포함됨")
    # C5 우열 어휘
    ranking = [w for w in RANKING_WORDS if w in text]
    if ranking:
        v.append(f"C5 우열 어휘: {', '.join(ranking)}")
    # C6 도메인 밖 키워드, 기술군·전망 표기
    lowered = text.lower()
    outside_domain = [w for w in OUT_OF_DOMAIN_WORDS if w.lower() in lowered]
    if outside_domain:
        v.append(f"C6 도메인 밖 키워드: {', '.join(outside_domain)}")
    if any(store.get(e, {}).get("direct_or_proxy") == "proxy" for e in ids) and "기술군" not in text:
        v.append("C6 기술군(class) 근거 인용 시 '기술군' 표기 필요")
    if any(store.get(e, {}).get("evidence_level") == "forecast" for e in ids) and "전망" not in text:
        v.append("C6 전망(forecast) 근거 인용 시 '전망' 표기 필요")
    # C7 TRL 범위가 매트릭스와 일치
    for mention in _TRL_MENTION.findall(text):
        bounds = trl_bounds(mention)
        allowed = ctx["trl"].get(tech, set()) if tech in ("sw", "hw") else set().union(*ctx["trl"].values())
        if bounds not in allowed:
            v.append(f"C7 매트릭스와 다른 TRL 범위: {mention}")
    if "직접 근거" in text or "direct 근거" in lowered:
        weak = {r.get("basis") for e in ids for r in ctx["records_by_evidence"].get(e, [])} - {"direct"}
        if weak:
            v.append("C7 직접 근거가 아닌 판정을 직접 근거처럼 서술")
    return v


def review_claims(claims: list[dict], explanations: dict, ctx: dict, writer, payload: dict) -> dict:
    kept, dropped, revised = [], [], 0
    for claim in claims:
        violations = check_claim(claim, ctx)
        if violations:
            second = writer.revise(claim, violations, payload) if writer else None
            second_violations = check_claim(second, ctx) if second else violations
            if second and not second_violations:
                claim, revised = second, revised + 1
            else:
                dropped.append({"text": claim.get("text"), "technology": claim.get("technology"),
                                "violations": violations, "stage": "summary_claim",
                                "revised_attempted": second is not None,
                                "revised_text": second.get("text") if second else None,
                                "revised_violations": second_violations if second else None})
                continue
        kept.append(claim)

    checked_explanations = {}
    findings = {f["id"]: f for f in payload.get("cross_findings") or []}
    for finding_id, (text, resolved) in explanations.items():
        finding = findings.get(finding_id)
        if finding is None:
            dropped.append({"text": text, "finding_id": finding_id, "violations": ["없는 관계 ID 에 대한 설명"],
                            "stage": "explanation", "revised_attempted": False})
            continue
        # 설명은 해당 상충의 근거를 인용한 문장으로 보고 같은 검사를 한다 (조건 필수 수치 C4 는 제외)
        as_claim = {"text": text or "", "technology": finding["technology"], "evidence_ids": finding["evidence_ids"]}
        thresholds = RULE_THRESHOLD_NUMBERS.get(finding.get("rule_id"), ())
        quoted_claims = " ".join([ctx["claim_texts"].get(r, "") for r in finding["record_refs"]] + list(thresholds))
        rule_ctx = ctx
        if finding.get("rule_id") == "SX1":  # "운영 환경(TRL 7)"은 규칙 기준이라 C7 에서 허용
            rule_ctx = {**ctx, "trl": {t: v | {(7, 7)} for t, v in ctx["trl"].items()}}
        bad = [x for x in check_claim(as_claim, rule_ctx, quoted_claims)
               if x.startswith(("C3", "C5", "C7 매트릭스", "검사 규칙 코드")) or x.startswith("길이")]
        if bad:
            dropped.append({"text": text, "finding_id": finding_id, "violations": bad,
                            "stage": "explanation", "revised_attempted": False})
            continue
        checked_explanations[finding_id] = (text, resolved)

    return {"claims": kept, "explanations": checked_explanations, "dropped": dropped, "revised": revised}

"""② 관점 간 관계 분석. LLM 없이 코드로만 판정한다.

설계서 그림의 "보완 / 상충 / 일치" 세 갈래는 경로 분기가 아니라 레코드 쌍의 분류이므로 한 단계로 합쳤다.
결과는 모두 CrossFinding 하나의 목록에 kind 로 구분해 담는다.

  conflict        SX1~SX7 (§8.6.1)
  shared_evidence 서로 다른 관점이 같은 근거(content_hash)를 씀 → 일치로 세지 않음 (이중 계산 방지)
  agreement       같은 assessment_vocab + 같은 방향 (§2.2.6: 어휘가 다르면 일치로 세지 않음)
  complement      다른 assessment_vocab + 같은 방향, 또는 technology="both" 주장이 병행을 직접 언급
"""

from __future__ import annotations

from datetime import date
from itertools import combinations

from agents.synthesis.matrix import content_key, record_ref
from agents.synthesis.rules import (
    EVIDENCE_LEVEL_RANK,
    NUMERIC_GAP_RATIO,
    PERSPECTIVES,
    TECHNOLOGIES,
    TEMPORAL_GAP_DAYS,
    condition_violations,
    numbers_in,
    trl_bounds,
)


def direction(record: dict) -> str:
    """record 의 판단 방향. stance_counts 에서 support 와 counter 중 많은 쪽."""
    counts = record.get("stance_counts") or {}
    support, counter = counts.get("support", 0), counts.get("counter", 0)
    if support > counter:
        return "positive"
    if counter > support:
        return "negative"
    return "neutral"


def _records(findings: dict) -> list[dict]:
    out = []
    for perspective in PERSPECTIVES:
        for record in (findings.get(perspective) or {}).get("records") or []:
            if record.get("technology") in TECHNOLOGIES and "criterion" in record:
                out.append(record)
    out.sort(key=lambda r: (TECHNOLOGIES.index(r["technology"]), PERSPECTIVES.index(r["perspective"]), r["criterion"]))
    return out


def _claims(findings: dict) -> list[dict]:
    return [c for p in PERSPECTIVES for c in (findings.get(p) or {}).get("claims") or []]


def _finding(kind, technology, refs, evidence_ids, rule_id=None, conflict_type=None, note=None):
    return {
        "id": "",  # 정렬 후 부여
        "kind": kind,
        "conflict_type": conflict_type,
        "technology": technology,
        "record_refs": sorted(set(refs)),
        "evidence_ids": sorted(set(evidence_ids)),
        "detected_by": "rule",
        "rule_id": rule_id,
        "explanation": None,
        "resolution": "unresolved",
        "_note": note,  # 서술 단계에 넘기는 규칙 설명. 최종 결과에서는 제거
    }


def _date(text):
    try:
        return date.fromisoformat((text or "")[:10])
    except ValueError:
        return None


def analyze_relations(state: dict) -> dict:
    findings = state.get("findings") or {}
    store = state.get("evidence_store") or {}
    records = _records(findings)
    claims = _claims(findings)
    out: list[dict] = []

    by_tech = {t: [r for r in records if r["technology"] == t] for t in TECHNOLOGIES}

    for tech, recs in by_tech.items():
        # --- 쌍 단위: shared_evidence / agreement / complement / SX2 / SX5 ---
        for a, b in combinations(recs, 2):
            if a["perspective"] == b["perspective"]:
                continue
            keys_a = {content_key(e, store) for e in a["evidence_ids"] if e in store}
            keys_b = {content_key(e, store) for e in b["evidence_ids"] if e in store}
            refs = [record_ref(a), record_ref(b)]
            ids = [e for e in a["evidence_ids"] + b["evidence_ids"] if e in store]
            da, db = direction(a), direction(b)

            shared = keys_a & keys_b
            if shared:
                shared_ids = [e for e in ids if content_key(e, store) in shared]
                out.append(_finding("shared_evidence", tech, refs, shared_ids,
                                    note=f"두 관점이 같은 근거 {len(shared)}건을 공유함. 독립된 두 확인이 아님"))
            elif da == db and da != "neutral":
                if a["assessment_vocab"] == b["assessment_vocab"]:
                    out.append(_finding("agreement", tech, refs, ids, note="같은 질문(어휘)에 같은 방향 판단"))
                else:
                    out.append(_finding("complement", tech, refs, ids,
                                        note=f"서로 다른 질문({a['assessment_vocab']}, {b['assessment_vocab']})에 같은 방향. 일치가 아니라 보완"))

            if {da, db} == {"positive", "negative"}:
                dates_a = [d for d in (_date(store[e].get("published_at")) for e in a["evidence_ids"] if e in store) if d]
                dates_b = [d for d in (_date(store[e].get("published_at")) for e in b["evidence_ids"] if e in store) if d]
                if dates_a and dates_b and abs((max(dates_a) - max(dates_b)).days) > TEMPORAL_GAP_DAYS:
                    out.append(_finding("conflict", tech, refs, ids, "SX5", "temporal",
                                        f"반대 방향 결론의 근거 날짜 차이가 {TEMPORAL_GAP_DAYS}일 초과"))

        # --- SX2: 기술군(class) 근거 record 와 방향이 다른 대상 기술(direct) record. class record 하나당 한 건 ---
        judged = [r for r in recs if r["basis"] in ("direct", "inferred")]
        for c in (r for r in judged if r["scope"] == "class"):
            others = [r for r in judged if r["scope"] == "direct" and r["perspective"] != c["perspective"]
                      and direction(r) != direction(c)]
            if others:
                out.append(_finding("conflict", tech, [record_ref(c)] + [record_ref(r) for r in others],
                                    [e for r in [c] + others for e in r["evidence_ids"] if e in store], "SX2", "scope",
                                    "한 관점은 기술군(class), 다른 관점은 대상 기술(direct) 근거에 기댐. 범위 불일치일 수 있음"))

        # --- SX1: 시장 증거 수준 ≥ announcement 인데 TRL 은 운영 환경(7단계) 근거 없음 ---
        trl = [r for r in recs if r["assessment_vocab"] == "trl_stage"]
        market = [r for r in recs if r["perspective"] == "market"
                  and EVIDENCE_LEVEL_RANK.get(r["evidence_level"], 0) >= EVIDENCE_LEVEL_RANK["announcement"]]
        for t in trl:
            bounds = trl_bounds(t.get("value"))
            if bounds and bounds[1] < 7:
                for m in market:
                    out.append(_finding("conflict", tech, [record_ref(t), record_ref(m)],
                                        [e for e in t["evidence_ids"] + m["evidence_ids"] if e in store], "SX1", "evidence_level",
                                        f"시장 증거 수준 {m['evidence_level']} 이지만 TRL {t['value']} 은 운영 환경(7단계) 근거가 없음"))

        # --- SX7: TRL 판정 이후 들어온 성숙도 근거 (pilot 이상, TRL 입력에 없음) ---
        technical = findings.get("technical") or {}
        seen = set(technical.get("input_evidence_ids") or [])
        if trl and technical:
            for r in recs:
                if r["perspective"] not in ("market", "stakeholder"):
                    continue
                late = [e for e in r["evidence_ids"] if e in store and e not in seen
                        and EVIDENCE_LEVEL_RANK.get(store[e].get("evidence_level", "unknown"), 0) >= EVIDENCE_LEVEL_RANK["pilot"]]
                if late:
                    out.append(_finding("conflict", tech, [record_ref(t) for t in trl] + [record_ref(r)], late, "SX7", "trl_input",
                                        "TRL 판정 입력에 없던 pilot 이상 근거가 다른 관점에 있음. TRL 재확인 필요"))

    # --- SX3: 관점 주장에서 조건 필수 수치의 조건 누락 ---
    for claim in claims:
        text = " ".join([claim.get("text", "")] + list(claim.get("conditions") or []))
        problems = condition_violations(text, claim.get("technology"))
        if problems:
            out.append(_finding("conflict", claim.get("technology", "both"), [f"{claim.get('perspective')}/claim/{claim.get('claim_id')}"],
                                [e for e in claim.get("evidence_ids", []) if e in store], "SX3", "condition", "; ".join(problems)))

    # --- SX4: 같은 metric_tag 수치가 10% 넘게 차이 ---
    metric_values: dict[tuple, list[tuple[float, str, str]]] = {}
    for claim in claims:
        for eid in claim.get("evidence_ids", []):
            ev = store.get(eid) or {}
            tag = ev.get("metric_tag")
            nums = numbers_in(ev.get("quote", ""))
            if tag and nums:
                metric_values.setdefault((tag, claim.get("technology", "both")), []).append(
                    (float(nums[0]), eid, f"{claim.get('perspective')}/claim/{claim.get('claim_id')}"))
    for (tag, tech), values in sorted(metric_values.items()):
        nums = [v for v, _, _ in values]
        if len(set(nums)) > 1 and min(nums) > 0 and (max(nums) - min(nums)) / min(nums) > NUMERIC_GAP_RATIO:
            out.append(_finding("conflict", tech, [ref for _, _, ref in values], [e for _, e, _ in values], "SX4", "numeric",
                                f"metric_tag={tag} 값 차이 {min(nums)}~{max(nums)} (임계 {int(NUMERIC_GAP_RATIO * 100)}% 초과)"))

    # --- SX6: 같은 근거(content_hash)에 support 와 counter 가 공존 ---
    stances: dict[str, dict[str, set]] = {}
    for eid, ev in store.items():
        stances.setdefault(content_key(eid, store), {}).setdefault(ev.get("stance"), set()).add(eid)
    for key, by_stance in sorted(stances.items()):
        if "support" in by_stance and "counter" in by_stance:
            ids = sorted(by_stance["support"] | by_stance["counter"])
            refs = [record_ref(r) for r in records if set(r["evidence_ids"]) & set(ids)]
            techs = {r["technology"] for r in records if set(r["evidence_ids"]) & set(ids)}
            out.append(_finding("conflict", techs.pop() if len(techs) == 1 else "both", refs, ids, "SX6", "stance",
                                "같은 근거가 support 와 counter 로 동시에 쓰임"))

    # --- 보완: technology="both" 주장이 병행을 직접 언급할 때만 (§9.1) ---
    for claim in claims:
        if claim.get("technology") == "both":
            out.append(_finding("complement", "both", [f"{claim.get('perspective')}/claim/{claim.get('claim_id')}"],
                                [e for e in claim.get("evidence_ids", []) if e in store],
                                note="SW·HW 병행을 직접 언급한 주장"))

    out = _dedupe_and_number(out)
    return {"cross_findings": out, "contrast_table": contrast_table(records, store, out)}


def _dedupe_and_number(items: list[dict]) -> list[dict]:
    order = {"conflict": 0, "shared_evidence": 1, "agreement": 2, "complement": 3}
    seen, unique = set(), []
    for item in items:
        key = (item["kind"], item["rule_id"], item["technology"], tuple(item["record_refs"]))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    unique.sort(key=lambda f: (order[f["kind"]], f["rule_id"] or "", f["technology"], f["record_refs"]))
    counters: dict[str, int] = {}
    for item in unique:
        prefix = (item["rule_id"] or item["kind"]).lower()
        counters[prefix] = counters.get(prefix, 0) + 1
        item["id"] = f"{prefix}:{item['technology']}:{counters[prefix]:03d}"
    return unique


def contrast_table(records: list[dict], store: dict, cross_findings: list[dict]) -> list[dict]:
    """관점·기준별 SW/HW 대조표. 두 기술은 다른 레이어라 기본 관계는 independent 다.

    한쪽 기술에 record 가 없으면 unknown, technology="both" 보완 주장이 있는 관점은 complement.
    """
    rows: dict[tuple, dict] = {}
    for r in records:
        key = (PERSPECTIVES.index(r["perspective"]), r["perspective"], r["criterion"])
        row = rows.setdefault(key, {"criterion": f"{r['perspective']}/{r['criterion']}", "sw": None, "hw": None,
                                    "relation": "independent", "evidence_ids": set()})
        value = f" {r['value']}" if r.get("value") else ""
        n = len({content_key(e, store) for e in r["evidence_ids"] if e in store})
        row[r["technology"]] = f"{r['assessment']}{value} (basis={r['basis']}, {r['evidence_level']}, 근거 {n}건)"
        row["evidence_ids"] |= {e for e in r["evidence_ids"] if e in store}
    both = {ref.split("/")[0] for f in cross_findings if f["kind"] == "complement" and f["technology"] == "both"
            for ref in f["record_refs"]}
    out = []
    for key in sorted(rows):
        row = rows[key]
        if row["sw"] is None or row["hw"] is None:
            row["relation"] = "unknown"
        elif key[1] in both:
            row["relation"] = "complement"
        row["evidence_ids"] = sorted(row["evidence_ids"])
        out.append(row)
    return out

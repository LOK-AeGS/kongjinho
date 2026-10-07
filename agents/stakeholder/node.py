"""이해관계자 에이전트 노드: 부모 AppState ↔ 원문 검증형 내부 서브그래프 변환.

부모 그래프가 아는 것은 make_node 하나뿐이다. 검색·원문 확보·검증 반복은 subgraph.py 에 있다.
LLM 이 쓴 인용문·URL·날짜는 원문 snapshot 과 코드로 대조하고, 통과한 것만 evidence_store 에 넣는다.
통과하지 못한 항목과 찾지 못한 반응은 가짜 근거 대신 gaps / search_outcomes 로 남긴다.
"""
from functools import partial

from agents.stakeholder.subgraph import GROUP_LABELS, GROUPS, default_request, run_stakeholder

LEVEL_RANK = {"unknown": 0, "forecast": 1, "announcement": 2, "pilot": 3, "production": 4}
LABEL_TO_GROUP = {label: group for group, label in GROUP_LABELS.items()}
MAX_SEARCH_LOG = 30


def build_request(state: dict) -> dict:
    domain = str(state.get("domain") or "")
    if "datacenter" not in domain:
        raise ValueError(f"이해관계자 평가는 데이터센터 도메인만 지원합니다: {domain!r}")
    parent = state.get("request") or {}
    hint = state.get("rework_hint") or {}
    request = default_request(parent.get("as_of"))
    # 재작업 때 늘어난 라운드 수는 Supervisor 워커 래퍼가 request.max_search_rounds 에 이미 더해 준다.
    request["max_search_rounds"] = max(1, min(int(parent.get("max_search_rounds") or request["max_search_rounds"]), 3))
    request["max_queries"] = min(24, 2 * 2 * len(GROUPS) * request["max_search_rounds"])  # 쌍당 질의 2개 × 라운드
    request["domain"] = "datacenter"
    for side in ("sw", "hw"):
        spec = (state.get("selected_tech") or {}).get(side)
        if spec:
            request[side] = {"name": spec["name"], "selection_reason": spec.get("selection_reason", ""), "seed_urls": []}
    if hint:
        request["focus_queries"] = list(hint.get("focus_queries") or [])
        pairs = []
        for gap in hint.get("gaps") or []:
            group = LABEL_TO_GROUP.get(gap.get("criterion"))
            if group and gap.get("technology") in ("sw", "hw"):
                pairs.append([gap["technology"], group])
        request["only_pairs"] = pairs
    return request


def project_technical(findings: dict | None) -> dict | None:
    """다른 관점의 결론은 보지 않고 기술 조사의 주장 문장만 맥락으로 넘긴다."""
    if not findings:
        return None
    return {"claims": [c["text"] for c in (findings.get("claims") or [])[:8]]}


def to_evidence(ev: dict) -> dict:
    return {
        "id": ev["id"], "claim_id": ev["claim_id"], "doc_id": None, "title": ev["title"],
        "author_or_org": ev["author_or_org"], "source_type": ev["source_type"],
        "primary_or_secondary": ev["primary_or_secondary"], "direct_or_proxy": ev["direct_or_proxy"],
        "url": ev["url"], "published_at": ev["published_at"], "accessed_at": ev["accessed_at"],
        "page_or_locator": ev["page_or_locator"], "quote": ev["quote"], "stance": ev["stance"],
        "evidence_level": ev.get("evidence_level", "unknown"), "metric_tag": None,
        "perspective": "stakeholder", "content_hash": ev["content_hash"],
    }


def _dominant(counts: dict) -> str:
    top = max(counts.values())
    leaders = [stance for stance, n in counts.items() if n == top]
    return leaders[0] if len(leaders) == 1 else "neutral"


def to_app_update(final: dict) -> dict:
    result = final["result"]
    completion = result["completion"]
    store = {eid: to_evidence(ev) for eid, ev in result["evidence_store"].items()}
    claim_by_id = {c["claim_id"]: c for c in result["claims"]}
    position_by_claim = {cid: p for p in result["positions"] for cid in p["claim_ids"]}

    claims = []
    for cid, claim in claim_by_id.items():
        position = position_by_claim[cid]
        limitations = [x for x in [claim["uncertainty"], *position["bias_notes"]] if x]
        if position["target_scope"] != "selected_technology":
            limitations.append("선정 기술 자체가 아닌 기술 계열·기타 대상에 관한 의견")
        claims.append({
            "claim_id": cid, "technology": claim["technology_ids"][0], "perspective": "stakeholder",
            "text": f"[{GROUP_LABELS[position['group']]}] {position['speaker']}: {claim['statement']}",
            "evidence_ids": claim["evidence_ids"], "conditions": claim["conditions"], "limitations": limitations,
        })

    outcomes = result["search_outcomes"]
    records, gaps = [], []
    for tech in ("sw", "hw"):
        for group in GROUPS:
            positions = [p for p in result["positions"] if p["technology_id"] == tech and p["group"] == group]
            statuses = {o["stance"]: o["status"] for o in outcomes if o["technology_id"] == tech and o["group"] == group}
            if not positions:
                unresolved = [s for s in statuses.values() if s in ("blocked", "unsearched")]
                reason = ("검색·검증이 끝나지 않음(" + ", ".join(sorted(set(unresolved))) + ")" if unresolved
                          else "not_found: 탐색했으나 확인 가능한 반응을 찾지 못함 (의견 부재의 증명은 아님)")
                gaps.append({"technology": tech, "perspective": "stakeholder", "criterion": GROUP_LABELS[group],
                             "reason": reason, "missing_evidence": []})
                continue
            counts = {"support": 0, "counter": 0, "neutral": 0}
            for p in positions:
                counts[p["evidence_stance"]] += 1
            n_direct = sum(p["target_scope"] == "selected_technology" for p in positions)
            scope = "direct" if n_direct == len(positions) else "class" if n_direct == 0 else "mixed"
            evidence_ids = sorted({eid for p in positions for cid in p["claim_ids"] for eid in claim_by_id[cid]["evidence_ids"]})
            level = max((store[e]["evidence_level"] for e in evidence_ids), key=LEVEL_RANK.get)
            limitations = [f"{stance} 방향: 탐색했으나 확인되지 않음(not_found)"
                           for stance in ("support", "counter") if counts[stance] == 0 and statuses.get(stance) == "not_found"]
            dominant = _dominant(counts)
            records.append({
                "technology": tech, "perspective": "stakeholder", "criterion": GROUP_LABELS[group],
                "basis": "direct" if scope == "direct" else "inferred", "evidence_level": level, "scope": scope,
                "stance_counts": counts, "evidence_ids": evidence_ids,
                "assessment": dominant, "assessment_vocab": "support/counter/neutral", "value": None,
                "findings": f"{GROUP_LABELS[group]} 발언 {len(positions)}건 중 {dominant} 우세"
                            + (" (의견 엇갈림)" if counts["support"] and counts["counter"] else ""),
                "limitations": limitations,
            })

    if not claims:
        status = "failed" if completion["errors"] else "partial"
    else:
        status = completion["status"]
    findings = {"perspective": "stakeholder", "status": status, "records": records, "claims": claims, "gaps": gaps,
                "limitations": (completion["gaps"] + completion["errors"])[:20], "input_evidence_ids": sorted(store)}

    logs = [{"id": log["id"], "technology": log["technology_id"], "group": log["group"], "status": log["status"],
             "n_urls": len(log.get("urls", [])), "retry": log["id"].endswith(":retry")}
            for batch in final.get("batches", []) for log in batch.get("search_logs", [])][-MAX_SEARCH_LOG:]
    quality = {
        "status": "failed" if not claims and completion["errors"] else "needs_review" if completion["gaps"] or completion["errors"] else "passed",
        "violations": list(completion["errors"]), "warnings": completion["gaps"][:20], "checked_claim_ids": list(claim_by_id),
    }
    return {"stakeholder_findings": findings, "evidence_store": store,
            "search_log_by_perspective": {"stakeholder": logs},
            "quality_by_perspective": {"stakeholder": quality}}


def merge_findings(previous: dict | None, new: dict, retried: set[tuple[str, str]]) -> dict:
    """재작업 결과를 이전 결과와 합친다. 재작업은 gap 이 있는 쌍만 다시 조사하므로 덮어쓰면 앞서 얻은 주장이 사라진다.

    - claims: claim_id 기준 합집합
    - records: 같은 (기술, 그룹)은 지지/반대/중립 건수와 근거 ID 를 합산하고 우세 입장을 다시 계산
    - gaps: 합친 records 에 없는 쌍만 남기되, 이번에 다시 조사한 쌍은 새 사유를, 안 한 쌍은 이전 사유를 쓴다
    """
    if not previous:
        return new
    claims = {c["claim_id"]: c for c in [*previous["claims"], *new["claims"]]}
    records = {}
    for record in [*previous["records"], *new["records"]]:
        key = (record["technology"], record["criterion"])
        if key not in records:
            records[key] = {**record, "stance_counts": dict(record["stance_counts"]), "evidence_ids": list(record["evidence_ids"])}
            continue
        merged = records[key]
        for stance, n in record["stance_counts"].items():
            merged["stance_counts"][stance] = merged["stance_counts"].get(stance, 0) + n
        merged["evidence_ids"] = sorted(set(merged["evidence_ids"]) | set(record["evidence_ids"]))
        merged["scope"] = merged["scope"] if merged["scope"] == record["scope"] else "mixed"
        merged["basis"] = "direct" if merged["scope"] == "direct" else "inferred"
        merged["evidence_level"] = max(merged["evidence_level"], record["evidence_level"], key=LEVEL_RANK.get)
        merged["assessment"] = _dominant(merged["stance_counts"])
        merged["limitations"] = list(dict.fromkeys([*merged["limitations"], *record["limitations"]]))
        total = sum(merged["stance_counts"].values())
        merged["findings"] = f"{merged['criterion']} 발언 {total}건 중 {merged['assessment']} 우세"
    covered = set(records)
    new_gaps = {(g["technology"], g["criterion"]): g for g in new["gaps"]}
    old_gaps = {(g["technology"], g["criterion"]): g for g in previous["gaps"]}
    gaps = []
    for key in sorted(set(new_gaps) | set(old_gaps)):
        if key in covered:
            continue
        retried_now = (key[0], LABEL_TO_GROUP.get(key[1])) in retried
        gaps.append(new_gaps[key] if retried_now and key in new_gaps else old_gaps.get(key, new_gaps.get(key)))
    limitations = list(dict.fromkeys([*new["limitations"], *previous["limitations"]]))[:20]
    status = "partial" if gaps or (new["status"] != "complete") else "complete"
    if not claims:
        status = "failed" if new["status"] == "failed" else "partial"
    return {"perspective": "stakeholder", "status": status, "records": list(records.values()), "claims": list(claims.values()),
            "gaps": gaps, "limitations": limitations,
            "input_evidence_ids": sorted(set(previous["input_evidence_ids"]) | set(new["input_evidence_ids"]))}


def stakeholder_node(state: dict, *, backend=None) -> dict:
    request = build_request(state)
    final = run_stakeholder(request, project_technical(state.get("technical_findings")), backend)
    update = to_app_update(final)
    if state.get("rework_hint") and state.get("stakeholder_findings"):
        retried = {(t, g) for t, g in request.get("only_pairs") or []}
        update["stakeholder_findings"] = merge_findings(state["stakeholder_findings"], update["stakeholder_findings"], retried)
        merged_ids = {cid for c in update["stakeholder_findings"]["claims"] for cid in [c["claim_id"]]}
        quality = update["quality_by_perspective"]["stakeholder"]
        quality["checked_claim_ids"] = sorted(merged_ids)
    return update


def make_node(backend=None):
    """부모 그래프(graph/build.py)의 stakeholder= 자리에 꽂을 노드 함수."""
    return partial(stakeholder_node, backend=backend)


__all__ = ["make_node", "build_request", "to_app_update", "merge_findings"]

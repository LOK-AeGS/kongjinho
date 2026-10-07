"""부모 AppState와 원문 검증형 stakeholder 서브그래프 사이의 변환."""

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
    request["max_search_rounds"] = max(
        1, min(int(parent.get("max_search_rounds") or request["max_search_rounds"]), 3)
    )
    request["max_queries"] = min(
        24, 2 * len(GROUPS) * request["max_search_rounds"] + 2
    )
    request["domain"] = "datacenter"
    for side in ("sw", "hw"):
        spec = (state.get("selected_tech") or {}).get(side)
        if spec:
            request[side] = {
                "name": spec["name"],
                "selection_reason": spec.get("selection_reason", ""),
                "seed_urls": [],
            }
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
    if not findings:
        return None
    return {"claims": [claim["text"] for claim in (findings.get("claims") or [])[:8]]}


def to_evidence(evidence: dict) -> dict:
    return {
        "id": evidence["id"],
        "claim_id": evidence["claim_id"],
        "doc_id": None,
        "title": evidence["title"],
        "author_or_org": evidence["author_or_org"],
        "source_type": evidence["source_type"],
        "primary_or_secondary": evidence["primary_or_secondary"],
        "direct_or_proxy": evidence["direct_or_proxy"],
        "url": evidence["url"],
        "published_at": evidence["published_at"],
        "accessed_at": evidence["accessed_at"],
        "page_or_locator": evidence["page_or_locator"],
        "quote": evidence["quote"],
        "stance": evidence["stance"],
        "evidence_level": evidence.get("evidence_level", "unknown"),
        "metric_tag": None,
        "perspective": "stakeholder",
        "content_hash": evidence["content_hash"],
    }


def _dominant(counts: dict) -> str:
    top = max(counts.values())
    leaders = [stance for stance, count in counts.items() if count == top]
    return leaders[0] if len(leaders) == 1 else "neutral"


def to_app_update(final: dict) -> dict:
    result = final["result"]
    completion = result["completion"]
    store = {
        evidence_id: to_evidence(evidence)
        for evidence_id, evidence in result["evidence_store"].items()
    }
    claim_by_id = {claim["claim_id"]: claim for claim in result["claims"]}
    position_by_claim = {
        claim_id: position
        for position in result["positions"]
        for claim_id in position["claim_ids"]
    }

    claims = []
    for claim_id, claim in claim_by_id.items():
        position = position_by_claim[claim_id]
        limitations = [
            value for value in [claim["uncertainty"], *position["bias_notes"]] if value
        ]
        if position["target_scope"] != "selected_technology":
            limitations.append("선정 기술 자체가 아닌 기술 계열·기타 대상에 관한 의견")
        claims.append({
            "claim_id": claim_id,
            "technology": claim["technology_ids"][0],
            "perspective": "stakeholder",
            "text": f"[{GROUP_LABELS[position['group']]}] {position['speaker']}: {claim['statement']}",
            "evidence_ids": claim["evidence_ids"],
            "conditions": claim["conditions"],
            "limitations": limitations,
        })

    outcomes = result["search_outcomes"]
    records, gaps = [], []
    for technology in ("sw", "hw"):
        for group in GROUPS:
            positions = [
                position for position in result["positions"]
                if position["technology_id"] == technology and position["group"] == group
            ]
            statuses = {
                outcome["stance"]: outcome["status"]
                for outcome in outcomes
                if outcome["technology_id"] == technology and outcome["group"] == group
            }
            if not positions:
                unresolved = [
                    status for status in statuses.values()
                    if status in ("blocked", "unsearched")
                ]
                reason = (
                    "검색·검증이 끝나지 않음(" + ", ".join(sorted(set(unresolved))) + ")"
                    if unresolved
                    else "not_found: 탐색했으나 확인 가능한 반응을 찾지 못함 (의견 부재의 증명은 아님)"
                )
                gaps.append({
                    "technology": technology,
                    "perspective": "stakeholder",
                    "criterion": GROUP_LABELS[group],
                    "reason": reason,
                    "missing_evidence": [],
                })
                continue
            counts = {"support": 0, "counter": 0, "neutral": 0}
            for position in positions:
                counts[position["evidence_stance"]] += 1
            direct_count = sum(
                position["target_scope"] == "selected_technology" for position in positions
            )
            scope = (
                "direct" if direct_count == len(positions)
                else "class" if direct_count == 0
                else "mixed"
            )
            evidence_ids = sorted({
                evidence_id
                for position in positions
                for claim_id in position["claim_ids"]
                for evidence_id in claim_by_id[claim_id]["evidence_ids"]
            })
            level = max(
                (store[evidence_id]["evidence_level"] for evidence_id in evidence_ids),
                key=LEVEL_RANK.get,
            )
            limitations = [
                f"{stance} 방향: 탐색했으나 확인되지 않음(not_found)"
                for stance in ("support", "counter")
                if counts[stance] == 0 and statuses.get(stance) == "not_found"
            ]
            dominant = _dominant(counts)
            records.append({
                "technology": technology,
                "perspective": "stakeholder",
                "criterion": GROUP_LABELS[group],
                "basis": "direct" if scope == "direct" else "inferred",
                "evidence_level": level,
                "scope": scope,
                "stance_counts": counts,
                "evidence_ids": evidence_ids,
                "assessment": dominant,
                "assessment_vocab": "support/counter/neutral",
                "value": None,
                "findings": f"{GROUP_LABELS[group]} 발언 {len(positions)}건 중 {dominant} 우세"
                + (" (의견 엇갈림)" if counts["support"] and counts["counter"] else ""),
                "limitations": limitations,
            })

    status = (
        "failed" if not claims and completion["errors"]
        else completion["status"] if claims
        else "partial"
    )
    findings = {
        "perspective": "stakeholder",
        "status": status,
        "records": records,
        "claims": claims,
        "gaps": gaps,
        "limitations": (completion["gaps"] + completion["errors"])[:20],
        "input_evidence_ids": sorted(store),
    }
    logs = [{
        "id": log["id"],
        "technology": log["technology_id"],
        "group": log["group"],
        "status": log["status"],
        "n_urls": len(log.get("urls", [])),
        "retry": log["id"].endswith(":retry"),
    } for batch in final.get("batches", []) for log in batch.get("search_logs", [])][-MAX_SEARCH_LOG:]
    quality = {
        "status": (
            "failed" if not claims and completion["errors"]
            else "needs_review" if completion["gaps"] or completion["errors"]
            else "passed"
        ),
        "violations": list(completion["errors"]),
        "warnings": completion["gaps"][:20],
        "checked_claim_ids": list(claim_by_id),
    }
    return {
        "stakeholder_findings": findings,
        "evidence_store": store,
        "search_log_by_perspective": {"stakeholder": logs},
        "quality_by_perspective": {"stakeholder": quality},
    }


def stakeholder_node(state: dict, *, backend=None) -> dict:
    request = build_request(state)
    final = run_stakeholder(
        request,
        project_technical(state.get("technical_findings")),
        backend,
    )
    return to_app_update(final)


def make_node(backend=None):
    return partial(stakeholder_node, backend=backend)


__all__ = ["make_node", "build_request", "to_app_update"]

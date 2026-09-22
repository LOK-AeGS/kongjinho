"""기술조사 내부 LangGraph: 수집 → 추출 → TRL Gate → 검증."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph

from .config import MAX_REVISION_ROUNDS, TOP_K_PER_QUERY
from .evidence import make_claim_id, merge_evidence, to_parent_evidence, validate_reference
from .models import TechnicalExtraction
from .retrieval import CRITERIA, coverage_by_technology, merge_candidates, missing_criteria, select_query
from .state import TechnicalLocalState
from .trl import determine_trl, validate_metric_records, validate_trl_record
from .web import web_query


@dataclass(frozen=True)
class TechnicalAgentDeps:
    retriever: Any
    web_provider: Any
    analyzer: Any
    top_k: int = TOP_K_PER_QUERY


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _compact_candidate(candidate: dict) -> dict:
    return {
        "candidate_id": candidate["chunk_id"],
        "technology": candidate.get("technology"),
        "title": candidate.get("title"),
        "role": candidate.get("role"),
        "origin": candidate.get("origin", "pdf"),
        "url": candidate.get("url"),
        "published_at": candidate.get("published_at"),
        "locator": candidate.get("locator"),
        "matched_criteria": candidate.get("matched_criteria", []),
        "text": candidate.get("text", ""),
    }


def _bind_refs(refs: list[dict], candidate_map: dict[str, dict], claim_id: str, evidence_level: str = "unknown") -> tuple[list[str], dict[str, dict], list[str]]:
    evidence_ids: list[str] = []
    store: dict[str, dict] = {}
    violations: list[str] = []
    for reference in refs:
        candidate, error = validate_reference(reference, candidate_map)
        if error:
            violations.append(error)
            continue
        evidence = to_parent_evidence(candidate, reference["quote"], claim_id)
        evidence["evidence_level"] = evidence_level
        evidence_ids.append(evidence["id"])
        store[evidence["id"]] = evidence
    return _unique(evidence_ids), store, violations


def _bind_extraction(raw: TechnicalExtraction, candidates: list[dict]) -> tuple[dict, dict[str, dict], list[str]]:
    candidate_map = {candidate["chunk_id"]: candidate for candidate in candidates}
    store: dict[str, dict] = {}
    violations: list[str] = []
    records: list[dict] = []
    claims: list[dict] = []
    metrics: list[dict] = []
    observations: list[dict] = []

    for item in raw.records:
        value = item.model_dump()
        provisional = make_claim_id(value["technology"], value["findings"], [])
        ids, delta, errors = _bind_refs(value.pop("evidence_refs"), candidate_map, provisional)
        claim_id = make_claim_id(value["technology"], value["findings"], ids)
        for evidence in delta.values():
            evidence["claim_id"] = claim_id
        value.update({"claim_id": claim_id, "evidence_ids": ids})
        records.append(value)
        store = merge_evidence(store, delta)
        violations.extend(errors)

    for item in raw.claims:
        value = item.model_dump()
        provisional = make_claim_id(value["technology"], value["text"], [])
        ids, delta, errors = _bind_refs(value.pop("evidence_refs"), candidate_map, provisional)
        claim_id = make_claim_id(value["technology"], value["text"], ids)
        for evidence in delta.values():
            evidence["claim_id"] = claim_id
        value.update({"claim_id": claim_id, "evidence_ids": ids})
        claims.append(value)
        store = merge_evidence(store, delta)
        violations.extend(errors)

    for item in raw.metrics:
        value = item.model_dump()
        metric_text = " | ".join(str(value.get(key) or "") for key in ("metric", "value", "baseline"))
        provisional = make_claim_id(value["technology"], metric_text, [])
        ids, delta, errors = _bind_refs(value.pop("evidence_refs"), candidate_map, provisional)
        claim_id = make_claim_id(value["technology"], metric_text, ids)
        for evidence in delta.values():
            evidence["claim_id"] = claim_id
        value.update({"claim_id": claim_id, "evidence_ids": ids})
        metrics.append(value)
        store = merge_evidence(store, delta)
        violations.extend(errors)

    for item in raw.readiness_observations:
        value = item.model_dump()
        provisional = make_claim_id(value["technology"], value["summary"], [])
        ids, delta, errors = _bind_refs(
            value.pop("evidence_refs"), candidate_map, provisional, value["evidence_level"]
        )
        claim_id = make_claim_id(value["technology"], value["summary"], ids)
        for evidence in delta.values():
            evidence["claim_id"] = claim_id
        value.update({"claim_id": claim_id, "evidence_ids": ids})
        observations.append(value)
        store = merge_evidence(store, delta)
        violations.extend(errors)

    return {
        "records": records,
        "claims": claims,
        "metrics": metrics,
        "readiness_observations": observations,
        "gaps": raw.gaps,
    }, store, _unique(violations)


def _evidence_level(ids: list[str], store: dict[str, dict]) -> str:
    rank = {"unknown": 0, "forecast": 1, "announcement": 2, "pilot": 3, "production": 4}
    values = [store[item].get("evidence_level", "unknown") for item in ids if item in store]
    return max(values, key=lambda value: rank.get(value, 0), default="unknown")


def _make_findings(state: TechnicalLocalState, trl_records: list[dict], metrics: list[dict], warnings: list[str], quality_violations: list[str]) -> dict:
    extraction = state["extraction"]
    store = state["evidence_store"]
    records: list[dict] = []
    claims: list[dict] = []
    gaps: list[dict] = []

    for item in extraction.get("records", []):
        ids = item.get("evidence_ids", [])
        records.append({
            "technology": item["technology"],
            "perspective": "technical",
            "criterion": item["criterion"],
            "basis": "direct" if ids else "unknown",
            "evidence_level": _evidence_level(ids, store),
            "scope": item["source_scope"],
            "stance_counts": {"support": 0, "counter": 0, "neutral": len(ids)},
            "evidence_ids": ids,
            "assessment": item["assessment"],
            "assessment_vocab": "supported|conditional|unsupported|unknown",
            "value": None,
            "findings": item["findings"],
            "limitations": item["limitations"] + item["conditions"],
        })
        claims.append({
            "claim_id": item["claim_id"], "technology": item["technology"],
            "perspective": "technical", "text": item["findings"],
            "evidence_ids": ids, "conditions": item["conditions"],
            "limitations": item["limitations"],
        })

    for record in trl_records:
        ids = record["supporting_evidence"]
        range_text = "확인 불가" if not record["level_range"] else "–".join(map(str, record["level_range"]))
        records.append({
            "technology": record["technology"], "perspective": "technical", "criterion": "trl",
            "basis": "inferred" if record["level"] is not None else "unknown",
            "evidence_level": _evidence_level(ids, store), "scope": "direct",
            "stance_counts": {"support": len(ids), "counter": len(record["blocking_evidence"]), "neutral": 0},
            "evidence_ids": ids, "assessment": "estimated", "assessment_vocab": "estimated|unknown",
            "value": f"TRL {range_text}",
            "findings": f"공개 직접 근거의 연속 Gate 기준으로 {record['stage_name']} 단계로 추정한다.",
            "limitations": record["blocking_evidence"] + record["missing_evidence"],
        })

    for item in extraction.get("claims", []):
        claims.append({
            "claim_id": item["claim_id"], "technology": item["technology"],
            "perspective": "technical", "text": item["text"],
            "evidence_ids": item["evidence_ids"], "conditions": item["conditions"],
            "limitations": item["limitations"],
        })

    for item in metrics:
        text = " | ".join(str(item.get(key) or "") for key in ("metric", "value", "baseline"))
        claims.append({
            "claim_id": item["claim_id"], "technology": item["technology"],
            "perspective": "technical", "text": text,
            "evidence_ids": item["evidence_ids"], "conditions": [],
            "limitations": item.get("validation_issues", []),
        })
    for item in extraction.get("readiness_observations", []):
        claims.append({
            "claim_id": item["claim_id"], "technology": item["technology"],
            "perspective": "technical", "text": item["summary"],
            "evidence_ids": item["evidence_ids"], "conditions": [],
            "limitations": item["blocking_facts"] + item["missing_facts"],
        })
    claims = list({item["claim_id"]: item for item in claims}.values())

    gap_strings = _unique(state.get("gaps", []) + extraction.get("gaps", []) + state.get("reference_violations", []) + quality_violations)
    for reason in gap_strings:
        technology = "sw" if reason.startswith("sw:") else "hw" if reason.startswith("hw:") else "both"
        gaps.append({
            "technology": technology, "perspective": "technical", "criterion": "evidence",
            "reason": reason, "missing_evidence": [reason],
        })
    status = "failed" if not records and state.get("errors") else "partial" if gap_strings or state.get("errors") or warnings else "complete"
    return {
        "perspective": "technical", "status": status, "records": records, "claims": claims,
        "gaps": gaps, "limitations": _unique(state.get("errors", []) + warnings + quality_violations),
        "input_evidence_ids": sorted(store),
        "meta": {"trl_records": trl_records, "metrics": metrics},
    }


def build_technical_graph(deps: TechnicalAgentDeps):
    def collect_and_check(state: TechnicalLocalState) -> dict:
        attempt = state["search_round"] + 1
        candidates = list(state["candidate_evidence"])
        logs = list(state["search_logs"])
        errors = list(state["errors"])
        previous_missing = set(state.get("missing_criteria", []))
        for technology in ("sw", "hw"):
            for criterion in CRITERIA:
                key = f"{technology}:{criterion}"
                if attempt > 1 and key not in previous_missing:
                    continue
                query = select_query(technology, criterion, attempt)
                try:
                    hits = deps.retriever.search(technology, criterion, query, top_k=deps.top_k)
                    candidates = merge_candidates(candidates, [hit.to_candidate() for hit in hits])
                    logs.append({"provider": "local_pdf", "technology": technology, "criterion": criterion, "query": query, "round": attempt, "hits": len(hits), "status": "ok" if hits else "no_results"})
                except Exception as exc:
                    errors.append(f"{technology}:{criterion} PDF 검색 실패 ({type(exc).__name__})")

            operational_key = f"{technology}:operational_evidence"
            if attempt == 1 or operational_key in previous_missing:
                try:
                    web_candidates, log = deps.web_provider.search_and_extract(
                        technology, web_query(technology, attempt), state["request"]["as_of"]
                    )
                    candidates = merge_candidates(candidates, web_candidates)
                    logs.append({**log, "round": attempt})
                except Exception as exc:
                    # 타입명만 남기면 "패키지 미설치"와 "네트워크 오류"를 구분할 수 없어
                    # 설정 문제를 조사 실패로 오인하게 된다. 사유를 함께 남긴다.
                    reason = f"{type(exc).__name__}: {exc}"[:200]
                    errors.append(f"{technology} Tavily 검색 실패 ({reason})")
                    logs.append({"provider": "tavily", "technology": technology, "round": attempt, "status": "error", "error": reason})

        coverage = coverage_by_technology(candidates)
        missing = missing_criteria(coverage)
        for technology in ("sw", "hw"):
            if not any(item.get("origin") == "tavily" and item.get("technology") == technology for item in candidates):
                missing.append(f"{technology}:operational_evidence")
        max_rounds = state["request"]["max_search_rounds"]
        needs_retry = bool(missing) and attempt < max_rounds
        gaps = list(state["gaps"])
        if missing and not needs_retry:
            gaps.extend(
                f"{item}: 제한된 공개 검색에서 검증 가능한 근거를 확인하지 못함; 부재를 의미하지 않음"
                for item in missing
            )
        return {"candidate_evidence": candidates, "coverage": coverage, "missing_criteria": _unique(missing), "search_round": attempt, "needs_search_retry": needs_retry, "search_logs": logs, "gaps": _unique(gaps), "errors": _unique(errors)}

    def route_search(state: TechnicalLocalState) -> str:
        return "retry" if state["needs_search_retry"] else "extract"

    def extract_normalize_and_bind(state: TechnicalLocalState) -> dict:
        payload = {
            "request": state["request"], "selected_tech": state["selected_tech"],
            "coverage": state["coverage"], "missing_criteria": state["missing_criteria"],
            "candidate_evidence": [_compact_candidate(item) for item in state["candidate_evidence"]],
        }
        try:
            raw = deps.analyzer.extract(payload, state.get("reference_violations", []))
            extraction, evidence_store, violations = _bind_extraction(raw, state["candidate_evidence"])
            return {"extraction": extraction, "evidence_store": evidence_store, "reference_violations": violations}
        except Exception as exc:
            return {"extraction": {"records": [], "claims": [], "metrics": [], "readiness_observations": [], "gaps": []}, "evidence_store": {}, "errors": _unique(state["errors"] + [f"구조화 추출 실패 ({type(exc).__name__})"]), "reference_violations": []}

    def trl_gate_assess(state: TechnicalLocalState) -> dict:
        observations = state["extraction"].get("readiness_observations", [])
        records = [determine_trl(technology, observations).to_dict() for technology in ("sw", "hw")]
        return {"trl_records": records, "gate_traces": {item["technology"]: item["gate_trace"] for item in records}}

    def validate_and_finalize(state: TechnicalLocalState) -> dict:
        metrics, metric_warnings = validate_metric_records(state["extraction"].get("metrics", []))
        violations = list(state["reference_violations"])
        for record in state["trl_records"]:
            violations.extend(validate_trl_record(record, state["evidence_store"]))
        violations = _unique(violations)
        can_revise = bool(violations) and state["revision_round"] < MAX_REVISION_ROUNDS and not state["errors"]
        findings = _make_findings(state, state["trl_records"], metrics, metric_warnings, violations)
        # 추출 자체가 실패하면 검사할 주장이 없어 violations 도 비는데, 그걸 passed 로
        # 찍으면 "빈 결과"와 "문제 없음"이 구분되지 않는다. 실행 오류를 먼저 본다.
        quality = {
            "status": (
                "failed" if state["errors"] or violations
                else "needs_review" if metric_warnings
                else "passed"
            ),
            "violations": _unique(violations + list(state["errors"])), "warnings": metric_warnings,
            "checked_claim_ids": [item["claim_id"] for item in findings.get("claims", [])],
        }
        return {"needs_revision": can_revise, "quality_report": quality, "technical_findings": findings}

    def route_validation(state: TechnicalLocalState) -> str:
        return "revise" if state["needs_revision"] else "finish"

    def mark_revision(state: TechnicalLocalState) -> dict:
        return {"revision_round": state["revision_round"] + 1}

    graph = StateGraph(TechnicalLocalState)
    graph.add_node("collect_and_check", collect_and_check)
    graph.add_node("extract_normalize_and_bind", extract_normalize_and_bind)
    graph.add_node("trl_gate_assess", trl_gate_assess)
    graph.add_node("validate_and_finalize", validate_and_finalize)
    graph.add_node("mark_revision", mark_revision)
    graph.add_edge(START, "collect_and_check")
    graph.add_conditional_edges("collect_and_check", route_search, {"retry": "collect_and_check", "extract": "extract_normalize_and_bind"})
    graph.add_edge("extract_normalize_and_bind", "trl_gate_assess")
    graph.add_edge("trl_gate_assess", "validate_and_finalize")
    graph.add_conditional_edges("validate_and_finalize", route_validation, {"revise": "mark_revision", "finish": END})
    graph.add_edge("mark_revision", "extract_normalize_and_bind")
    return graph.compile()


def run_technical(request: dict, selected_tech: dict, corpus_manifest: list[dict], deps: TechnicalAgentDeps) -> dict:
    initial: TechnicalLocalState = {
        "request": request, "selected_tech": selected_tech, "corpus_manifest": corpus_manifest,
        "candidate_evidence": [], "coverage": {}, "missing_criteria": [], "search_round": 0,
        "needs_search_retry": False, "search_logs": [], "gaps": [], "errors": [],
        "extraction": {"records": [], "claims": [], "metrics": [], "readiness_observations": [], "gaps": []},
        "evidence_store": {}, "reference_violations": [], "trl_records": [], "gate_traces": {},
        "revision_round": 0, "needs_revision": False, "quality_report": {},
        "technical_findings": {}, "run_meta": {"started_on": date.today().isoformat()},
    }
    return build_technical_graph(deps).invoke(initial, {"recursion_limit": 30})

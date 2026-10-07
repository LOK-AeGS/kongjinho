"""코드 검사와 선택적 LLM judge를 결합한다."""

from __future__ import annotations

from pathlib import Path

from agents.quality.checks import run_checks


PERSPECTIVES = ("technical", "market", "stakeholder", "domain")


def _report_markdown(state: dict) -> str:
    path = (state.get("artifacts") or {}).get("report_with_ids_md")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return (state.get("report_sections") or {}).get("final_markdown_with_ids", "")


def _evidence_owners(evidence_id: str, findings: dict[str, dict | None]) -> set[str]:
    prefix = evidence_id.split(":", 1)[0]
    if prefix in {"technical", "market", "domain"}:
        return {prefix}
    owners = set()
    for perspective, result in findings.items():
        for item in (result or {}).get("records", []) + (result or {}).get("claims", []):
            if evidence_id in (item.get("evidence_ids") or []):
                owners.add(perspective)
    return owners


def _entailment_rework_targets(llm_criteria: dict, findings: dict[str, dict | None]) -> list[str]:
    targets = []
    unsupported = (llm_criteria.get("groundedness") or {}).get("unsupported_items") or []
    for item in unsupported:
        evidence_ids = list(item.get("evidence_ids") or [])
        owners = [_evidence_owners(evidence_id, findings) for evidence_id in evidence_ids]
        if evidence_ids and all(len(value) == 1 for value in owners):
            perspective = next(iter(owners[0]))
            if all(next(iter(value)) == perspective for value in owners) and perspective not in targets:
                targets.append(perspective)
    return targets


def quality_agent(state: dict, *, judge=None) -> dict:
    markdown = _report_markdown(state)
    findings = {name: state.get(f"{name}_findings") for name in PERSPECTIVES}
    report_quality = ((state.get("quality_by_perspective") or {}).get("report") or {})
    report_violations = list(report_quality.get("violations") or [])
    code = run_checks(
        markdown,
        state.get("evidence_store") or {},
        findings,
        report_violations,
    )
    statuses = {name: (value or {}).get("status", "missing") for name, value in findings.items()}
    judged = judge(markdown, statuses, state.get("evidence_store") or {}) if judge is not None else None
    llm_criteria = (judged or {}).get("criteria") or {}

    criteria = {}
    failed = []
    feedback = []
    warnings = []
    for name, result in code.items():
        # v1 실험에서 whole-report judge가 bias_control·coverage 결함을 놓쳐 두 기준은 code-only로 둔다.
        llm = llm_criteria.get(name)
        unsupported = name == "groundedness" and int((llm or {}).get("unsupported_count", 0)) > 0
        passed = bool(
            result["passed"]
            and (llm is None or int(llm.get("score", 0)) >= 3)
            and not unsupported
        )
        reasons = list(result.get("details") or [])
        if llm and llm.get("reasons"):
            reasons.append(str(llm["reasons"]))
        criteria[name] = {
            "passed": passed,
            "score": result["score"],
            "code": result,
            "llm": llm,
            "reasons": reasons,
        }
        warnings.extend(item for item in result.get("details") or [] if item.startswith("경고:"))
        if not passed:
            failed.append(name)
            for detail in result.get("details") or []:
                feedback.append(f"{name}: {detail}")
            for sentence in (llm or {}).get("problem_sentences", []):
                feedback.append(f"{name}: {sentence}")
    feedback = list(dict.fromkeys(feedback))[:8]
    rework_targets = list(code["coverage"].get("missing_perspectives") or [])
    rework_targets.extend(
        target for target in _entailment_rework_targets(llm_criteria, findings)
        if target not in rework_targets
    )
    verdict = {
        "passed": not failed,
        "mode": "hybrid" if judge is not None else "code_only",
        "criteria": criteria,
        "failed_criteria": failed,
        "feedback": feedback,
        "rework_targets": rework_targets,
        "judge_model": (judged or {}).get("model") or getattr(judge, "model", None),
        "evaluated_report_version": int(state.get("report_version", 0)),
    }
    return {
        "eval_result": verdict,
        "quality_by_perspective": {"quality_eval": {
            "status": "passed" if verdict["passed"] else "needs_review",
            "violations": feedback,
            "warnings": warnings,
            "checked_claim_ids": [],
        }},
    }


def make_node(judge=None):
    def node(state: dict) -> dict:
        return quality_agent(state, judge=judge)
    return node

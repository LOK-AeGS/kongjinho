"""State만 보고 다음 작업을 정하는 supervisor 정책."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from graph.decision_log import DecisionLogger


PERSPECTIVES = ("technical", "market", "stakeholder", "domain")
MAX_REWORK_PER_AGENT = 1
MAX_REPORT_VERSIONS = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SupervisorPolicy:
    max_rework_per_agent: int = MAX_REWORK_PER_AGENT
    max_report_versions: int = MAX_REPORT_VERSIONS
    min_evidence: dict[str, int] = field(default_factory=lambda: {
        "technical": 3,
        "market": 3,
        "stakeholder": 3,
        "domain": 3,
    })
    min_coverage: float = 0.5
    rework_search_rounds: int = 2


def assess_sufficiency(
    perspective: str,
    findings: dict | None,
    evidence_store: dict,
    policy: SupervisorPolicy,
) -> tuple[str, str, list[str]]:
    """관점 결과의 완료 상태·실재 근거 수·record coverage를 결정적으로 평가한다."""
    if not findings:
        return "insufficient", "findings 없음", []
    status = findings.get("status")
    focus = []
    for gap in findings.get("gaps") or []:
        label = "/".join(str(gap.get(key) or "") for key in ("technology", "criterion")).strip("/")
        if label and label not in focus:
            focus.append(label)
    focus = focus[:5]
    if status == "failed":
        return "insufficient", "status=failed", focus
    if status == "complete":
        return "sufficient", "status=complete", focus

    records = findings.get("records") or []
    cited = {
        evidence_id
        for item in records + (findings.get("claims") or [])
        for evidence_id in item.get("evidence_ids", [])
        if evidence_id in evidence_store
    }
    covered = sum(record.get("basis") in {"direct", "inferred"} for record in records)
    coverage = covered / len(records) if records else 0.0
    required = policy.min_evidence.get(perspective, 3)
    enough = len(cited) >= required and coverage >= policy.min_coverage
    comparisons = []
    if len(cited) < required:
        comparisons.append(f"근거 {len(cited)}건(<{required})")
    else:
        comparisons.append(f"근거 {len(cited)}건")
    if coverage < policy.min_coverage:
        comparisons.append(f"coverage {coverage:.2f}(<{policy.min_coverage:g})")
    else:
        comparisons.append(f"coverage {coverage:.2f}")
    return ("sufficient" if enough else "insufficient"), f"partial, {', '.join(comparisons)}", focus


def _attempts(state: dict, name: str) -> int:
    return int(((state.get("node_status") or {}).get(name) or {}).get("attempts", 0))


def _completed_step(state: dict, name: str) -> int:
    value = ((state.get("node_status") or {}).get(name) or {}).get("completed_step")
    return int(value) if value is not None else -1


def _can_rework(state: dict, name: str, policy: SupervisorPolicy) -> bool:
    return max(0, _attempts(state, name) - 1) < policy.max_rework_per_agent


def _directive(reason: str, focus: list[str], state: dict, name: str, policy: SupervisorPolicy, feedback=None) -> dict:
    return {
        "reason": reason,
        "focus": focus[:5],
        "round": max(1, _attempts(state, name)),
        "max_search_rounds": policy.rework_search_rounds,
        "feedback": list(feedback or [])[:8],
        "created_step": int(state.get("step_count", 0)) + 1,
    }


def decide(state: dict, policy: SupervisorPolicy | None = None) -> dict:
    """우선순위 규칙의 첫 일치 항목과 그에 필요한 State 갱신을 반환한다."""
    policy = policy or SupervisorPolicy()
    old_step = int(state.get("step_count", 0))
    step = old_step + 1
    now = utc_now()
    statuses = dict(state.get("node_status") or {})
    status_updates: dict[str, dict] = {}
    rework = dict(state.get("rework") or {})
    accepted: list[str] = []

    def finish(next_nodes: list[str], reason: str) -> dict:
        for name in next_nodes:
            directive = rework.get(name)
            if directive and int(directive.get("created_step", -1)) <= _completed_step(state, name):
                rework.pop(name, None)
            previous = dict(statuses.get(name) or {})
            status_updates[name] = {
                "status": "running",
                "attempts": int(previous.get("attempts", 0)),
                "completed_step": previous.get("completed_step"),
                "last_error": previous.get("last_error"),
                "sufficiency": previous.get("sufficiency"),
                "updated_at": now,
            }
        decision = {"step": step, "next": next_nodes, "reason": reason, "ts": now}
        return {
            "step_count": step,
            "next": next_nodes,
            "rework": rework,
            "last_decision": decision,
            "node_status": status_updates,
        }

    if old_step >= int(state.get("max_steps", 20)):
        return finish([], "max_steps 도달로 종료")

    if _attempts(state, "technical") == 0:
        return finish(["technical"], "다른 관점의 공통 입력인 technical 요약이 필요함")

    verdict, reason, focus = assess_sufficiency(
        "technical", state.get("technical_findings"), state.get("evidence_store") or {}, policy
    )
    technical_status = dict(statuses.get("technical") or {})
    technical_status["sufficiency"] = verdict
    status_updates["technical"] = technical_status
    if verdict == "insufficient":
        if _can_rework(state, "technical", policy):
            rework["technical"] = _directive(reason, focus, state, "technical", policy)
            return finish(["technical"], f"technical 근거 부족으로 재작업: {reason}")
        technical_status["sufficiency"] = "accepted_insufficient"
        accepted.append(f"technical({reason})")

    dispatch: list[str] = []
    for name in ("market", "stakeholder", "domain"):
        if _attempts(state, name) == 0:
            dispatch.append(name)
            continue
        verdict, reason, focus = assess_sufficiency(
            name, state.get(f"{name}_findings"), state.get("evidence_store") or {}, policy
        )
        perspective_status = dict(statuses.get(name) or {})
        perspective_status["sufficiency"] = verdict
        status_updates[name] = perspective_status
        if verdict == "insufficient":
            if _can_rework(state, name, policy):
                rework[name] = _directive(reason, focus, state, name, policy)
                dispatch.append(name)
            else:
                perspective_status["sufficiency"] = "accepted_insufficient"
                accepted.append(f"{name}({reason})")
    if dispatch:
        first = [name for name in dispatch if _attempts(state, name) == 0]
        retry = [name for name in dispatch if name not in first]
        parts = []
        if first:
            parts.append("미실행 관점 병렬 조사: " + ", ".join(first))
        if retry:
            parts.append("근거 부족 관점 재작업: " + ", ".join(retry))
        return finish(dispatch, "; ".join(parts))

    suffix = f"; 재작업 상한으로 부족 상태 수용: {', '.join(accepted)}" if accepted else ""
    synthesis_missing = not state.get("synthesis") or _attempts(state, "synthesis") == 0
    latest_perspective = max(_completed_step(state, name) for name in PERSPECTIVES)
    if synthesis_missing or latest_perspective > _completed_step(state, "synthesis"):
        why = "synthesis 미실행" if synthesis_missing else "관점 결과가 갱신되어 synthesis가 stale"
        return finish(["synthesis"], why + suffix)

    report_missing = _attempts(state, "report") == 0
    if report_missing or _completed_step(state, "synthesis") > _completed_step(state, "report"):
        why = "report 미실행" if report_missing else "synthesis가 갱신되어 report가 stale"
        if not report_missing and int(state.get("report_version", 0)) >= policy.max_report_versions:
            return finish([], "품질 루프 상한 도달, needs_review로 종료")
        return finish(["report"], why)

    verdict = state.get("eval_result")
    if not verdict or int(verdict.get("evaluated_report_version", -1)) < int(state.get("report_version", 0)):
        return finish(["quality_eval"], "현재 report_version의 품질 평가가 필요함")
    if verdict.get("passed"):
        return finish([], "품질 평가 통과")
    if int(state.get("report_version", 0)) >= policy.max_report_versions:
        return finish([], "품질 루프 상한 도달, needs_review로 종료")

    feedback = verdict.get("feedback") or []
    rework["report"] = _directive(
        "품질 평가 불합격", feedback, state, "report", policy, feedback
    )
    targets = [name for name in verdict.get("rework_targets") or [] if name in PERSPECTIVES and _can_rework(state, name, policy)]
    if targets:
        for name in targets:
            rework[name] = _directive("품질 평가에서 관점 근거 재작업 필요", feedback, state, name, policy, feedback)
        return finish(targets, "품질 평가 지적으로 관점 재작업: " + ", ".join(targets))
    if int(state.get("report_version", 0)) < policy.max_report_versions:
        return finish(["report"], "품질 평가 지적을 반영해 report 재작성")
    return finish([], "품질 루프 상한 도달, needs_review로 종료")


def make_supervisor(
    *,
    policy: SupervisorPolicy | None = None,
    logger: DecisionLogger | None = None,
    on_decision: Callable[[dict], None] | None = None,
):
    policy = policy or SupervisorPolicy()
    logger = logger or DecisionLogger()

    def supervisor(state: dict) -> dict:
        update = decide(state, policy)
        decision = update["last_decision"]
        logger.log(state.get("trace_id", ""), decision["step"], decision["next"], decision["reason"], decision["ts"])
        if on_decision:
            on_decision(decision)
        return update

    return supervisor


def route(state: dict) -> list[str] | str:
    """Supervisor가 계산한 next만 적용한다. 빈 목록은 END로 보낸다."""
    return list(state.get("next") or []) or "__end__"

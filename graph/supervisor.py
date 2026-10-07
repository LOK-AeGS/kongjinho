"""State가 허용 결정을 계산하고, 선택적 LLM 제안을 guard하는 supervisor 정책."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Callable

from graph.decision_log import DecisionLogger


PERSPECTIVES = ("technical", "market", "stakeholder", "domain")
MAX_REWORK_PER_AGENT = 1
MAX_REPORT_VERSIONS = 2
DEFAULT_SUPERVISOR_MODEL = "gpt-4.1"


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
    comparisons = [
        f"근거 {len(cited)}건" + (f"(<{required})" if len(cited) < required else ""),
        f"coverage {coverage:.2f}" + (f"(<{policy.min_coverage:g})" if coverage < policy.min_coverage else ""),
    ]
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


def _choice(action: str, targets: list[str], reason: str) -> dict:
    return {"action": action, "targets": list(targets), "reason": reason[:300]}


def _subsets(values: list[str]) -> list[list[str]]:
    return [list(group) for size in range(1, len(values) + 1) for group in combinations(values, size)]


def _build_plan(state: dict, policy: SupervisorPolicy) -> dict:
    """의존성·예산·상한을 적용해 실제로 실행 가능한 선택지만 만든다."""
    assessments: dict[str, dict] = {}
    accepted: list[str] = []

    def inspect(name: str) -> dict:
        verdict, reason, focus = assess_sufficiency(
            name, state.get(f"{name}_findings"), state.get("evidence_store") or {}, policy
        )
        assessments[name] = {
            "verdict": verdict,
            "reason": reason,
            "focus": focus,
            "attempts": _attempts(state, name),
            "budget_left": _can_rework(state, name, policy),
        }
        return assessments[name]

    if int(state.get("step_count", 0)) >= max(0, int(state.get("max_steps", 20)) - 1):
        default = _choice("finish", [], "max_steps 도달로 종료")
        return {"allowed": [default], "default": default, "assessments": assessments, "accepted": accepted}
    if _attempts(state, "technical") == 0:
        default = _choice("dispatch", ["technical"], "다른 관점의 공통 입력인 technical 요약이 필요함")
        return {"allowed": [default], "default": default, "assessments": assessments, "accepted": accepted}

    technical = inspect("technical")
    technical_accepted = (
        ((state.get("node_status") or {}).get("technical") or {}).get("sufficiency")
        == "accepted_insufficient"
    )
    if technical["verdict"] == "insufficient" and technical["budget_left"] and not technical_accepted:
        retry = _choice("rework", ["technical"], f"technical 근거 부족으로 재작업: {technical['reason']}")
        accept = _choice("accept_insufficient", ["technical"], f"technical 부족 상태를 수용: {technical['reason']}")
        return {"allowed": [retry, accept], "default": retry, "assessments": assessments, "accepted": accepted}
    if technical["verdict"] == "insufficient":
        accepted.append(f"technical({technical['reason']})")

    pending = [name for name in PERSPECTIVES[1:] if _attempts(state, name) == 0]
    if pending:
        default = _choice("dispatch", pending, "미실행 관점 병렬 조사: " + ", ".join(pending))
        return {"allowed": [default], "default": default, "assessments": assessments, "accepted": accepted}

    candidates = []
    for name in PERSPECTIVES[1:]:
        result = inspect(name)
        if result["verdict"] != "insufficient":
            continue
        already_accepted = (
            ((state.get("node_status") or {}).get(name) or {}).get("sufficiency")
            == "accepted_insufficient"
        )
        if result["budget_left"] and not already_accepted:
            candidates.append(name)
        else:
            accepted.append(f"{name}({result['reason']})")
    if candidates:
        allowed = [
            _choice("rework", subset, "근거 부족 관점 재작업: " + ", ".join(subset))
            for subset in _subsets(candidates)
        ]
        allowed.append(_choice(
            "accept_insufficient",
            candidates,
            "재작업 대신 부족 상태를 수용: " + ", ".join(candidates),
        ))
        default = next(item for item in allowed if item["action"] == "rework" and item["targets"] == candidates)
        return {"allowed": allowed, "default": default, "assessments": assessments, "accepted": accepted}

    suffix = f"; 재작업 상한으로 부족 상태 수용: {', '.join(accepted)}" if accepted else ""
    synthesis_missing = not state.get("synthesis") or _attempts(state, "synthesis") == 0
    latest_perspective = max(_completed_step(state, name) for name in PERSPECTIVES)
    if synthesis_missing or latest_perspective > _completed_step(state, "synthesis"):
        why = "synthesis 미실행" if synthesis_missing else "관점 결과가 갱신되어 synthesis가 stale"
        default = _choice("synthesis", [], why + suffix)
        return {"allowed": [default], "default": default, "assessments": assessments, "accepted": accepted}

    report_missing = _attempts(state, "report") == 0
    if report_missing or _completed_step(state, "synthesis") > _completed_step(state, "report"):
        if not report_missing and int(state.get("report_version", 0)) >= policy.max_report_versions:
            default = _choice("finish", [], "품질 루프 상한 도달, needs_review로 종료")
        else:
            why = "report 미실행" if report_missing else "synthesis가 갱신되어 report가 stale"
            default = _choice("report", [], why)
        return {"allowed": [default], "default": default, "assessments": assessments, "accepted": accepted}

    verdict = state.get("eval_result")
    if not verdict or int(verdict.get("evaluated_report_version", -1)) < int(state.get("report_version", 0)):
        default = _choice("quality_eval", [], "현재 report_version의 품질 평가가 필요함")
        return {"allowed": [default], "default": default, "assessments": assessments, "accepted": accepted}
    if verdict.get("passed"):
        default = _choice("finish", [], "품질 평가 통과")
        return {"allowed": [default], "default": default, "assessments": assessments, "accepted": accepted}
    if int(state.get("report_version", 0)) >= policy.max_report_versions:
        default = _choice("finish", [], "품질 루프 상한 도달, needs_review로 종료")
        return {"allowed": [default], "default": default, "assessments": assessments, "accepted": accepted}

    targets = [
        name for name in verdict.get("rework_targets") or []
        if name in PERSPECTIVES and _can_rework(state, name, policy)
    ]
    allowed = [
        _choice("quality_rework", subset, "품질 평가 지적으로 관점 재작업: " + ", ".join(subset))
        for subset in _subsets(targets)
    ]
    rewrite = _choice("rewrite_report", ["report"], "품질 평가 지적을 반영해 report 재작성")
    finish = _choice("finish", [], "품질 미달 상태로 종료(needs_review)")
    allowed.extend([rewrite, finish])
    default = allowed[len(allowed) - 2] if not targets else next(
        item for item in allowed if item["action"] == "quality_rework" and item["targets"] == targets
    )
    return {"allowed": allowed, "default": default, "assessments": assessments, "accepted": accepted}


def options(state: dict, policy: SupervisorPolicy | None = None) -> list[dict]:
    """현재 State에서 guard가 허용할 정확한 결정 목록."""
    return _build_plan(state, policy or SupervisorPolicy())["allowed"]


def rule_choice(state: dict, allowed: list[dict] | None = None, policy: SupervisorPolicy | None = None) -> dict:
    """같은 State에서 규칙이 택하는 기본 결정."""
    plan = _build_plan(state, policy or SupervisorPolicy())
    return plan["default"]


def _target_key(targets: list[str]) -> tuple[str, ...]:
    order = {name: index for index, name in enumerate((*PERSPECTIVES, "report"))}
    return tuple(sorted(targets, key=lambda name: order.get(name, len(order))))


def guard(proposal: dict | None, allowed: list[dict], rule: dict) -> tuple[dict, str]:
    """action과 targets가 허용 목록의 한 항목과 정확히 일치할 때만 LLM 제안을 채택한다."""
    if proposal is None:
        return rule, "rule"
    action = str(proposal.get("action") or "")
    targets = [str(value) for value in proposal.get("targets") or []]
    match = next((
        item for item in allowed
        if item["action"] == action
        and len(targets) == len(set(targets))
        and _target_key(item["targets"]) == _target_key(targets)
    ), None)
    if match is None:
        allowed_text = ", ".join(f"{item['action']}:{item['targets']}" for item in allowed)
        reason = f"fallback: 허용 밖 제안 {action}:{targets}; allowed={allowed_text}; {rule['reason']}"
        return {**rule, "reason": reason[:300]}, "fallback"
    return {
        "action": action,
        "targets": list(match["targets"]),
        "reason": str(proposal.get("reason") or match["reason"])[:300],
    }, "llm"


def _allowed_meaning(decision: dict) -> str:
    meanings = {
        "dispatch": "아직 실행하지 않은 관점의 최초 조사를 시작한다.",
        "rework": "이미 실행했지만 근거가 부족한 관점을 예산 안에서 재조사한다.",
        "accept_insufficient": "근거 부족을 명시적으로 수용하고 다음 단계로 진행한다.",
        "synthesis": "판정이 끝난 네 관점을 종합하거나 최신 관점 결과로 종합을 갱신한다.",
        "report": "종합 결과를 바탕으로 최초 보고서 또는 stale 보고서를 작성한다.",
        "quality_eval": "현재 report_version을 품질 평가한다.",
        "quality_rework": "품질 평가가 지목한 upstream 관점을 재조사한 뒤 보고서를 다시 작성한다.",
        "rewrite_report": "upstream 재조사 없이 품질 피드백만 반영해 보고서를 다시 쓴다.",
        "finish": "현재 품질 또는 안전 상한 상태로 실행을 종료한다.",
    }
    return meanings[decision["action"]]


def state_summary(
    state: dict,
    plan: dict | None = None,
    policy: SupervisorPolicy | None = None,
) -> dict:
    """LLM이 미실행과 근거 부족을 혼동하지 않도록 실행 상태를 먼저 요약한다."""
    policy = policy or SupervisorPolicy()
    plan = plan or _build_plan(state, policy)
    perspectives = {}
    for name in PERSPECTIVES:
        attempts = _attempts(state, name)
        node = ((state.get("node_status") or {}).get(name) or {})
        rework_used = max(0, attempts - 1)
        entry = {
            "state": "pending",
            "attempts": attempts,
            "rework_left": max(0, policy.max_rework_per_agent - rework_used),
        }
        if node.get("status") == "running":
            entry["state"] = "running"
        elif attempts > 0:
            verdict, reason, _ = assess_sufficiency(
                name,
                state.get(f"{name}_findings"),
                state.get("evidence_store") or {},
                policy,
            )
            if node.get("sufficiency") == "accepted_insufficient":
                entry["state"] = "accepted_insufficient"
            else:
                entry["state"] = verdict
            entry["sufficiency_reason"] = reason
        perspectives[name] = entry
    allowed = [
        {
            "action": item["action"],
            "targets": item["targets"],
            "meaning": _allowed_meaning(item),
        }
        for item in plan["allowed"]
    ]
    return {
        "step": int(state.get("step_count", 0)),
        "max_steps": int(state.get("max_steps", 20)),
        "perspectives": perspectives,
        "rework_budget_per_agent": policy.max_rework_per_agent,
        "report_version": int(state.get("report_version", 0)),
        "max_report_versions": policy.max_report_versions,
        "eval": {
            "failed_criteria": list((state.get("eval_result") or {}).get("failed_criteria") or []),
            "rework_targets": list((state.get("eval_result") or {}).get("rework_targets") or []),
        },
        "rule_default": {
            "action": plan["default"]["action"],
            "targets": plan["default"]["targets"],
            "reason": plan["default"]["reason"],
        },
        "allowed": allowed,
    }


def make_llm_proposer(model: str | None = None):
    """OpenAI structured output으로 허용 집합 안의 결정을 제안한다."""
    from typing import Literal

    from openai import OpenAI
    from pydantic import BaseModel, ConfigDict

    chosen_model = model or os.getenv("SUPERVISOR_MODEL") or DEFAULT_SUPERVISOR_MODEL

    class LLMDecision(BaseModel):
        model_config = ConfigDict(extra="forbid")
        action: Literal[
            "dispatch", "rework", "accept_insufficient", "synthesis", "report",
            "quality_eval", "quality_rework", "rewrite_report", "finish",
        ]
        targets: list[str]
        reason: str

    client = OpenAI(timeout=60, max_retries=1)
    instructions = (
        "너는 KV cache 다관점 평가 파이프라인의 Supervisor다. allowed에 정확히 포함된 action과 targets 조합만 고른다. "
        "재작업 예산이 남아 있고 근거가 부족하면 재작업을 우선한다. 근거가 충분하다고 판정되거나 부족 상태가 명시적으로 "
        "수용되기 전에는 보고서를 작성하지 않는다. reason은 summary에 실제로 있는 state·attempts·rework_left·근거 수·"
        "failed_criteria만 인용해 한국어 1~2문장으로 쓴다. sufficient 관점을 근거 부족이라고 말하지 않는다. "
        "pending 관점의 첫 dispatch는 재작업이 아니라 최초 조사라고 표현한다. summary에 없는 사실을 추정하지 않는다."
    )

    def propose(summary: dict, allowed: list[dict]) -> dict:
        response = client.responses.parse(
            model=chosen_model,
            instructions=instructions,
            input=json.dumps(summary, ensure_ascii=False, default=str),
            text_format=LLMDecision,
            max_output_tokens=800,
            store=False,
        )
        if response.status != "completed" or response.output_parsed is None:
            raise ValueError("supervisor_decision_incomplete")
        return response.output_parsed.model_dump()

    propose.model = chosen_model
    return propose


def decide(
    state: dict,
    policy: SupervisorPolicy | None = None,
    proposer: Callable[[dict, list[dict]], dict] | None = None,
) -> dict:
    """규칙이 허용 집합을 만들고 proposer 선택을 guard한 뒤 State 갱신을 반환한다."""
    policy = policy or SupervisorPolicy()
    plan = _build_plan(state, policy)
    rule = plan["default"]
    if proposer is None:
        decision, source = guard(None, plan["allowed"], rule)
    else:
        try:
            proposal = proposer(state_summary(state, plan, policy), plan["allowed"])
            decision, source = guard(proposal, plan["allowed"], rule)
        except Exception as exc:
            decision = {
                **rule,
                "reason": f"fallback: proposer 실패({type(exc).__name__}); {rule['reason']}"[:300],
            }
            source = "fallback"

    old_step = int(state.get("step_count", 0))
    step = old_step + 1
    now = utc_now()
    statuses = dict(state.get("node_status") or {})
    status_updates: dict[str, dict] = {}
    rework = dict(state.get("rework") or {})

    for name, assessment in plan["assessments"].items():
        previous = dict(statuses.get(name) or {})
        previous["sufficiency"] = assessment["verdict"]
        status_updates[name] = previous
    for entry in plan["accepted"]:
        name = entry.split("(", 1)[0]
        status_updates.setdefault(name, dict(statuses.get(name) or {}))["sufficiency"] = "accepted_insufficient"

    action = decision["action"]
    targets = decision["targets"]
    next_nodes: list[str]
    if action in {"dispatch", "rework"}:
        next_nodes = targets
        if action == "rework":
            for name in targets:
                assessment = plan["assessments"][name]
                rework[name] = _directive(assessment["reason"], assessment["focus"], state, name, policy)
    elif action == "accept_insufficient":
        for name in targets:
            status_updates.setdefault(name, dict(statuses.get(name) or {}))["sufficiency"] = "accepted_insufficient"
        pending = [name for name in PERSPECTIVES[1:] if _attempts(state, name) == 0]
        next_nodes = pending or ["synthesis"]
    elif action in {"synthesis", "report", "quality_eval"}:
        next_nodes = [action]
    elif action == "rewrite_report":
        feedback = (state.get("eval_result") or {}).get("feedback") or []
        rework["report"] = _directive("품질 평가 불합격", feedback, state, "report", policy, feedback)
        next_nodes = ["report"]
    elif action == "quality_rework":
        verdict = state.get("eval_result") or {}
        feedback = verdict.get("feedback") or []
        rework["report"] = _directive("품질 평가 불합격", feedback, state, "report", policy, feedback)
        for name in targets:
            rework[name] = _directive(
                "품질 평가에서 관점 근거 재작업 필요", feedback, state, name, policy, feedback
            )
        next_nodes = targets
    else:
        next_nodes = []

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
            "sufficiency": status_updates.get(name, previous).get("sufficiency"),
            "updated_at": now,
        }
    recorded = {
        "step": step,
        "next": next_nodes,
        "reason": decision["reason"],
        "source": source,
        "ts": now,
    }
    return {
        "step_count": step,
        "next": next_nodes,
        "rework": rework,
        "last_decision": recorded,
        "node_status": status_updates,
    }


# allowed-set/guard 패턴은 teammate의 feat/supervisor-pattern(commit 1d1d82a)에서 포팅했다.
def make_supervisor(
    *,
    policy: SupervisorPolicy | None = None,
    proposer: Callable[[dict, list[dict]], dict] | None = None,
    logger: DecisionLogger | None = None,
    on_decision: Callable[[dict], None] | None = None,
):
    """허용 결정 계산과 선택적 proposer를 하나의 supervisor 노드로 묶는다."""
    policy = policy or SupervisorPolicy()
    logger = logger or DecisionLogger()

    def supervisor(state: dict) -> dict:
        update = decide(state, policy, proposer)
        decision = update["last_decision"]
        logger.log(
            state.get("trace_id", ""), decision["step"], decision["next"],
            decision["reason"], decision["source"], decision["ts"],
        )
        if on_decision:
            on_decision(decision)
        return update

    return supervisor


def route(state: dict) -> list[str] | str:
    """Supervisor가 계산한 next만 적용한다. 빈 목록은 END로 보낸다."""
    return list(state.get("next") or []) or "__end__"

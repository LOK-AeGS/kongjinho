"""Supervisor 노드와 라우터.

설계 원칙
  - 워커(technical·market·stakeholder·domain·synthesis·report·quality)는 끝나면 반드시 supervisor 로만 돌아온다.
    워커끼리 직접 통신하지 않는다 (graph/build.py 의 edge 가 이를 강제).
  - 다음 행동은 현재 State(관점별 결과·근거 충분성·재작업 예산·품질 평가 결과)에서 계산한 "허용 행동 집합"(options) 안에서
    LLM 이 고른다. 순서는 코드에 박혀 있지 않고 State 가 바뀌면 달라진다.
  - guard: LLM 제안이 허용 집합 밖이거나 API 오류·파싱 실패면 규칙 선택으로 대체하고 source=fallback 으로 기록한다.
    따라서 LLM 이 없어도(rule 모드) 그래프는 같은 규칙으로 끝까지 돌고 종료가 보장된다.
  - 종료 보장: step_count >= max_steps 면 새 재작업 없이 synthesis → report 만 마치고 끝낸다 (final_status=degraded).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from langgraph.graph import END
from langgraph.types import Send

from graph.sufficiency import (
    GLOBAL_REWORK, MAX_QUALITY_LOOPS, PERSPECTIVES, assess, attempts, budget_left, exhausted_perspectives,
    is_stale, make_rework_request, rework_candidates,
)

STAGE_MAX_ATTEMPTS = 4
ACTIONS = ("dispatch", "synthesis", "report", "quality", "quality_rework", "finish")
DEFAULT_SUPERVISOR_MODEL = "gpt-6.1-sol"


# ---------------------------------------------------------------- 단계 상태
def stage_done(state: dict, stage: str) -> bool:
    status = (state.get("node_status") or {}).get(stage) or {}
    if stage == "synthesis":
        return state.get("synthesis") is not None and status.get("status") != "stale"
    if stage == "report":
        return status.get("status") in ("ok", "partial") and bool((state.get("report_sections") or {}))
    return state.get("quality_verdict") is not None


def stage_runnable(state: dict, stage: str) -> bool:
    status = (state.get("node_status") or {}).get(stage) or {}
    return status.get("status") != "failed" or int(status.get("attempts", 0)) < STAGE_MAX_ATTEMPTS


def has_cap(state: dict) -> bool:
    return int(state.get("step_count") or 0) >= int(state.get("max_steps") or 14)


# ---------------------------------------------------------------- 허용 행동
def options(state: dict) -> dict[str, list[str]]:
    """현재 State 에서 허용되는 행동 → 후보 대상. 항상 하나의 행동 종류만 열린다."""
    cap = has_cap(state)
    candidates = {} if cap else rework_candidates(state)
    if candidates:
        first = [p for p in candidates if p == "technical"]
        return {"dispatch": first or list(candidates)}
    for stage in ("synthesis", "report"):
        if not stage_done(state, stage):
            return {stage: []} if stage_runnable(state, stage) else {"finish": []}
    if cap:
        return {"finish": []}
    verdict = state.get("quality_verdict")
    if verdict is None:
        return {"quality": []}
    if verdict["passed"] or int(state.get("quality_iterations") or 0) >= MAX_QUALITY_LOOPS:
        return {"finish": []}
    targets = [t for t in verdict["target_perspectives"] if t == "report" or (t in PERSPECTIVES and budget_left(state, t))]
    return {"quality_rework": targets} if targets else {"finish": []}


def rule_choice(state: dict, opts: dict[str, list[str]]) -> dict:
    action, candidates = next(iter(opts.items()))
    why = {
        "dispatch": lambda: "; ".join(f"{p}: {info['reason']}" for p, info in rework_candidates(state).items() if p in candidates),
        "synthesis": lambda: "네 관점 결과가 충분하거나 재작업 예산을 모두 사용함 → 평가 종합",
        "report": lambda: "평가 종합 완료 → 보고서 작성",
        "quality": lambda: "보고서 완료 → 품질 평가",
        "quality_rework": lambda: "품질 평가 미달: " + ", ".join((state.get("quality_verdict") or {}).get("failed_checks", [])),
        "finish": lambda: "종료 조건 충족(품질 통과 / 예산 소진 / step 상한)",
    }[action]()
    return {"action": action, "targets": list(candidates), "reason": why[:300]}


def guard(proposal: dict | None, opts: dict[str, list[str]], rule: dict, sequential: bool) -> tuple[dict, str]:
    if proposal is None:
        decision, source = rule, "rule"
    else:
        action = proposal.get("action")
        if action not in opts:
            return {**rule, "reason": f"fallback: LLM 제안 {action!r} 은 허용 행동 {sorted(opts)} 밖 · {rule['reason']}"[:300]}, "fallback"
        candidates = opts[action]
        targets = [t for t in proposal.get("targets") or [] if t in candidates]
        if candidates and not targets:
            return {**rule, "reason": f"fallback: LLM 대상 {proposal.get('targets')} 이 후보 {candidates} 와 겹치지 않음 · {rule['reason']}"[:300]}, "fallback"
        decision = {"action": action, "targets": targets, "reason": str(proposal.get("reason") or rule["reason"])[:300]}
        source = "llm"
    if sequential and decision["action"] == "dispatch":
        decision = {**decision, "targets": decision["targets"][:1]}
    return decision, source


# ---------------------------------------------------------------- 상태 갱신
def merge_view(state: dict, updates: dict) -> dict:
    """reducer 가 있는 dict 키를 합쳐 '업데이트가 반영된 State' 를 로컬에서 미리 본다."""
    view = {**state, **{k: v for k, v in updates.items() if k not in ("rework_requests", "node_status", "decision_log")}}
    for key in ("rework_requests", "node_status"):
        view[key] = {**(state.get(key) or {}), **(updates.get(key) or {})}
    return view


def combine(first: dict, second: dict) -> dict:
    merged = {**first, **second}
    for key in ("rework_requests", "node_status"):
        if key in first and key in second:
            merged[key] = {**first[key], **second[key]}
    return merged


def invalidate_downstream(state: dict) -> dict:
    stale = {}
    for stage in ("synthesis", "report"):
        previous = (state.get("node_status") or {}).get(stage)
        if previous:
            stale[stage] = {"status": "stale", "error": None, "attempts": previous.get("attempts", 0)}
    return {"node_status": stale, "quality_verdict": None} if stale else {}


def apply_dispatch(state: dict, targets: list[str]) -> dict:
    infos = rework_candidates(state)
    requests, rework = {}, False
    for p in targets:
        if state.get(f"{p}_findings") is not None:  # 이미 한 번 돈 관점 → 재작업
            rework = True
            if not (state.get("rework_requests") or {}).get(p):
                requests[p] = make_rework_request(state, p, infos[p]["requested_by"] or "supervisor-sufficiency")
    updates = {"rework_requests": requests} if requests else {}
    if rework:
        updates.update(invalidate_downstream(state))
    return updates


def apply_quality_rework(state: dict, targets: list[str]) -> dict:
    verdict = state["quality_verdict"]
    feedback = [{"technology": "both", "perspective": "report", "criterion": check, "reason": "; ".join(verdict["details"].get(check, []))[:200],
                 "missing_evidence": []} for check in verdict["failed_checks"]]
    requests = {t: make_rework_request(state, t, "quality", feedback if t == "report" else None) for t in targets}
    return combine({"rework_requests": requests}, invalidate_downstream(state))


def degraded_reasons(state: dict) -> list[str]:
    reasons = []
    if has_cap(state):
        reasons.append(f"step 상한 도달(max_steps={state.get('max_steps')})")
    reasons += [f"{p}: 근거 부족인 채 재작업 예산 소진 ({', '.join(assess(state, p)['reasons'])})" for p in exhausted_perspectives(state)]
    verdict = state.get("quality_verdict")
    if verdict is None:
        reasons.append("품질 평가를 마치지 못함")
    elif not verdict["passed"]:
        reasons.append("품질 평가 미달인 채 루프 상한 도달: " + ", ".join(verdict["failed_checks"]))
    reasons += [f"{stage} 단계 실패" for stage in ("synthesis", "report") if not stage_done(state, stage)]
    return reasons


# ---------------------------------------------------------------- LLM 제안
def make_llm_proposer(client, model: str):
    from typing import Literal

    from pydantic import BaseModel

    class LLMDecision(BaseModel):
        action: Literal["dispatch", "synthesis", "report", "quality", "quality_rework", "finish"]
        targets: list[str]
        reason: str

    instructions = (
        "너는 KV cache 다관점 평가 파이프라인의 Supervisor 다. 아래 State 요약과 allowed(허용 행동 → 후보 대상) 안에서 "
        "다음 행동을 정한다. allowed 밖의 행동은 코드가 거부한다. dispatch 는 후보 중 이번 턴에 실행할 워커를 targets 로 고른다 "
        "(독립 워커는 함께 고를 수 있고 technical 이 후보면 먼저 한다). 근거가 부족한 관점은 반드시 재작업하고 "
        "근거가 충분해지기 전에는 보고서를 쓰지 않는다. reason 에 State 에서 본 근거를 한두 문장으로 적는다.")

    def propose(state: dict, opts: dict[str, list[str]]) -> dict:
        summary = {
            "step": state.get("step_count"), "max_steps": state.get("max_steps"),
            "perspectives": {p: {k: v for k, v in assess(state, p).items()} for p in PERSPECTIVES},
            "rework_used": sum(max(attempts(state, p) - 1, 0) for p in PERSPECTIVES), "rework_budget": GLOBAL_REWORK,
            "synthesis_done": stage_done(state, "synthesis"), "report_done": stage_done(state, "report"),
            "quality_verdict": state.get("quality_verdict"), "allowed": opts,
        }
        response = client.responses.parse(model=model, instructions=instructions, store=False, text_format=LLMDecision,
                                          input=json.dumps(summary, ensure_ascii=False, default=str), max_output_tokens=1500)
        if response.status != "completed" or response.output_parsed is None:
            raise ValueError("supervisor_decision_incomplete")
        return response.output_parsed.model_dump()

    return propose


# ---------------------------------------------------------------- 노드와 라우터
def make_supervisor(proposer=None, *, log_dir: str | Path | None = None, sequential: bool | None = None):
    sequential = bool(os.getenv("SUPERVISOR_SEQUENTIAL")) if sequential is None else sequential

    def record(state: dict, step: int, decision: dict, source: str) -> dict:
        entry = {"step": step, "action": decision["action"], "targets": decision["targets"], "reason": decision["reason"], "source": source}
        if log_dir:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f"{state.get('trace_id', 'trace')}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**entry, "trace_id": state.get("trace_id"), "run_id": state.get("run_id")}, ensure_ascii=False) + "\n")
        return entry

    def supervisor_node(state: dict) -> dict:
        step = int(state.get("step_count") or 0) + 1
        log, updates = [], {}
        view = state
        for _ in range(2):  # quality_rework 는 같은 턴에서 곧바로 다음 행동까지 정한다
            opts = options(view)
            rule = rule_choice(view, opts)
            proposal, note = None, ""
            if proposer is not None:
                try:
                    proposal = proposer(view, opts)
                except Exception as exc:
                    note = f"fallback: LLM 호출 실패({type(exc).__name__}) · "
            decision, source = guard(proposal, opts, rule, sequential)
            if note:
                decision, source = {**decision, "reason": (note + decision["reason"])[:300]}, "fallback"
            log.append(record(state, step, decision, source))
            if decision["action"] == "dispatch":
                updates = combine(updates, apply_dispatch(view, decision["targets"]))
                break
            if decision["action"] == "quality_rework":
                more = apply_quality_rework(view, decision["targets"])
                updates = combine(updates, more)
                view = merge_view(view, more)
                continue
            break
        errors = [f"{name}: {run['error']}" for name, run in (state.get("node_status") or {}).items() if run.get("error")]
        out = {**updates, "step_count": step, "next_action": decision["action"], "next_targets": decision["targets"],
               "decision_log": log, "last_error": errors[-1] if errors else None}
        if decision["action"] == "finish":
            reasons = degraded_reasons(view)
            out["final_status"] = "degraded" if reasons else "ok"
            out["run_meta"] = {"supervisor": {"degraded_reasons": reasons, "steps": step}}
        return out

    return supervisor_node


def route(state: dict):
    action = state.get("next_action")
    if action == "dispatch":
        view = state
        return [Send(target, view) for target in state.get("next_targets") or []]
    if action in ("synthesis", "report", "quality"):
        return action
    return END

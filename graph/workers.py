"""워커 래퍼. 팀원의 make_node() 결과를 수정하지 않고 Supervisor 계약에 맞춘다.

워커가 하는 일
  1) 재작업 요청이 있으면 rework_hint 를 State 에 실어 주고 max_search_rounds 를 요청한 만큼 늘린다.
  2) 예외를 잡아 크래시 대신 node_status=failed 로 기록한다 (관점 노드는 failed findings 를 대신 반환).
  3) 시도 횟수(node_status.attempts)를 올리고 처리한 재작업 요청을 비운다.
워커는 동시 실행될 수 있으므로 reducer 가 있는 키(node_status, rework_requests)와 자기 결과 키만 쓴다.
"""

from __future__ import annotations

from typing import Callable

from graph.sufficiency import PERSPECTIVES, attempts

MAX_ROUNDS = 3
STATUS_MAP = {"complete": "ok", "partial": "partial", "failed": "failed",
              "passed": "ok", "needs_review": "partial"}


def failed_findings(perspective: str, error: str) -> dict:
    return {"perspective": perspective, "status": "failed", "records": [], "claims": [], "gaps": [],
            "limitations": [f"실행 실패: {error}"], "input_evidence_ids": []}


def result_status(name: str, update: dict) -> str:
    if name in PERSPECTIVES:
        result = update.get(f"{name}_findings")
    elif name == "synthesis":
        result = update.get("synthesis")
    else:
        result = ((update.get("quality_by_perspective") or {}).get("report"))
    if not result:
        return "failed"
    return STATUS_MAP.get(result.get("status"), "partial")


def as_worker(name: str, node: Callable[[dict], dict]) -> Callable[[dict], dict]:
    def worker(state: dict) -> dict:
        request = (state.get("rework_requests") or {}).get(name)
        inp = state
        if request:
            parent = dict(state.get("request") or {})
            parent["max_search_rounds"] = min(MAX_ROUNDS, int(parent.get("max_search_rounds") or 1) + int(request["extra_rounds"]))
            inp = {**state, "request": parent, "rework_hint": request}
        error = None
        try:
            update = dict(node(inp) or {})
            status = result_status(name, update)
        except Exception as exc:  # 한 워커의 실패가 그래프 전체를 죽이지 않게 한다
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
            update = {f"{name}_findings": failed_findings(name, error)} if name in PERSPECTIVES else {}
            status = "failed"
        update["node_status"] = {name: {"status": status, "error": error, "attempts": attempts(state, name) + 1}}
        update["rework_requests"] = {name: None}
        return update

    worker.__name__ = f"worker_{name}"
    return worker

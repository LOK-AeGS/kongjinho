"""보고서 에이전트 노드: 확정 AppState ↔ 보고서 내부 정규형 변환."""

from __future__ import annotations

from functools import partial

from agents.report.prompts import PROMPT_VERSION
from agents.report.state import ReportAgentDeps
from agents.report.subgraph import run_report


def report_agent(state: dict, *, deps: ReportAgentDeps | None = None) -> dict:
    """확정 AppState에서 읽고 보고서 에이전트 소유 키만 부분 업데이트한다."""
    final = run_report(state, deps)
    report = final["report"]
    report_meta = {
        "prompt_version": PROMPT_VERSION,
        "finalization_steps": final["finalization_steps"],
        "generation": final["generation"],
    }
    if report.get("pdf_path"):
        report_meta["pdf_path"] = report["pdf_path"]
    return {
        "report_sections": final["report_sections"],
        "references": final["references"],
        "quality_by_perspective": {
            "report": {
                "status": report["quality_status"],
                "violations": report["completion"]["errors"],
                "warnings": list(report["completion"]["gaps"]),
                "checked_claim_ids": final["checked_claim_ids"],
            }
        },
        "retries": {"report": report["completion"]["revision_rounds_used"]},
        "run_meta": {"report": report_meta},
    }


def make_node(deps: ReportAgentDeps | None = None):
    """부모 Graph에 등록할 보고서 노드 함수를 만든다."""
    return partial(report_agent, deps=deps)

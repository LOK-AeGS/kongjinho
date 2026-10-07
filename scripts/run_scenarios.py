"""Supervisor 의 동적 동작을 API 비용 없이 재현하는 시나리오 3종.

같은 코드(graph/build.py + graph/supervisor.py + graph/quality.py)를 State 만 다르게 해서 실행하고,
시나리오마다 경로·재작업 횟수·종료 상태가 달라지는 것을 표로 남긴다. LangSmith 가 켜져 있으면 시나리오별로 구분해서 트레이스된다.

  python -m scripts.run_scenarios            # docs/TRACE_SCENARIOS.md 를 새로 쓴다
  LANGSMITH_TRACING=true LANGSMITH_API_KEY=... python -m scripts.run_scenarios   # 시나리오별 트레이스 (캡처용)

워커는 가짜지만 supervisor·충분성 판단·품질 평가·종료 보장은 실제 코드다. 실제 에이전트의 실행 기록은 `python main.py --live all` 참고.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.build import build_graph  # noqa: E402
from graph.state import create_initial_state  # noqa: E402
from graph.supervisor import make_supervisor  # noqa: E402

PERSPECTIVES = ("technical", "market", "stakeholder", "domain")
REQUEST = {"as_of": "2026-09-22", "language": "ko", "scope": "datacenter_inference", "max_search_rounds": 1}
GOOD_REPORT = "# SUMMARY\n관점별 평가 요약. 〔근거: {ids}〕\n# 1. 본문\n내용\n# REFERENCE\n- 출처\n"
BAD_REPORT = "# SUMMARY\n이 기술을 추천한다. 〔근거: {ids}〕\n# 1. 본문\n내용\n# REFERENCE\n- 출처\n"


def evidence(perspective: str, i: int) -> dict:
    return {"id": f"ev:{perspective}:{i}", "claim_id": f"c:{perspective}:{i}", "doc_id": None, "title": "t", "author_or_org": "a",
            "source_type": "web", "primary_or_secondary": "primary", "direct_or_proxy": "direct",
            "url": f"https://src{i}.example.com/{perspective}", "published_at": "2026-01", "accessed_at": "2026-09-22",
            "page_or_locator": "b", "quote": "q", "stance": "support", "evidence_level": "unknown", "metric_tag": None,
            "perspective": perspective, "content_hash": "h"}


class Scenario:
    """관점별 호출 횟수에 따라 근거 건수를 정하는 가짜 워커 모음. counts[p] 는 호출 순서대로의 근거 건수(마지막 값 반복)."""

    def __init__(self, name: str, description: str, counts: dict[str, list[int]], reports: list[str]):
        self.name, self.description, self.counts, self.reports = name, description, counts, reports
        self.calls: dict[str, int] = {}

    def perspective(self, p: str):
        def node(state: dict) -> dict:
            n_call = self.calls[p] = self.calls.get(p, 0) + 1
            seq = self.counts.get(p, [3])
            n = seq[min(n_call, len(seq)) - 1]
            findings = {"perspective": p, "status": "complete", "records": [], "claims": [], "gaps": [], "limitations": [], "input_evidence_ids": []}
            return {f"{p}_findings": findings, "evidence_store": {e["id"]: e for e in (evidence(p, i) for i in range(n))}}
        return node

    def synthesis(self, state: dict) -> dict:
        self.calls["synthesis"] = self.calls.get("synthesis", 0) + 1
        return {"synthesis": {"status": "complete", "retry_requests": [], "meta": {}}}

    def report(self, state: dict) -> dict:
        n_call = self.calls["report"] = self.calls.get("report", 0) + 1
        ids = ", ".join(sorted(next(i for i, e in state["evidence_store"].items() if e["perspective"] == p) for p in PERSPECTIVES))
        template = self.reports[min(n_call, len(self.reports)) - 1]
        return {"report_sections": {"final_markdown": template.format(ids=ids)}, "references": {},
                "quality_by_perspective": {"report": {"status": "passed", "violations": [], "warnings": [], "checked_claim_ids": []}}}

    def nodes(self) -> dict:
        return {**{p: self.perspective(p) for p in PERSPECTIVES}, "synthesis": self.synthesis, "report": self.report}


SCENARIOS = [
    ("01-normal", "모든 관점의 근거가 충분하고 보고서가 한 번에 품질 평가를 통과", {}, [GOOD_REPORT]),
    ("02-rework", "이해관계자 근거 부족 → 그 관점만 재작업, 보고서가 중립성 미달 → 보고서만 재작업(품질 Loop)",
     {"stakeholder": [1, 3]}, [BAD_REPORT, GOOD_REPORT]),
    ("03-degraded", "이해관계자 근거가 계속 부족 → 재작업 예산(2회) 소진, 한계를 안고 보고서까지 진행해 degraded 로 종료",
     {"stakeholder": [1]}, [GOOD_REPORT]),
]


def run_one(name: str, description: str, counts: dict, reports: list[str]) -> dict:
    scenario = Scenario(name, description, counts, reports)
    app = build_graph(**scenario.nodes(), supervisor=make_supervisor())
    state = create_initial_state(request=REQUEST, selected_tech={}, corpus_manifest=[], run_id=name)
    config = {"run_name": "supervisor_run", "tags": ["supervisor", f"scenario-{name}"],
              "metadata": {"scenario": name, "trace_id": state["trace_id"], "run_id": state["run_id"]},
              "configurable": {"thread_id": f"{name}"}, "recursion_limit": 50}
    final = app.invoke(state, config)
    path = [f"{d['action']}({','.join(d['targets']) or '-'})" for d in final["decision_log"]]
    return {"name": name, "description": description, "turns": final["step_count"], "path": path,
            "calls": scenario.calls, "quality_iterations": final["quality_iterations"],
            "reworks": sum(max(v["attempts"] - 1, 0) for k, v in final["node_status"].items() if k in PERSPECTIVES),
            "final_status": final["final_status"], "degraded_reasons": (final["run_meta"].get("supervisor") or {}).get("degraded_reasons", []),
            "trace_id": final["trace_id"]}


def render(results: list[dict]) -> str:
    lines = ["# Supervisor 동작 시나리오", "",
             "`python -m scripts.run_scenarios` 가 만든 파일이다. 워커는 가짜지만 supervisor·충분성 판단·품질 평가·종료 보장은 실제 코드이고, "
             "State 만 달라서 경로가 달라진다.", "",
             "| 시나리오 | supervisor 턴 | 관점 재작업 | 품질 평가 | 종료 상태 |", "|---|---|---|---|---|"]
    lines += [f"| {r['name']} | {r['turns']} | {r['reworks']}회 | {r['quality_iterations']}회 | {r['final_status']} |" for r in results]
    for r in results:
        lines += ["", f"## {r['name']}", "", r["description"], "", f"- trace_id: `{r['trace_id']}` (LangSmith 태그 `scenario-{r['name']}`)",
                  f"- 워커 호출 수: {', '.join(f'{k} {v}' for k, v in r['calls'].items())}", "",
                  "경로: " + " → ".join(r["path"])]
        if r["degraded_reasons"]:
            lines += ["", "저하 사유:"] + [f"- {reason}" for reason in r["degraded_reasons"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervisor 시나리오 3종 재현")
    parser.add_argument("--output", type=Path, default=Path("docs/TRACE_SCENARIOS.md"))
    args = parser.parse_args()
    from main import load_env
    load_env()
    results = [run_one(*scenario) for scenario in SCENARIOS]
    text = render(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    paths = {tuple(r["path"]) for r in results}
    print(f"서로 다른 경로 {len(paths)}개 / 시나리오 {len(results)}개 — 결과 저장: {args.output}")
    return 0 if len(paths) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

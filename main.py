"""부모 그래프 실행 진입점. 의존성 준비 → 노드 만들기 → build_graph → 실행 → 결과 저장.

  python main.py                                   # 전부 오프라인 (API 키 불필요)
  python main.py --live synthesis                  # 평가 종합만 실제 LLM
  python main.py --live market,stakeholder,synthesis,report   # 가능한 노드 전부 실제 실행 (비용 발생)

노드별 실행 방식
  technical          : 임시 노드 (fixture 재생). 기술 조사 에이전트 PR 전.
  market, stakeholder, domain: 기본 fixture 재생, --live 면 실제 에이전트 (Tavily·OpenAI)
  synthesis          : 기본 템플릿 서술 (LLM 없음), --live 면 OpenAIWriter
  report             : 기본 deterministic (LLM 없음), --live 면 LLM 작성

결과: outputs/graph/<실행시각>/ 에 summary.md, report.md, final_state.json, trace.json, graph.mmd
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from graph.build import build_graph
from graph.state import create_initial_state
from graph.stubs import DEFAULT_FIXTURE, load_fixture, replay_node

AGENTS = ("technical", "market", "stakeholder", "domain", "synthesis", "report")
LIVE_CAPABLE = ("market", "stakeholder", "domain", "synthesis", "report")


def load_env(path: Path = Path(".env")) -> None:
    """.env 의 키를 환경변수로. 이미 있으면 덮어쓰지 않는다. 값은 출력하지 않는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_nodes(live: set[str], fixture: dict) -> tuple[dict, dict]:
    """노드 함수와 노드별 실행 방식 설명을 만든다."""
    nodes, modes = {}, {}

    nodes["technical"], modes["technical"] = replay_node("technical", fixture), "임시 (fixture 재생, 기술 조사 PR 전)"

    if "domain" in live:
        from langchain.chat_models import init_chat_model

        from agents.domain import DomainAgentDeps, make_node as domain_node
        from agents.domain.tools.websearch import build_search_provider

        # scripts/run_domain.py 와 같은 설정. 임베딩은 긴 문서 색인에만 쓰인다 (DOMAIN_EMBEDDING="" 이면 BM25만).
        embedding = os.getenv("DOMAIN_EMBEDDING", "BAAI/bge-m3") or None
        deps = DomainAgentDeps(
            llm=init_chat_model(os.getenv("DOMAIN_MODEL", "gpt-4o"), model_provider="openai", temperature=0),
            search_provider=build_search_provider(Path("data/search_cache")),
            embedding_model=embedding,
            fetch_cache_dir=Path("data/fetch_cache"),
        )
        nodes["domain"], modes["domain"] = domain_node(deps), f"실제 ({os.getenv('DOMAIN_MODEL', 'gpt-4o')} + Tavily, 임베딩 {embedding or '없음'})"
    else:
        nodes["domain"], modes["domain"] = replay_node("domain", fixture), "fixture 재생"

    if "market" in live:
        from langchain_openai import ChatOpenAI

        from agents.market import MarketAgentDeps, make_node as market_node
        from agents.market.rag.fetch import fetch_document
        from agents.market.rag.tools import tavily_web_search

        deps = MarketAgentDeps(
            llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0, timeout=60, max_retries=2),
            strong_llm=ChatOpenAI(model="gpt-4.1", temperature=0, timeout=60, max_retries=2),
            web_search=lambda q: tavily_web_search(q, max_results=3),
            fetch_body=lambda url: fetch_document(url, cache_dir=Path("data/fetch_cache/market")),
        )
        nodes["market"], modes["market"] = market_node(deps), "실제 (gpt-4.1-mini/gpt-4.1 + Tavily)"
    else:
        nodes["market"], modes["market"] = replay_node("market", fixture), "fixture 재생"

    if "stakeholder" in live:
        from agents.stakeholder_eval import make_node as stakeholder_node

        nodes["stakeholder"], modes["stakeholder"] = stakeholder_node(), "실제 (agents.stakeholder_eval, gpt-4.1-mini)"
    else:
        nodes["stakeholder"], modes["stakeholder"] = replay_node("stakeholder", fixture), "fixture 재생"

    from agents.synthesis import make_node as synthesis_node

    if "synthesis" in live:
        from agents.synthesis.writer import OpenAIWriter

        writer = OpenAIWriter()
        nodes["synthesis"], modes["synthesis"] = synthesis_node(writer), f"실제 ({writer.model})"
    else:
        nodes["synthesis"], modes["synthesis"] = synthesis_node(), "실제 노드, 템플릿 서술 (LLM 없음)"

    from agents.report import ReportAgentDeps, make_node as report_node

    if "report" in live:
        deps = ReportAgentDeps()
        nodes["report"], modes["report"] = report_node(deps), f"실제 ({deps.model})"
    else:
        nodes["report"], modes["report"] = report_node(ReportAgentDeps(generation_mode="deterministic")), "실제 노드, deterministic (LLM 없음)"

    return nodes, modes


def _updated_keys(result) -> list[str]:
    if isinstance(result, dict):
        return sorted(result)
    if isinstance(result, (list, tuple)):
        return sorted({item[0] for item in result if isinstance(item, (list, tuple)) and item})
    return []


def run_graph(nodes: dict, initial: dict) -> tuple[dict, list[dict]]:
    """그래프를 실행하고 (최종 State, 실행 기록)을 돌려준다. 같은 step 의 노드는 병렬 실행이다."""
    app = build_graph(**nodes)
    final, trace = initial, []
    for mode, event in app.stream(initial, stream_mode=["debug", "values"]):
        if mode == "values":
            final = event
        elif event.get("type") == "task_result":
            payload = event["payload"]
            trace.append({"step": event["step"], "node": payload["name"],
                          "updated": _updated_keys(payload.get("result")),
                          "error": payload.get("error")})
    return final, trace


def node_status(final: dict) -> dict[str, str]:
    status = {p: (final.get(f"{p}_findings") or {}).get("status", "없음") for p in ("technical", "market", "stakeholder", "domain")}
    status["synthesis"] = (final.get("synthesis") or {}).get("status", "없음")
    status["report"] = ((final.get("quality_by_perspective") or {}).get("report") or {}).get("status", "없음")
    return status


def steps_view(trace: list[dict]) -> list[str]:
    by_step: dict[int, list[str]] = {}
    for t in trace:
        by_step.setdefault(t["step"], []).append(t["node"])
    return [" + ".join(sorted(names, key=AGENTS.index)) for _, names in sorted(by_step.items())]


def render_summary(modes: dict, trace: list[dict], final: dict, fixture_note: str | None) -> str:
    status = node_status(final)
    synthesis = final.get("synthesis") or {}
    counts = (synthesis.get("meta") or {}).get("counts", {})
    lines = ["# 부모 그래프 실행 결과", ""]
    if fixture_note:
        lines += [f"> ⚠️ fixture 재생 노드가 있습니다: {fixture_note}", ""]
    lines += ["## 실행 경로", "", "설계(§8.1): `① 기술 조사 → ②③④ 병렬 → ⑤ 평가 종합 → ⑥ 보고서`", "",
              "실제: `START → " + " → ".join(steps_view(trace)) + " → END`", "",
              "| step | 노드 | 갱신한 키 | 오류 |", "|---|---|---|---|"]
    lines += [f"| {t['step']} | {t['node']} | {', '.join(t['updated'])} | {t['error'] or ''} |" for t in trace]
    lines += ["", "## 노드별 실행 방식과 결과", "", "| 노드 | 실행 방식 | 상태 |", "|---|---|---|"]
    lines += [f"| {a} | {modes[a]} | {status[a]} |" for a in AGENTS]
    lines += ["", "## 공통 State", "",
              f"- evidence_store: 근거 {len(final.get('evidence_store') or {})}건",
              f"- synthesis: 매트릭스 {counts.get('matrix_cells', 0)}칸, 상충 {counts.get('conflicts', 0)}, "
              f"공유 근거 {counts.get('shared_evidence', 0)}, 보완 {counts.get('complements', 0)}, "
              f"요약 주장 {len(synthesis.get('summary_claims') or [])}개",
              f"- report_sections: {len(final.get('report_sections') or {})}개 섹션, references {len(final.get('references') or {})}건",
              "", "보고서 본문은 같은 폴더의 `report.md`, 평가 종합 상세는 `final_state.json`의 `synthesis` 참고."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python main.py", description="KV cache 다관점 평가 부모 그래프 실행")
    parser.add_argument("--live", default="", help=f"실제로 실행할 노드 (쉼표 구분): {', '.join(LIVE_CAPABLE)}. 비용 발생")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="fixture 재생 노드가 쓸 AppState JSON")
    parser.add_argument("--as-of", default=None, help="조사 기준일 YYYY-MM-DD (기본: fixture 의 기준일)")
    parser.add_argument("--rounds", type=int, default=1, help="실제 실행 노드의 최대 검색 라운드")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/graph"))
    args = parser.parse_args()

    live = {x.strip() for x in args.live.split(",") if x.strip()}
    unknown = live - set(LIVE_CAPABLE)
    if unknown:
        parser.error(f"--live 에 쓸 수 없는 노드: {', '.join(sorted(unknown))} (가능: {', '.join(LIVE_CAPABLE)})")
    if live:
        load_env()
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY 가 없습니다. .env 에 넣으세요.")
        if live & {"market", "domain"} and not os.getenv("TAVILY_API_KEY"):
            parser.error("시장·도메인 실제 실행에는 TAVILY_API_KEY 가 필요합니다.")
        print(f"실제 실행 노드: {', '.join(sorted(live, key=AGENTS.index))} (API 비용이 발생합니다)")

    fixture = load_fixture(args.fixture)
    initial = create_initial_state(
        request={"as_of": args.as_of or fixture["request"]["as_of"], "language": "ko",
                 "scope": "datacenter_inference", "max_search_rounds": args.rounds},
        selected_tech=fixture["selected_tech"],
        corpus_manifest=fixture.get("corpus_manifest") or [],
    )
    nodes, modes = build_nodes(live, fixture)
    final, trace = run_graph(nodes, initial)

    folder = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=True)
    replayed = [a for a in AGENTS if modes[a].startswith(("임시", "fixture"))]
    note = f"{', '.join(replayed)} 는 합성 fixture 결과이며 실제 조사가 아닙니다." if replayed else None
    (folder / "summary.md").write_text(render_summary(modes, trace, final, note), encoding="utf-8")
    (folder / "report.md").write_text("\n\n".join((final.get("report_sections") or {}).values()) + "\n", encoding="utf-8")
    (folder / "final_state.json").write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (folder / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "graph.mmd").write_text(build_graph(**nodes).get_graph().draw_mermaid(), encoding="utf-8")

    status = node_status(final)
    print("실행 경로: START → " + " → ".join(steps_view(trace)) + " → END")
    for a in AGENTS:
        print(f"  {a:<12} {status[a]:<13} {modes[a]}")
    if note:
        print(f"주의: {note}")
    print(f"결과 폴더: {folder.resolve()}")
    return 0 if all(not t["error"] for t in trace) else 1


if __name__ == "__main__":
    raise SystemExit(main())

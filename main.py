"""부모 그래프 실행 진입점. 의존성 준비 → 노드 만들기 → build_graph → 실행 → 결과 저장.

  python main.py                                   # 전부 오프라인 (API 키 불필요)
  python main.py --live synthesis                  # 평가 종합만 실제 LLM
  python main.py --live all                        # 여섯 노드 전부 실제 실행 (비용 발생)

입력: 기술 조사 에이전트의 팀 고정 입력(agents/technical/config.py)을 부모 그래프 공통 입력으로 쓴다.
      corpus_manifest 는 Pool A 고정 코퍼스(data/technical/manifest.json)에서 채운다.

노드별 실행 방식
  technical, market, stakeholder, domain: 기본 fixture 재생, --live 면 실제 에이전트 (Tavily·OpenAI)
  synthesis          : 기본 템플릿 서술 (LLM 없음), --live 면 OpenAIWriter
  report             : 기본 deterministic (LLM 없음), --live 면 LLM 작성

결과: outputs/graph/<실행시각>/ 에 report.pdf, report.md, summary.md, final_state.json, trace.json, graph.mmd,
      steps/<단계>_<노드>.json (노드별 중간 결과). 실행 중에는 노드가 끝날 때마다 콘솔에 한 줄씩 출력, --debug 면 상세까지
      (PDF 는 reportlab 과 한글 폰트가 필요. --no-pdf 로 끌 수 있음)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

from graph.build import build_graph
from graph.state import create_initial_state
from graph.stubs import DEFAULT_FIXTURE, load_fixture, replay_node

AGENTS = ("technical", "market", "stakeholder", "domain", "synthesis", "report")
LIVE_CAPABLE = AGENTS


def initial_state(*, as_of: str | None = None, rounds: int = 1) -> dict:
    """팀 고정 입력으로 AppState 초기값을 만든다.

    기술 조사 에이전트는 selected_tech 가 자기 고정값과 정확히 같아야 실행되므로(agents/technical/config.py),
    그 값을 부모 그래프 전체의 입력으로 쓴다. corpus_manifest 는 Pool A 고정 코퍼스 목록이다.
    """
    from agents.technical.config import DEFAULT_REQUEST, DEFAULT_SELECTED_TECH
    from agents.technical.corpus import default_source_dir, load_manifest

    request = {**DEFAULT_REQUEST, "as_of": as_of or DEFAULT_REQUEST["as_of"], "max_search_rounds": rounds}
    manifest = [doc.to_parent_meta(default_source_dir()) for doc in load_manifest()]
    return create_initial_state(request=request, selected_tech=copy.deepcopy(DEFAULT_SELECTED_TECH), corpus_manifest=manifest)


def load_env(path: Path = Path(".env")) -> None:
    """.env 의 키를 환경변수로. 이미 있으면 덮어쓰지 않는다. 값은 출력하지 않는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_nodes(live: set[str], fixture: dict, pdf_path: Path | None = None) -> tuple[dict, dict]:
    """노드 함수와 노드별 실행 방식 설명을 만든다. pdf_path 를 주면 보고서 노드가 PDF 도 저장한다."""
    nodes, modes = {}, {}

    if "technical" in live:
        from agents.technical import make_node as technical_node

        # 기본 의존성: Pool A 고정 PDF(BM25 + BGE-M3 + RRF) + Tavily + gpt-4.1. 첫 호출 때 코퍼스를 파싱한다.
        model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        nodes["technical"], modes["technical"] = technical_node(), f"실제 (Pool A RAG + Tavily, {model}, TRL Gate)"
    else:
        nodes["technical"], modes["technical"] = replay_node("technical", fixture), "fixture 재생"

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
            web_search=tavily_web_search,
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
        deps = ReportAgentDeps(pdf_output_path=pdf_path)
        nodes["report"], modes["report"] = report_node(deps), f"실제 ({deps.model})"
    else:
        deps = ReportAgentDeps(generation_mode="deterministic", pdf_output_path=pdf_path)
        nodes["report"], modes["report"] = report_node(deps), "실제 노드, deterministic (LLM 없음)"

    return nodes, modes


def _updated_keys(result) -> list[str]:
    if isinstance(result, dict):
        return sorted(result)
    if isinstance(result, (list, tuple)):
        return sorted({item[0] for item in result if isinstance(item, (list, tuple)) and item})
    return []


def _as_dict(result) -> dict:
    """debug 이벤트의 노드 반환값(list of (key, value) 또는 dict)을 dict 로."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        return {item[0]: item[1] for item in result if isinstance(item, (list, tuple)) and len(item) == 2}
    return {}


def step_brief(node: str, update: dict) -> str:
    """노드가 방금 반환한 값의 한 줄 요약 (진행 표시용)."""
    findings = update.get(f"{node}_findings")
    if findings:
        return (f"status={findings.get('status')}, 판정 {len(findings.get('records') or [])}칸, "
                f"주장 {len(findings.get('claims') or [])}개, 공백 {len(findings.get('gaps') or [])}개, "
                f"근거 {len(update.get('evidence_store') or {})}건")
    if node == "synthesis" and update.get("synthesis"):
        s = update["synthesis"]
        counts = (s.get("meta") or {}).get("counts", {})
        return (f"status={s.get('status')}, 매트릭스 {counts.get('matrix_cells', 0)}칸, 상충 {counts.get('conflicts', 0)}, "
                f"보완 {counts.get('complements', 0)}, 요약 주장 {len(s.get('summary_claims') or [])}개, "
                f"검사 {((s.get('meta') or {}).get('quality') or {}).get('status')}")
    if node == "report":
        q = (update.get("quality_by_perspective") or {}).get("report") or {}
        return f"status={q.get('status')}, 섹션 {len(update.get('report_sections') or {})}개, 참고문헌 {len(update.get('references') or {})}건, 위반 {len(q.get('violations') or [])}건"
    return ", ".join(sorted(update)) or "(반환값 없음)"


def step_detail(node: str, update: dict) -> list[str]:
    """--debug 때 보여줄 상세: 주장·공백·요약 문장·위반 샘플."""
    lines = []
    findings = update.get(f"{node}_findings")
    if findings:
        lines += [f"      주장: [{c.get('technology')}] {c.get('text', '')[:110]}" for c in (findings.get("claims") or [])[:3]]
        lines += [f"      공백: [{g.get('technology')}] {g.get('criterion')}: {g.get('reason', '')[:80]}" for g in (findings.get("gaps") or [])[:3]]
        lines += [f"      한계: {x[:110]}" for x in (findings.get("limitations") or [])[:2]]
    if node == "synthesis" and update.get("synthesis"):
        s = update["synthesis"]
        lines += [f"      내부 경로: {' → '.join(t['node'] for t in (s.get('meta') or {}).get('trace', []))}"]
        lines += [f"      요약: [{c.get('technology')}] {c.get('text', '')[:110]}" for c in (s.get("summary_claims") or [])[:3]]
        lines += [f"      제거: {d.get('violations')}" for d in (s.get("dropped_sentences") or [])[:2]]
    if node == "report":
        q = (update.get("quality_by_perspective") or {}).get("report") or {}
        lines += [f"      위반: {v[:110]}" for v in (q.get("violations") or [])[:5]]
    return lines


def run_graph(nodes: dict, initial: dict, on_step=None) -> tuple[dict, list[dict]]:
    """그래프를 실행하고 (최종 State, 실행 기록)을 돌려준다. 같은 step 의 노드는 병렬 실행이다.

    on_step(record, update) 를 주면 노드가 끝날 때마다 호출한다 (진행 표시·중간 결과 저장용).
    """
    app = build_graph(**nodes)
    final, trace, started = initial, [], {}
    for mode, event in app.stream(initial, stream_mode=["debug", "values"]):
        if mode == "values":
            final = event
        elif event.get("type") == "task":
            started[event["payload"]["id"]] = event["timestamp"]
        elif event.get("type") == "task_result":
            payload = event["payload"]
            begin = started.get(payload.get("id"))
            seconds = (datetime.fromisoformat(event["timestamp"]) - datetime.fromisoformat(begin)).total_seconds() if begin else None
            record = {"step": event["step"], "node": payload["name"],
                      "updated": _updated_keys(payload.get("result")),
                      "seconds": round(seconds, 1) if seconds is not None else None,
                      "error": payload.get("error")}
            trace.append(record)
            if on_step:
                on_step(record, _as_dict(payload.get("result")))
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
              "| step | 노드 | 소요 시간 | 갱신한 키 | 오류 |", "|---|---|---|---|---|"]
    lines += [f"| {t['step']} | {t['node']} | {t.get('seconds') if t.get('seconds') is not None else '-'}초 | "
              f"{', '.join(t['updated'])} | {t['error'] or ''} |" for t in trace]
    lines += ["", "## 노드별 실행 방식과 결과", "", "| 노드 | 실행 방식 | 상태 |", "|---|---|---|"]
    lines += [f"| {a} | {modes[a]} | {status[a]} |" for a in AGENTS]
    lines += ["", "## 공통 State", "",
              f"- evidence_store: 근거 {len(final.get('evidence_store') or {})}건",
              f"- synthesis: 매트릭스 {counts.get('matrix_cells', 0)}칸, 상충 {counts.get('conflicts', 0)}, "
              f"공유 근거 {counts.get('shared_evidence', 0)}, 보완 {counts.get('complements', 0)}, "
              f"요약 주장 {len(synthesis.get('summary_claims') or [])}개",
              f"- report_sections: {len(final.get('report_sections') or {})}개 섹션, references {len(final.get('references') or {})}건",
              f"- PDF: {((final.get('run_meta') or {}).get('report') or {}).get('pdf_path') or '만들지 않음'}",
              "", "보고서는 같은 폴더의 `report.pdf`·`report.md`, 평가 종합 상세는 `final_state.json`의 `synthesis` 참고."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python main.py", description="KV cache 다관점 평가 부모 그래프 실행")
    parser.add_argument("--live", default="", help=f"실제로 실행할 노드 (쉼표 구분): all 또는 {', '.join(LIVE_CAPABLE)}. 비용 발생")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="fixture 재생 노드가 쓸 AppState JSON")
    parser.add_argument("--as-of", default=None, help="조사 기준일 YYYY-MM-DD (기본: 팀 고정 입력의 기준일 2026-09-22)")
    parser.add_argument("--rounds", type=int, default=1, choices=(1, 2), help="시장·도메인의 최대 검색 라운드 (기술 조사는 자체 고정값 2)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/graph"))
    parser.add_argument("--no-pdf", action="store_true", help="보고서 PDF 를 만들지 않음 (reportlab 없이 실행할 때)")
    parser.add_argument("--debug", action="store_true", help="노드가 끝날 때마다 주장·공백·위반 샘플까지 출력")
    args = parser.parse_args()

    live = {x.strip() for x in args.live.split(",") if x.strip()}
    if "all" in live:
        live = set(LIVE_CAPABLE)
    unknown = live - set(LIVE_CAPABLE)
    if unknown:
        parser.error(f"--live 에 쓸 수 없는 노드: {', '.join(sorted(unknown))} (가능: {', '.join(LIVE_CAPABLE)})")
    if live:
        load_env()
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY 가 없습니다. .env 에 넣으세요.")
        if live & {"technical", "market", "domain"} and not os.getenv("TAVILY_API_KEY"):
            parser.error("기술 조사·시장·도메인 실제 실행에는 TAVILY_API_KEY 가 필요합니다.")
        print(f"실제 실행 노드: {', '.join(sorted(live, key=AGENTS.index))} (API 비용이 발생합니다)")

    fixture = load_fixture(args.fixture)
    initial = initial_state(as_of=args.as_of, rounds=args.rounds)
    # 보고서 노드가 실행 중에 PDF 를 쓰므로 결과 폴더를 먼저 만든다.
    folder = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = None if args.no_pdf else (folder / "report.pdf").resolve()
    if pdf_path:
        try:
            import reportlab  # noqa: F401
        except ImportError:
            parser.error("PDF 생성에는 reportlab 이 필요합니다: pip install 'reportlab>=4.4.9,<5' (또는 --no-pdf)")

    nodes, modes = build_nodes(live, fixture, pdf_path)
    steps_dir = folder / "steps"
    steps_dir.mkdir()

    def on_step(record: dict, update: dict) -> None:
        # 노드가 끝나는 즉시: 콘솔에 한 줄, steps/ 에 그 노드의 반환값 전체를 저장
        took = f"{record['seconds']:.1f}초" if record["seconds"] is not None else "-"
        mark = "✗" if record["error"] else "✓"
        print(f"  [{record['step']}] {mark} {record['node']:<12} {took:>7}  {step_brief(record['node'], update)}", flush=True)
        if record["error"]:
            print(f"      오류: {record['error']}", flush=True)
        if args.debug:
            for line in step_detail(record["node"], update):
                print(line, flush=True)
        path = steps_dir / f"{record['step']:02d}_{record['node']}.json"
        path.write_text(json.dumps({"trace": record, "update": update}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"실행 시작 (결과 폴더: {folder.resolve()})", flush=True)
    final, trace = run_graph(nodes, initial, on_step)
    replayed = [a for a in AGENTS if modes[a].startswith(("임시", "fixture"))]
    note = f"{', '.join(replayed)} 는 합성 fixture 결과이며 실제 조사가 아닙니다." if replayed else None
    (folder / "summary.md").write_text(render_summary(modes, trace, final, note), encoding="utf-8")
    sections = final.get("report_sections") or {}
    # final_markdown 이 보고서 에이전트가 만든 완성본이다 (섹션을 다시 이어 붙이면 본문이 중복된다).
    (folder / "report.md").write_text(sections.get("final_markdown") or "\n\n".join(sections.values()), encoding="utf-8")
    (folder / "final_state.json").write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (folder / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "graph.mmd").write_text(build_graph(**nodes).get_graph().draw_mermaid(), encoding="utf-8")

    status = node_status(final)
    seconds = {t["node"]: t.get("seconds") for t in trace}
    print("실행 경로: START → " + " → ".join(steps_view(trace)) + " → END")
    for a in AGENTS:
        took = f"{seconds[a]:>6.1f}초" if seconds.get(a) is not None else "      -"
        print(f"  {a:<12} {status[a]:<13} {took}  {modes[a]}")
    if note:
        print(f"주의: {note}")
    pdf = ((final.get("run_meta") or {}).get("report") or {}).get("pdf_path")
    print(f"보고서 PDF: {pdf or '만들지 않음'}")
    print(f"결과 폴더: {folder.resolve()}")
    return 0 if all(not t["error"] for t in trace) else 1


if __name__ == "__main__":
    raise SystemExit(main())

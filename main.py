"""부모 그래프 실행 진입점. 의존성 준비 → 노드 만들기 → build_graph → 실행 → 결과 저장.

  python main.py                                   # 전부 오프라인 (API 키 불필요)
  python main.py --live synthesis                  # 평가 종합만 실제 LLM
  python main.py --live all                        # 여섯 노드 전부 실제 실행 (비용 발생)
  python main.py --live all --supervisor llm       # Supervisor 도 LLM(SUPERVISOR_MODEL, 기본 gpt-5.5)이 라우팅

패턴: Supervisor. 모든 노드가 supervisor 로만 돌아오고, supervisor 가 State(관점별 결과·근거 충분성·재작업 예산·품질 평가)를
      보고 다음 노드를 고른다 (graph/supervisor.py). 근거가 부족하면 해당 관점에 재작업을 요청하고, 보고서 뒤 품질 평가가
      미달이면 Loop 를 돈다. 종료는 max_steps·재작업 예산·품질 루프 상한이 보장한다.

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
from graph.quality import make_llm_judge, make_quality_node
from graph.state import create_initial_state
from graph.stubs import DEFAULT_FIXTURE, load_fixture, replay_node
from graph.supervisor import DEFAULT_SUPERVISOR_MODEL, make_llm_proposer, make_supervisor

AGENTS = ("technical", "market", "stakeholder", "domain", "synthesis", "report")
LIVE_CAPABLE = AGENTS
CONTROL_NODES = ("supervisor", "quality")
RECURSION_LIMIT = 50


def initial_state(*, as_of: str | None = None, rounds: int = 1, max_steps: int = 14) -> dict:
    """팀 고정 입력으로 AppState 초기값을 만든다.

    기술 조사 에이전트는 selected_tech 가 자기 고정값과 정확히 같아야 실행되므로(agents/technical/config.py),
    그 값을 부모 그래프 전체의 입력으로 쓴다. corpus_manifest 는 Pool A 고정 코퍼스 목록이다.
    """
    from agents.technical.config import DEFAULT_REQUEST, DEFAULT_SELECTED_TECH
    from agents.technical.corpus import default_source_dir, load_manifest

    request = {**DEFAULT_REQUEST, "as_of": as_of or DEFAULT_REQUEST["as_of"], "max_search_rounds": rounds}
    manifest = [doc.to_parent_meta(default_source_dir()) for doc in load_manifest()]
    return create_initial_state(request=request, selected_tech=copy.deepcopy(DEFAULT_SELECTED_TECH), corpus_manifest=manifest, max_steps=max_steps)


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

        # 기본 의존성: Pool A 고정 PDF(BM25 + BGE-M3 + RRF) + Tavily + OPENAI_MODEL. 첫 호출 때 코퍼스를 파싱한다.
        technical_model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        nodes["technical"], modes["technical"] = (
            technical_node(),
            f"실제 (Pool A RAG + Tavily, {technical_model}, TRL Gate)",
        )
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
        from agents.market.rag import tier as market_tier
        from agents.market.rag.tools import tavily_web_search

        deps = MarketAgentDeps(
            llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0, timeout=60, max_retries=2),
            strong_llm=ChatOpenAI(model="gpt-4.1", temperature=0, timeout=60, max_retries=2),
            web_search=lambda q: tavily_web_search(q, include_domains=market_tier.trusted_domains()),
            fetch_body=lambda url: fetch_document(url, cache_dir=Path("data/fetch_cache/market")),
        )
        nodes["market"], modes["market"] = market_node(deps), "실제 (gpt-4.1-mini/gpt-4.1 + Tavily)"
    else:
        nodes["market"], modes["market"] = replay_node("market", fixture), "fixture 재생"

    if "stakeholder" in live:
        from agents.stakeholder import make_node as stakeholder_node
        from agents.stakeholder.backend import OpenAIBackend

        stakeholder_model = os.getenv("STAKEHOLDER_MODEL", "gpt-5-mini")
        # 원문을 직접 가져와 인용문·locator·날짜를 코드로 대조한다 (검증 없는 LLM 요약은 근거로 쓰지 않는다).
        backend = OpenAIBackend(stakeholder_model, cache_dir=Path("data/fetch_cache/stakeholder"))
        nodes["stakeholder"], modes["stakeholder"] = (
            stakeholder_node(backend),
            f"실제 (agents.stakeholder, 원문 검증, {stakeholder_model})",
        )
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


def build_control(supervisor_mode: str, live: set[str], log_dir: Path | None, *, judge: bool = False) -> tuple[dict, dict]:
    """supervisor·quality 노드를 만든다. supervisor_mode: rule(순수 규칙) | llm | auto(실제 실행이면 llm)."""
    mode = supervisor_mode if supervisor_mode != "auto" else ("llm" if live else "rule")
    model = os.getenv("SUPERVISOR_MODEL", DEFAULT_SUPERVISOR_MODEL)
    client = None
    if mode == "llm" or judge:
        from openai import OpenAI

        client = OpenAI(timeout=60, max_retries=1)
    proposer = make_llm_proposer(client, model) if mode == "llm" else None
    judge_fn = make_llm_judge(client, os.getenv("JUDGE_MODEL", model)) if judge else None
    nodes = {"supervisor": make_supervisor(proposer, log_dir=log_dir), "quality": make_quality_node(judge_fn)}
    modes = {"supervisor": f"LLM 라우팅 + 규칙 guard ({model})" if proposer else "규칙 라우팅 (LLM 없음)",
             "quality": "규칙 검사 + LLM Judge" if judge_fn else "규칙 검사 (groundedness·중립성·편향·커버리지)"}
    return nodes, modes


def make_checkpointer():
    """langgraph-checkpoint-sqlite 가 있으면 파일에, 없으면 메모리에 체크포인트를 둔다."""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        Path("data").mkdir(exist_ok=True)
        return SqliteSaver(sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False))
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


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
    if node == "supervisor":
        entry = (update.get("decision_log") or [{}])[-1]
        targets = ",".join(entry.get("targets") or []) or "-"
        return f"→ {update.get('next_action')} [{targets}] ({entry.get('source')}) {entry.get('reason', '')[:70]}"
    if node == "quality":
        verdict = update.get("quality_verdict") or {}
        return f"passed={verdict.get('passed')}, 미달 {verdict.get('failed_checks') or '없음'}, 대상 {verdict.get('target_perspectives') or '-'}"
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


def run_graph(nodes: dict, initial: dict, on_step=None, *, checkpointer=None, config: dict | None = None) -> tuple[dict, list[dict]]:
    """그래프를 실행하고 (최종 State, 실행 기록)을 돌려준다. 같은 step 의 노드는 병렬 실행이다.

    on_step(record, update) 를 주면 노드가 끝날 때마다 호출한다 (진행 표시·중간 결과 저장용).
    """
    app = build_graph(**nodes, checkpointer=checkpointer)
    final, trace, started = initial, [], {}
    config = {"recursion_limit": RECURSION_LIMIT, **(config or {})}
    for mode, event in app.stream(initial, config, stream_mode=["debug", "values"]):
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
    """supervisor 를 제외한 실행 경로. 같은 step 의 노드는 병렬 실행이라 ' + ' 로 묶는다."""
    order = AGENTS + CONTROL_NODES
    by_step: dict[int, list[str]] = {}
    for t in trace:
        if t["node"] != "supervisor":
            by_step.setdefault(t["step"], []).append(t["node"])
    return [" + ".join(sorted(names, key=order.index)) for _, names in sorted(by_step.items())]


def render_summary(modes: dict, trace: list[dict], final: dict, fixture_note: str | None) -> str:
    status = node_status(final)
    synthesis = final.get("synthesis") or {}
    counts = (synthesis.get("meta") or {}).get("counts", {})
    lines = ["# 부모 그래프 실행 결과", ""]
    if fixture_note:
        lines += [f"> ⚠️ fixture 재생 노드가 있습니다: {fixture_note}", ""]
    meta = (final.get("run_meta") or {}).get("supervisor") or {}
    lines += ["## Supervisor", "", f"- trace_id: `{final.get('trace_id')}` · run_id: `{final.get('run_id')}`",
              f"- 종료 상태: **{final.get('final_status')}** (supervisor 턴 {final.get('step_count')}회 / 상한 {final.get('max_steps')}, "
              f"품질 평가 {final.get('quality_iterations')}회)"]
    lines += [f"- 저하 사유: {reason}" for reason in meta.get("degraded_reasons") or []]
    lines += ["", "| 턴 | 행동 | 대상 | 결정 주체 | 사유 |", "|---|---|---|---|---|"]
    lines += [f"| {d['step']} | {d['action']} | {', '.join(d['targets']) or '-'} | {d['source']} | {d['reason'][:90]} |"
              for d in final.get("decision_log") or []]
    lines += ["", "(State 에는 최근 20건만 남고, 전체 결정 로그는 `logs/<trace_id>.jsonl` 과 LangSmith 트레이스에 있다.)", ""]
    lines += ["## 실행 경로", "", "고정 순서가 아니라 supervisor 가 State 에서 계산한 경로다 (같은 step 은 병렬 실행).", "",
              "실제: `START → " + " → ".join(steps_view(trace)) + " → END`", "",
              "| step | 노드 | 소요 시간 | 갱신한 키 | 오류 |", "|---|---|---|---|---|"]
    lines += [f"| {t['step']} | {t['node']} | {t.get('seconds') if t.get('seconds') is not None else '-'}초 | "
              f"{', '.join(t['updated'])} | {t['error'] or ''} |" for t in trace]
    lines += ["", "## 노드별 실행 방식과 결과", "", "| 노드 | 실행 방식 | 상태 | 시도 |", "|---|---|---|---|"]
    runs = final.get("node_status") or {}
    lines += [f"| {a} | {modes[a]} | {status[a]} | {(runs.get(a) or {}).get('attempts', 0)} |" for a in AGENTS]
    lines += [f"| {c} | {modes[c]} | {(runs.get(c) or {}).get('status', '-') if c == 'quality' else '-'} | "
              f"{(runs.get(c) or {}).get('attempts', 0) if c == 'quality' else final.get('step_count')} |" for c in CONTROL_NODES if c in modes]
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
    parser.add_argument("--supervisor", default="auto", choices=("auto", "llm", "rule"),
                        help="supervisor 라우팅: llm(SUPERVISOR_MODEL + 규칙 guard) / rule(규칙만, API 불필요) / auto(실제 실행이면 llm)")
    parser.add_argument("--judge", action="store_true", help="품질 평가에 LLM Judge 를 추가 (실패 항목만 추가할 수 있음)")
    parser.add_argument("--max-steps", type=int, default=14, help="supervisor 턴 상한 (종료 보장)")
    args = parser.parse_args()

    load_env()  # LANGSMITH_* 등 키를 환경변수로 (이미 있으면 덮어쓰지 않는다)
    live = {x.strip() for x in args.live.split(",") if x.strip()}
    if "all" in live:
        live = set(LIVE_CAPABLE)
    unknown = live - set(LIVE_CAPABLE)
    if unknown:
        parser.error(f"--live 에 쓸 수 없는 노드: {', '.join(sorted(unknown))} (가능: {', '.join(LIVE_CAPABLE)})")
    if live or args.supervisor == "llm" or args.judge:
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY 가 없습니다. .env 에 넣으세요.")
        if live & {"technical", "market", "domain"} and not os.getenv("TAVILY_API_KEY"):
            parser.error("기술 조사·시장·도메인 실제 실행에는 TAVILY_API_KEY 가 필요합니다.")
        print(f"실제 실행 노드: {', '.join(sorted(live, key=AGENTS.index))} (API 비용이 발생합니다)")

    fixture = load_fixture(args.fixture)
    initial = initial_state(as_of=args.as_of, rounds=args.rounds, max_steps=args.max_steps)
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
    control_nodes, control_modes = build_control(args.supervisor, live, Path("logs"), judge=args.judge)
    nodes.update(control_nodes)
    modes.update(control_modes)
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
    # 체크포인터: 같은 thread_id(run_id)로 중단된 실행을 이어갈 수 있다. LangSmith 가 켜져 있으면 metadata 로 trace_id 를 잇는다.
    config = {"run_name": "supervisor_run", "tags": ["supervisor", initial["domain"]],
              "metadata": {"trace_id": initial["trace_id"], "run_id": initial["run_id"], "live": sorted(live)},
              "configurable": {"thread_id": initial["run_id"]}}
    final, trace = run_graph(nodes, initial, on_step, checkpointer=make_checkpointer(), config=config)
    replayed = [a for a in AGENTS if modes[a].startswith(("임시", "fixture"))]
    note = f"{', '.join(replayed)} 는 합성 fixture 결과이며 실제 조사가 아닙니다." if replayed else None
    (folder / "summary.md").write_text(render_summary(modes, trace, final, note), encoding="utf-8")
    sections = final.get("report_sections") or {}
    # final_markdown 이 보고서 에이전트가 만든 완성본이다 (섹션을 다시 이어 붙이면 본문이 중복된다).
    (folder / "report.md").write_text(sections.get("final_markdown") or "\n\n".join(sections.values()), encoding="utf-8")
    (folder / "final_state.json").write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (folder / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "graph.mmd").write_text(build_graph(**nodes).get_graph().draw_mermaid(), encoding="utf-8")
    (folder / "decision_log.json").write_text(json.dumps(final.get("decision_log") or [], ensure_ascii=False, indent=2), encoding="utf-8")

    status = node_status(final)
    seconds = {t["node"]: t.get("seconds") for t in trace}
    print("실행 경로: START → " + " → ".join(steps_view(trace)) + " → END")
    print(f"supervisor: 종료 상태 {final.get('final_status')}, 턴 {final.get('step_count')}회, trace_id {final.get('trace_id')}")
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

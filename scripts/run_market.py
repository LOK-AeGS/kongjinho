"""단독 실행: python -m scripts.run_market --ask-key

DeepSeek-V2 MLA / ITME CXL-Hybrid 시장 평가를 단독으로 돌린다. --offline 없이 실행하면
OpenAI·Tavily API 비용이 발생한다.
"""

import argparse
import getpass
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.market.node import make_node  # noqa: E402
from agents.market.rag.fetch import fetch_document  # noqa: E402
from agents.market.rag.tools import stub_web_search, tavily_web_search  # noqa: E402
from agents.market.subgraph import MarketAgentDeps  # noqa: E402

SW = {"name": "DeepSeek-V2 (MLA)", "short_name": "MLA", "technology": "sw", "approach": "attention",
      "source_ids": ["arXiv:2405.04434"],
      "selection_reason": "MLA(Multi-head Latent Attention)로 KV cache를 저차원 잠재 벡터로 압축하는 모델 구조. "
                           "상위 기술군: LLM 추론(서빙)에서 KV cache를 줄이는 어텐션 구조·압축 기술. 자체 시장은 없고 "
                           "API·클라우드 제공, 서빙 프레임워크 지원 같은 채택 신호가 시장 근거임"}
HW = {"name": "ITME (CXL-Hybrid)", "short_name": "CXL-Hybrid", "technology": "hw", "approach": "memory",
      "source_ids": ["arXiv:2606.12556"],
      "selection_reason": "CXL-Hybrid 메모리로 KV cache를 담을 공간을 계층적으로 확장. "
                           "상위 기술군: CXL 메모리 확장(메모리 풀링·KV cache 오프로딩)"}


def render_markdown(findings: dict, evidence_store: dict) -> str:
    lines = ["# 시장 평가 결과", "", f"처리 상태: {findings['status']}", ""]
    names = {"sw": SW["name"], "hw": HW["name"]}
    for tid in ("sw", "hw"):
        lines.extend([f"## {names[tid]}", ""])
        for r in findings["records"]:
            if r["technology"] != tid:
                continue
            lines.append(f"### {r['criterion']} — basis={r['basis']} / assessment={r['assessment']} ({r['assessment_vocab']})")
            lines.append(f"- {r['findings']}")
            for eid in r["evidence_ids"]:
                e = evidence_store.get(eid)
                if e:
                    lines.append(f"  - 출처: [{e['title']}](<{e['url']}>) · 발행: {e['published_at'] or '미확인'}")
            if r["limitations"]:
                lines.append(f"  - 한계: {'; '.join(r['limitations'])}")
            lines.append("")
    lines.extend(["## Gaps", ""])
    for g in findings["gaps"]:
        lines.append(f"- [{g['technology']}] {g['criterion']}: {g['reason']}")
    lines.extend(["", "## 전체 한계", ""])
    lines.extend(f"- {x}" for x in findings["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-V2 MLA / ITME 시장 평가 단독 실행")
    parser.add_argument("--model", default=os.getenv("MARKET_LLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--strong-model", default=os.getenv("MARKET_STRONG_MODEL", "gpt-4.1"))
    parser.add_argument("--as-of", default=datetime.now().strftime("%Y-%m-%d"), help="조사 기준일 YYYY-MM-DD")
    parser.add_argument("--rounds", type=int, default=3, help="관점별 최대 검색 라운드")
    parser.add_argument("--page-budget", type=int, default=200, help="Pool B 본문 수집 누적 페이지 한도")
    parser.add_argument("--no-fetch-body", action="store_true", help="Pool B 본문 수집을 끄고 검색 스니펫만 쓴다")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/market"))
    parser.add_argument("--ask-key", action="store_true", help="OPENAI_API_KEY를 화면에 표시하지 않고 입력")
    parser.add_argument("--offline", action="store_true", help="API 키·네트워크 없이 스텁 검색·스텁 LLM으로 오프라인 실행")
    args = parser.parse_args()

    if not args.offline:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        if args.ask_key:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY를 .env/환경변수로 설정하거나 --ask-key를 쓰세요. (또는 --offline)")
        if not os.getenv("TAVILY_API_KEY"):
            parser.error("TAVILY_API_KEY를 .env/환경변수로 설정하세요. (또는 --offline)")

    folder = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=False)

    if args.offline:
        from agents.market.prompts import CompareOut

        class StubLLM:
            def with_structured_output(self, schema):
                raise RuntimeError("오프라인 모드: LLM 호출 없이 템플릿·스텁 검색 경로만 돈다")

        deps = MarketAgentDeps(llm=StubLLM(), web_search=stub_web_search, retriever=None)
        print("오프라인 스텁 실행: 실제 조사 결과가 아닙니다.", flush=True)
    else:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=args.model, temperature=0, timeout=60, max_retries=2)
        strong_llm = ChatOpenAI(model=args.strong_model, temperature=0, timeout=60, max_retries=2)
        # 캐시는 산출물(outputs/)이 아니라 data/에 둔다(규칙 16번). 재시도 사이에 같은 URL을 다시 받지 않는다.
        fetch_cache_dir = Path("data/fetch_cache/market")
        fetch_body = None if args.no_fetch_body else (lambda url: fetch_document(url, cache_dir=fetch_cache_dir))
        deps = MarketAgentDeps(
            llm=llm, strong_llm=strong_llm,
            web_search=tavily_web_search,
            retriever=None, fetch_body=fetch_body, page_budget=args.page_budget,
        )
        print(f"조사 시작: {args.model}/{args.strong_model}, 검색 라운드 최대 {args.rounds}회. API 비용이 발생합니다.", flush=True)

    state = {
        "selected_tech": {"sw": SW, "hw": HW},
        "domain": "datacenter_inference",
        "request": {"as_of": args.as_of, "language": "ko", "scope": "datacenter_inference", "max_search_rounds": args.rounds},
    }
    node = make_node(deps)
    out = node(state)
    findings = out["market_findings"]

    (folder / "market_findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "evidence_store.json").write_text(json.dumps(out["evidence_store"], ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "search_log.json").write_text(json.dumps(out["search_log_by_perspective"], ensure_ascii=False, indent=2), encoding="utf-8")
    banner = "# 오프라인 스텁 실행 결과 — 실제 조사 아님\n\n" if args.offline else ""
    (folder / "market_report.md").write_text(banner + render_markdown(findings, out["evidence_store"]), encoding="utf-8")

    print(f"상태: {findings['status']}")
    print(f"기준×판정 칸: {len(findings['records'])}개, 주장: {len(findings['claims'])}개, 근거: {len(out['evidence_store'])}개")
    for g in findings["gaps"]:
        print(f"gap: [{g['technology']}] {g['criterion']}: {g['reason']}", file=sys.stderr)
    for m in out["search_log_by_perspective"].get("market", []):
        print(f"search_log: {m.get('message', m)}", file=sys.stderr)
    print(f"결과 폴더: {folder.resolve()}")
    return 1 if findings["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

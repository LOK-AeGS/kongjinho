"""단독 실행: python -m scripts.run_domain

이해관계자 에이전트(scripts/run_stakeholder.py)와 대칭되는 진입점.
AppState(graph/state.py) 하나를 만들어 도메인 노드만 호출하고 결과를 저장한다.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from agents.domain import DomainAgentDeps, make_node
from agents.domain.tools.websearch import build_search_provider
from graph.state import create_initial_state


def build_request(as_of: str | None, rounds: int) -> dict:
    return {
        "as_of": as_of or datetime.now().strftime("%Y-%m-%d"),
        "language": "ko",
        "scope": "kv_cache_datacenter",
        "max_search_rounds": rounds,
    }


def build_selected_tech() -> dict:
    return {
        "sw": {
            "name": "DeepSeek-V2 MLA (Multi-head Latent Attention)",
            "short_name": "MLA",
            "technology": "sw",
            "approach": "저차원 잠재 압축으로 KV cache 크기 자체를 줄임",
            "source_ids": ["arxiv:2405.04434"],
            "selection_reason": "저차원 잠재 압축으로 KV cache 93.3% 감소",
        },
        "hw": {
            "name": "ITME (Inference Tiered Memory Expansion, CXL-Hybrid)",
            "short_name": "ITME",
            "technology": "hw",
            "approach": "CXL 기반 계층 메모리로 KV cache를 HBM 밖까지 확장",
            "source_ids": ["arxiv:2606.12556"],
            "selection_reason": "CXL-Hybrid 계층 메모리로 추론 처리량 1.80배 향상",
        },
    }


def render_markdown(findings: dict, evidence_store: dict, request: dict) -> str:
    lines = [
        "# 도메인 평가 결과 (데이터센터)", "",
        f"조사 기준일: {request['as_of']}",
        f"처리 상태: {findings['status']}", "",
    ]
    evidence_by_id = evidence_store
    for tech in ("sw", "hw"):
        lines.extend([f"## {tech.upper()}", ""])
        records = [r for r in findings["records"] if r["technology"] == tech]
        for record in records:
            value_suffix = f" ({record['value']})" if record["value"] else ""
            lines.append(f"- **{record['criterion']}** — {record['assessment']}{value_suffix}")
            lines.append(f"  - {record['findings']}")
            for lim in record["limitations"]:
                lines.append(f"  - 제약: {lim}")
            for eid in record["evidence_ids"]:
                ev = evidence_by_id.get(eid)
                if not ev:
                    continue
                lines.append(f"  - [{ev['title'][:60]}]({ev['url']}) · {ev['page_or_locator']}")
        lines.append("")
    lines.extend(["## 공백 (Gaps)", ""])
    for gap in findings["gaps"]:
        lines.append(f"- [{gap['technology']}] {gap['criterion']}: {gap['reason']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-V2 MLA / ITME 도메인(데이터센터) 평가")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--embedding", default="BAAI/bge-m3", help="빈 문자열이면 BM25만 사용")
    parser.add_argument("--as-of", help="조사 기준일 YYYY-MM-DD; 기본값 오늘")
    parser.add_argument("--rounds", type=int, default=2, help="최대 검색 라운드")
    parser.add_argument("--offline", action="store_true", help="검색 캐시를 재생하는 오프라인 모드로 강제")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/domain"))
    parser.add_argument("--ask-key", action="store_true", help="API 키를 화면에 표시하지 않고 입력")
    args = parser.parse_args()

    if args.ask_key:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
    if not args.offline and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY를 설정하거나 --ask-key 옵션을 사용하세요.")

    from langchain.chat_models import init_chat_model

    folder = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=False)

    # 검색·본문 캐시는 실행 간 공유해 오프라인 재생과 재현성을 유지한다.
    # 결과 JSON/마크다운만 실행별 타임스탬프 폴더에 분리한다.
    deps = DomainAgentDeps(
        llm=init_chat_model(args.model, model_provider="openai", temperature=0),
        search_provider=build_search_provider(
            Path("data/search_cache"), offline=(args.offline or None)
        ),
        embedding_model=args.embedding or None,
        fetch_cache_dir=Path("data/fetch_cache"),
    )
    mode = "오프라인 캐시 재생" if (args.offline or not os.getenv("TAVILY_API_KEY")) else "Tavily 실검색"
    print(f"조사 시작: {args.model} / {mode}. API 비용이 발생할 수 있습니다.", flush=True)

    state = create_initial_state(
        request=build_request(args.as_of, args.rounds),
        selected_tech=build_selected_tech(),
        corpus_manifest=[],
    )
    domain_node = make_node(deps)
    out = domain_node(state)

    findings = out["domain_findings"]
    (folder / "domain_findings.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "evidence_store.json").write_text(
        json.dumps(out["evidence_store"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "quality_report.json").write_text(
        json.dumps(out["quality_by_perspective"]["domain"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (folder / "run_meta.json").write_text(
        json.dumps(out["run_meta"]["domain"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "domain_report.md").write_text(
        render_markdown(findings, out["evidence_store"], state["request"]), encoding="utf-8"
    )

    print(f"상태: {findings['status']}")
    print(f"주장 {len(findings['claims'])}건 / 판정 {len(findings['records'])}건 / 근거 {len(out['evidence_store'])}건")
    for error in out["run_meta"]["domain"]["errors"]:
        print(f"오류: {error}", file=sys.stderr)
    print(f"결과 폴더: {folder.resolve()}")
    return 1 if findings["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""보고서 에이전트 단독 실행.

예시:
  python -m scripts.run_report --fixture
  python -m scripts.run_report --state state.json --output report.md --pdf-output report.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agents.report.state import ReportAgentDeps
from agents.report.subgraph import run_report
from agents.report.writer import load_report_environment


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "agents" / "report" / "fixtures" / "report_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="구조화된 upstream State로 Markdown·PDF 보고서를 생성합니다.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", action="store_true", help="테스트 complete fixture 사용")
    source.add_argument("--state", type=Path, help="확정 AppState JSON 파일")
    parser.add_argument("--output", type=Path, help="지정할 때만 Markdown 파일로 저장")
    parser.add_argument("--pdf-output", type=Path, help="지정할 때 PDF 파일로 저장")
    parser.add_argument(
        "--model",
        default=os.getenv("REPORT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        help="보고서 작성 LLM 모델",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="LLM을 호출하지 않고 기존 템플릿 writer로 재현 실행",
    )
    return parser.parse_args()


def load_state(args: argparse.Namespace) -> dict:
    path = FIXTURE if args.fixture else args.state
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["base"] if args.fixture else payload


def main() -> int:
    args = parse_args()
    if not args.deterministic and not load_report_environment():
        raise SystemExit(
            "저장소 루트의 .env에 OPENAI_API_KEY를 설정하세요. "
            "API 없이 재현하려면 --deterministic을 지정하세요."
        )
    result = run_report(
        load_state(args),
        ReportAgentDeps(
            generation_mode="deterministic" if args.deterministic else "llm",
            model=args.model,
            pdf_output_path=args.pdf_output,
        ),
    )
    report = result["report"]
    if args.output:
        args.output.write_text(report["markdown"], encoding="utf-8")
        print(f"저장: {args.output}")
    else:
        print(report["markdown"])
    if report.get("pdf_path"):
        print(f"PDF 저장: {report['pdf_path']}")
    generation = result["generation"]
    print(f"생성 방식: {generation['mode']} / 모델: {generation.get('model') or '-'}")
    print(f"상태: {report['quality_status']}")
    if report["completion"]["errors"]:
        print("미해결:")
        for value in report["completion"]["errors"]:
            print(f"- {value}")
    return 0 if report["quality_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

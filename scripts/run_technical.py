"""고정 입력 기술조사 실행: python -m scripts.run_technical"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agents.technical import make_node
from agents.technical.config import fixed_input


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V2 MLA와 ITME 기술조사")
    parser.add_argument("--output", type=Path, default=Path("outputs/technical/latest.json"))
    args = parser.parse_args()
    load_dotenv()
    missing = [name for name in ("OPENAI_API_KEY", "TAVILY_API_KEY") if not os.getenv(name)]
    if missing:
        parser.error(".env 또는 환경변수에 " + ", ".join(missing) + "를 설정하세요.")
    request, selected_tech = fixed_input()
    result = make_node()({"request": request, "selected_tech": selected_tech})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    findings = result["technical_findings"]
    print(f"상태: {findings['status']}")
    print(f"레코드: {len(findings['records'])}개, 근거: {len(result['evidence_store'])}개")
    print(f"결과: {args.output.resolve()}")


if __name__ == "__main__":
    main()

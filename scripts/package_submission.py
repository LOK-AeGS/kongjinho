"""제출용 압축 파일을 만든다: Agent_{캠퍼스}_{X반}_{이름1+이름2+...}.zip

안에 들어가는 것 (가이드의 제출물 3종: Git 링크 + 트레이싱 PNG + 평가 보고서 PDF)
  - GIT_LINK.txt        저장소·브랜치 주소
  - tracing-*.png       LangSmith 캡처 (직접 찍어서 --tracing-dir 에 둔다. 긴 경로는 tracing-1.png, tracing-2.png …)
  - 보고서 PDF          --report 로 지정 (python main.py --live all 이 outputs/graph/<실행시각>/report.pdf 로 만든다)

  python -m scripts.package_submission --report outputs/graph/<실행시각>/report.pdf --tracing-dir captures
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

NAMES = "강유성+이효은+지승환+이산+안균승+이동영"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="제출 zip 만들기")
    parser.add_argument("--campus", default="판교")
    parser.add_argument("--class-no", default="6반")
    parser.add_argument("--names", default=NAMES, help="이름을 + 로 이은 문자열")
    parser.add_argument("--report", type=Path, required=True, help="평가 보고서 PDF")
    parser.add_argument("--tracing-dir", type=Path, required=True, help="tracing-*.png 가 있는 폴더")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    pngs = sorted(args.tracing_dir.glob("tracing-*.png"))
    problems = []
    if not args.report.is_file():
        problems.append(f"보고서 PDF 없음: {args.report}")
    if not pngs:
        problems.append(f"{args.tracing_dir} 에 tracing-*.png 가 없음 (LangSmith 캡처를 먼저 저장하세요)")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    remote = git("remote", "get-url", "origin").removesuffix(".git")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "--short", "HEAD")
    target = args.output_dir / f"Agent_{args.campus}_{args.class_no}_{args.names}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("GIT_LINK.txt", f"{remote}/tree/{branch}\ncommit: {commit}\n")
        for png in pngs:
            archive.write(png, png.name)
        archive.write(args.report, args.report.name)
    print(f"저장: {target} (PNG {len(pngs)}장, PDF 1개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

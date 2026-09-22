"""평가 종합 에이전트 단독 실행.

  python -m scripts.run_synthesis                          # 합성 fixture + 템플릿 서술 (API 키 불필요)
  python -m scripts.run_synthesis --writer openai          # 실제 LLM 서술 (OPENAI_API_KEY 필요, 비용 발생)
  python -m scripts.run_synthesis --input <AppState.json>  # 다른 에이전트 결과가 담긴 AppState JSON
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from datetime import datetime
from pathlib import Path

from agents.synthesis import make_node

DEFAULT_INPUT = Path("tests/agents/synthesis/fixtures/appstate_sample.json")


def load_env(path: Path = Path(".env")) -> None:
    """.env 의 OPENAI_API_KEY 를 읽는다. 이미 환경변수가 있으면 그대로 둔다."""
    if os.getenv("OPENAI_API_KEY") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def render_markdown(state: dict, result: dict) -> str:
    names = {t: (state.get("selected_tech") or {}).get(t, {}).get("short_name", t) for t in ("sw", "hw")}
    meta, q = result["meta"], result["meta"].get("quality", {})
    lines = [
        "# 평가 종합 결과",
        "",
        f"- 기준일: {result['as_of']}  |  SW: {names['sw']}  |  HW: {names['hw']}",
        f"- 상태: **{result['status']}**  |  서술: {meta['writer']}{' / ' + meta['model'] if meta.get('model') else ''}"
        f"  |  중립성 검사: {q.get('status')} (검사 {q.get('checked')}, 유지 {q.get('kept')}, 재생성 {q.get('revised')}, 제거 {q.get('dropped')})",
        f"- 입력 해시: `{result['input_hash'][:16]}`  |  규칙: {meta['rules_version']}  |  프롬프트: {meta['prompt_version']}",
    ]
    if state.get("_note"):
        lines += ["", f"> ⚠️ {state['_note']}"]

    lines += ["", "## 요약 주장", ""]
    lines += [f"- **{c['claim_id']}** ({c['technology']}) {c['text']}  \n  근거: {', '.join(c['evidence_ids'])}"
              for c in result["summary_claims"]] or ["(없음)"]

    lines += ["", "## 비교 매트릭스", "", "| 기술 | 관점 | 기준 | basis | 판단 | 증거 수준 | 범위 | 근거 수 |", "|---|---|---|---|---|---|---|---|"]
    for c in result["matrix"]:
        value = f" {c['value']}" if c.get("value") else ""
        lines.append(f"| {c['technology']} | {c['perspective']} | {c['criterion']} | {c['basis']} | {c['assessment']}{value} "
                     f"| {c['evidence_level']} | {c['scope']} | {c['n_evidence']} |")

    lines += ["", "## 관점 간 관계", "", "| ID | 종류 | 규칙 | 기술 | 관련 record | 설명 | 해소 |", "|---|---|---|---|---|---|---|"]
    for f in result["cross_findings"]:
        lines.append(f"| {f['id']} | {f['kind']} | {f['rule_id'] or ''} | {f['technology']} | {'<br>'.join(f['record_refs'])} "
                     f"| {f['explanation'] or ''} | {f['resolution']} |")

    lines += ["", "## SW/HW 대조표", "", "| 기준 | SW | HW | 관계 |", "|---|---|---|---|"]
    lines += [f"| {r['criterion']} | {r['sw'] or '—'} | {r['hw'] or '—'} | {r['relation']} |" for r in result["contrast_table"]]

    lines += ["", "## 판단 보류·공백", ""]
    lines += [f"- [{g['perspective']}/{g['technology']}] {g['criterion']}: {g['reason']}" for g in result["gaps"]] or ["(없음)"]
    lines += ["", "## 재실행 요청", ""]
    lines += [f"- {r['perspective']}: {r['reason']}" for r in result["retry_requests"]] or ["(없음)"]
    lines += ["", "## 한계", ""]
    lines += [f"- {x}" for x in result["limitations"]] or ["(없음)"]
    imb = result["imbalance"]
    if imb:
        lines += [f"- 근거 수: sw {imb['evidence_counts']['sw']} / hw {imb['evidence_counts']['hw']}"
                  f"{' (불균형)' if imb.get('flagged') else ''}"]
    lines += ["", "## 제거된 문장", ""]
    lines += [f"- {d['text']}  \n  사유: {'; '.join(d['violations'])}" for d in result["dropped_sentences"]] or ["(없음)"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m scripts.run_synthesis", description="평가 종합 에이전트 단독 실행")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="AppState JSON (기본: 합성 fixture)")
    parser.add_argument("--writer", choices=("template", "openai"), default="template",
                        help="template: LLM 없이 규칙 문장 / openai: 실제 LLM 서술")
    parser.add_argument("--model", default=None, help="openai 모델 ID (기본: 환경변수 SYNTHESIS_MODEL 또는 gpt-5.5)")
    parser.add_argument("--ask-key", action="store_true", help="OpenAI API 키를 화면에 표시하지 않고 입력")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthesis"))
    args = parser.parse_args()

    state = json.loads(args.input.read_text(encoding="utf-8"))
    writer = None
    if args.writer == "openai":
        if args.ask_key:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
        load_env()
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY 가 없습니다. .env 에 넣거나 --ask-key 를 쓰세요.")
        from agents.synthesis.writer import OpenAIWriter
        writer = OpenAIWriter(model=args.model)
        print(f"서술: {writer.model} (API 비용이 발생합니다)")

    result = make_node(writer)(state)["synthesis"]

    folder = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "synthesis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "synthesis_report.md").write_text(render_markdown(state, result), encoding="utf-8")

    counts, q = result["meta"]["counts"], result["meta"]["quality"]
    if state.get("_note"):
        print("합성 fixture 입력: 실제 조사 결과로 간주하지 마세요.")
    print(f"상태: {result['status']}")
    print(f"매트릭스 {counts['matrix_cells']}칸 / 상충 {counts['conflicts']} / 공유 근거 {counts['shared_evidence']}"
          f" / 일치 {counts['agreements']} / 보완 {counts['complements']}")
    print(f"요약 주장 {q.get('kept', 0)}개 (검사 {q.get('checked', 0)}, 재생성 {q.get('revised', 0)}, 제거 {q.get('dropped', 0)})"
          f" / 중립성 검사: {q.get('status')}")
    print(f"결과 폴더: {folder.resolve()}")


if __name__ == "__main__":
    main()

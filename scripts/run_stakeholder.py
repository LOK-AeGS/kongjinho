"""단독 실행: python -m scripts.run_stakeholder --ask-key"""
import argparse
import getpass
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from agents.stakeholder.node import team_update
from agents.stakeholder.subgraph import GROUP_LABELS, default_request, run_stakeholder
from agents.stakeholder.backend import OpenAIBackend


def render_markdown(result: dict, request: dict) -> str:
    lines = ["# 이해관계자 평가 결과", "", f"조사 기준일: {request['as_of_date']}",
             "평가 도메인: 데이터센터",
             f"처리 상태: {result['completion']['status']}", "",
             "원문 snapshot의 hash·locator·인용문을 검사한 결과입니다. 발언 대상과 요약의 의미적 일치는 별도 검토가 필요합니다.", ""]
    claims = {c["claim_id"]: c for c in result["claims"]}
    evidence = {e["evidence_id"]: e for e in result["evidence"]}
    scope_labels = {"selected_technology": "선정 기술 직접 반응", "technology_family": "기술 계열 배경 의견", "other": "기타 대상 의견"}
    for side in ("sw", "hw"):
        lines.extend([f"## {request[side]['name']}", ""])
        for scope, label in scope_labels.items():
            positions = [p for p in result["positions"] if p["technology_id"] == side and p["target_scope"] == scope]
            if not positions:
                if scope == "selected_technology":
                    lines.extend(["선정 기술 자체에 대한 직접 반응을 확인하지 못했습니다.", ""])
                continue
            lines.extend([f"### {label}", ""])
            for p in positions:
                lines.extend([f"- **{p['speaker']}** ({GROUP_LABELS[p['group']]}, {p['stance']}, {p['evidence_stance']}) — 평가 대상: {p['target_name']}"])
                for cid in p["claim_ids"]:
                    c = claims[cid]
                    lines.append(f"  - {c['statement']}")
                    for eid in c["evidence_ids"]:
                        e = evidence[eid]
                        title = e['title'].replace('[', '').replace(']', '')
                        lines.append(f"  - 출처: [{title}](<{e['url']}>) · 발행일: {e['published_date'] or '미확인'}")
                    if c["uncertainty"]:
                        lines.append(f"  - 한계: {c['uncertainty']}")
                if p["bias_notes"]:
                    lines.append("  - 편향·이해관계: " + "; ".join(p["bias_notes"]))
                lines.append("")
    lines.extend(["## 미확인 사항 및 오류", ""])
    lines.extend(f"- {x}" for x in result["completion"]["gaps"] + result["completion"]["errors"])
    lines.extend(["", "## 탐색했지만 찾지 못한 반응", ""])
    for item in result.get("search_outcomes", []):
        if item["status"] == "not_found":
            lines.append(f"- {item['technology_id']} / {GROUP_LABELS[item['group']]} / {item['stance']}: not_found (질의: {', '.join(item['query_ids'])})")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V2 MLA / ITME 이해관계자 평가")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--as-of", help="조사 기준일 YYYY-MM-DD; 기본값 오늘")
    parser.add_argument("--rounds", type=int, default=2, help="최대 검색 라운드 1~3")
    parser.add_argument("--revisions", type=int, default=1, help="최대 구조화 수정 횟수 0~2")
    parser.add_argument("--max-queries", type=int, default=8, help="전체 실행의 검색 API 요청 한도 1~24")
    parser.add_argument("--allowed-domain", action="append", default=[], help="검색·원문 허용 도메인; 반복 지정 가능")
    parser.add_argument("--offline-fixture", type=Path, help="검증용 fixture JSON 재생. 네트워크/API 키 불필요")
    parser.add_argument("--technical-input", type=Path, help="선택: 기술 조사 결과 JSON 또는 전체 팀 State")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stakeholder"))
    parser.add_argument("--ask-key", action="store_true", help="API 키를 화면에 표시하지 않고 입력")
    args = parser.parse_args()
    if args.ask_key:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
    if not args.offline_fixture and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY를 설정하거나 --ask-key 옵션을 사용하세요.")
    request = default_request(args.as_of)
    request.update(max_search_rounds=args.rounds, max_revision_rounds=args.revisions, max_queries=args.max_queries)
    technical = None
    if args.technical_input:
        technical = json.loads(args.technical_input.read_text(encoding="utf-8"))
        if "technical_findings" in technical:
            technical = technical["technical_findings"]
    folder = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder.mkdir(parents=True, exist_ok=False)
    if args.offline_fixture:
        from agents.stakeholder.offline import FixtureBackend
        backend = FixtureBackend(args.offline_fixture)
        print("오프라인 fixture 재생: 실검색 결과로 간주하지 마세요.", flush=True)
    else:
        backend = OpenAIBackend(args.model, cache_dir=folder / "evidence_cache", allowed_domains=args.allowed_domain)
        print(f"조사 시작: {args.model}, 검색 API 최대 {args.max_queries}회. API 비용이 발생합니다.", flush=True)
    final = run_stakeholder(request, technical, backend)
    final['run_metadata'] = {'model': args.model, 'provider': 'offline_fixture' if args.offline_fixture else 'openai.responses',
                            'prompt_version': 'stakeholder-v0.3.1', 'temperature': None, 'seed': None,
                            'sampling_note': 'temperature/seed 미지정; API 기본값 사용',
                            'max_output_tokens_search': 3000, 'max_output_tokens_extract': 10000}
    (folder / "stakeholder_findings.json").write_text(json.dumps(final["result"], ensure_ascii=False, indent=2), encoding="utf-8")
    update = team_update(final)
    (folder / "team_state_update.json").write_text(json.dumps(update, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "research_trace.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    banner = "# 오프라인 검증용 결과 — 실제 조사 아님\n\n" if args.offline_fixture else ""
    (folder / "stakeholder_report.md").write_text(banner + render_markdown(final["result"], request), encoding="utf-8")
    print(f"상태: {final['result']['completion']['status']}")
    print(f"확인된 입장: {len(final['result']['positions'])}개")
    for error in final["result"]["completion"]["errors"]:
        print(f"오류: {error}", file=sys.stderr)
    print(f"결과 폴더: {folder.resolve()}")
    return 1 if final["result"]["completion"]["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

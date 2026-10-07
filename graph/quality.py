"""보고서 품질 평가 노드 (hybrid: 규칙 검사 + 선택적 LLM Judge).

보고서가 만들어진 뒤 Groundedness / 중립성 / 편향 통제 / 관점 커버리지를 검사한다.
미달이면 verdict.passed=False 와 failed_checks, 다시 돌려야 할 대상(target_perspectives)을 State 에 남기고
Supervisor 가 그 대상에게 재작업을 요청해 보고서를 다시 만든다 (Loop). 규칙 검사는 결정적이다.
"""

from __future__ import annotations

import re
from collections import Counter

from agents.report.validators import PROHIBITED_COMPARISON
from graph.sufficiency import MIN_SOURCES, PERSPECTIVES, perspective_evidence, source_key

CITATION = re.compile(r"〔근거:\s*([^〕]+)〕")
EXTRA_PROHIBITED = ("최고", "1위", "should adopt", "recommend", "best choice", "clear winner")
NEGATION = ("않", "아니", "없", "금지", "삼간", "하지 말", "not ", "no ", "never")
MAX_SINGLE_SOURCE_SHARE = 0.6
BIAS_PERSPECTIVES = ("market", "stakeholder", "domain")


def report_text(state: dict) -> str:
    sections = state.get("report_sections") or {}
    text = sections.get("final_markdown") or "\n\n".join(str(v) for v in sections.values())
    head, _, _ = text.partition("# REFERENCE")
    return head


def cited_ids(text: str) -> list[str]:
    ids = []
    for group in CITATION.findall(text):
        ids.extend(part.strip() for part in re.split(r"[,;]", group) if part.strip())
    return ids


def check_groundedness(state: dict, text: str) -> tuple[list[str], list[str]]:
    store = state.get("evidence_store") or {}
    cited = cited_ids(text)
    if not cited:
        return ["보고서에 〔근거: …〕 인용이 하나도 없음"], ["report"]
    unknown = sorted({eid for eid in cited if eid not in store})
    return ([f"evidence_store 에 없는 인용 ID {len(unknown)}건: {', '.join(unknown[:3])}"] if unknown else []), ["report"]


def check_neutrality(text: str) -> tuple[list[str], list[str]]:
    issues = []
    for sentence in re.split(r"(?<=[.!?。\n])\s*", text):
        low = sentence.lower()
        hit = [w for w in (*PROHIBITED_COMPARISON, *EXTRA_PROHIBITED) if w in low]
        if hit and not any(n in low for n in NEGATION):
            issues.append(f"{'/'.join(hit)}: {sentence.strip()[:80]}")
    return issues[:5], ["report"]


def check_bias(state: dict) -> tuple[list[str], list[str]]:
    issues, targets = [], []
    for p in BIAS_PERSPECTIVES:
        evidence = perspective_evidence(state, p)
        if not evidence:
            continue
        counts = Counter(source_key(ev) for ev in evidence)
        if len(counts) < MIN_SOURCES:
            issues.append(f"{p}: 출처가 {len(counts)}곳뿐")
            targets.append(p)
        elif len(evidence) >= 3 and counts.most_common(1)[0][1] / len(evidence) > MAX_SINGLE_SOURCE_SHARE:
            issues.append(f"{p}: 단일 출처 {counts.most_common(1)[0][0]} 비중 {counts.most_common(1)[0][1]}/{len(evidence)}")
            targets.append(p)
    return issues, targets


def check_coverage(state: dict, text: str) -> tuple[list[str], list[str]]:
    store = state.get("evidence_store") or {}
    cited_perspectives = {store[eid]["perspective"] for eid in cited_ids(text) if eid in store}
    issues, targets = [], []
    for p in PERSPECTIVES:
        if p in cited_perspectives:
            continue
        issues.append(f"{p} 관점이 보고서에 인용되지 않음")
        targets.append(p if not perspective_evidence(state, p) else "report")
    return issues, targets


def check_length(state: dict) -> tuple[list[str], list[str]]:
    """제출 보고서는 최대 10장. PDF 를 만든 실행에서만 검사한다 (report 노드가 run_meta 에 페이지 수를 남김)."""
    layout = ((state.get("run_meta") or {}).get("report") or {}).get("pdf_layout")
    if layout and not layout.get("within_limit", True):
        return [f"PDF {layout['pages']}쪽 > 최대 {layout['max_pages']}쪽 (표지 생략·글자 축소 후에도 초과)"], ["report"]
    return [], ["report"]


def make_quality_node(judge=None):
    """judge(state, text) -> list[(check, reason)] 를 주면 LLM Judge 가 실패 항목만 추가할 수 있다."""

    def quality_node(state: dict) -> dict:
        text = report_text(state)
        iteration = int(state.get("quality_iterations") or 0) + 1
        details, targets = {}, []
        for name, (issues, target) in {
            "groundedness": check_groundedness(state, text),
            "neutrality": check_neutrality(text),
            "bias": check_bias(state),
            "coverage": check_coverage(state, text),
            "length": check_length(state),
        }.items():
            if issues:
                details[name] = issues
                targets.extend(target)
        if judge is not None:
            for check, reason in judge(state, text):
                details.setdefault(check, []).append(f"judge: {reason}")
                targets.append("report")
        verdict = {"passed": not details, "iteration": iteration, "failed_checks": sorted(details), "details": details,
                   "target_perspectives": list(dict.fromkeys(targets)) if details else []}
        status = "ok" if verdict["passed"] else "failed"
        return {"quality_verdict": verdict, "quality_iterations": iteration,
                "node_status": {"quality": {"status": status, "error": None, "attempts": iteration}}}

    return quality_node


def make_llm_judge(client, model: str):
    """내용 기준 2차 검사. 규칙 검사가 못 보는 편향·우열 암시를 LLM 이 실패 항목으로만 추가한다."""
    from pydantic import BaseModel

    class Finding(BaseModel):
        check: str
        reason: str

    class Judgement(BaseModel):
        failures: list[Finding]

    def judge(state: dict, text: str):
        response = client.responses.parse(
            model=model, store=False, text_format=Judgement,
            instructions=("기술 평가 보고서를 검사한다. 특정 기술 추천·우열 판정(neutrality), 한쪽 근거 편중(bias), "
                          "근거 없는 주장(groundedness)이 있는 경우만 failures 에 check(neutrality|bias|groundedness)와 "
                          "한 줄 reason 으로 적는다. 문제가 없으면 빈 목록."),
            input=text[:30000])
        if response.status != "completed" or response.output_parsed is None:
            return []
        allowed = {"neutrality", "bias", "groundedness"}
        return [(f.check, f.reason) for f in response.output_parsed.failures if f.check in allowed]

    return judge

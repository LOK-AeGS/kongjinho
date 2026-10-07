"""API 호출 없이 재현 가능한 보고서 품질 검사."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from agents.report.references import source_identity
from graph.groundedness import (
    CITATION,
    GROUNDEDNESS_EXCLUDED_SECTIONS,
    META_PHRASES,
    MIN_CITATION_RATIO,
    _body,
    citation_ids,
    factual_body_lines,
)
from graph.metrics import MEASUREMENT
from graph.rules import PROHIBITED_COMPARISON, VERIFICATION_ADVICE_PATTERN


# 보고서 검증과 같은 목록(graph/rules.py).
PROHIBITED = PROHIBITED_COMPARISON
SECTIONS = {
    "technical": "4.1 기술 성숙도(TRL)",
    "market": "4.2 시장성",
    "stakeholder": "4.3 이해관계자",
    "domain": "4.4 도메인 적용",
}

# 13개 live report 보정: 정상 section 최대 집중도는 0.64였다. 0.6 초과는 관찰 경고,
# 병리 사례·bias_onesided의 1.0에 가까운 집중만 실패로 분리한다.
SECTION_SOURCE_WARNING_THRESHOLD = 0.6
SECTION_SOURCE_FAILURE_THRESHOLD = 0.8


def groundedness(
    markdown: str,
    evidence_store: dict,
    report_violations: list[str] | None = None,
) -> dict:
    factual = factual_body_lines(markdown)
    cited = [line for line in factual if citation_ids(line)]
    uncited_measurements = [
        line for line in factual if MEASUREMENT.search(line) and not citation_ids(line)
    ]
    all_ids = citation_ids(_body(markdown))
    unknown = sorted(set(all_ids) - set(evidence_store))
    ratio = len(cited) / len(factual) if factual else 0.0
    details = [f"사실 불릿·표 행 인용률 {ratio:.2f} ({len(cited)}/{len(factual)})"]
    if unknown:
        details.append("존재하지 않는 근거 ID: " + ", ".join(unknown))
    details.extend("측정값 무인용: " + line for line in uncited_measurements)
    details.extend(str(item) for item in (report_violations or []) if str(item).strip())
    passed = ratio >= MIN_CITATION_RATIO and not unknown and not uncited_measurements and not report_violations
    return {"passed": passed, "score": ratio if not unknown else 0.0, "details": details}


# "원본 확인 권장"처럼 검증 절차를 권하는 표현은 기술 추천이 아니다(live 1차 오탐).
VERIFICATION_ADVICE = re.compile(VERIFICATION_ADVICE_PATTERN)


def neutrality(markdown: str) -> dict:
    body = VERIFICATION_ADVICE.sub("", _body(markdown))
    found = sorted({word for word in PROHIBITED if word in body})
    return {
        "passed": not found,
        "score": 1.0 if not found else 0.0,
        "details": [] if not found else ["금지 표현: " + ", ".join(found)],
    }


def _normalized_evidence(raw: dict) -> dict:
    return {
        **raw,
        "document_id": raw.get("document_id") or raw.get("doc_id"),
        "author_or_organization": raw.get("author_or_organization") or raw.get("author_or_org"),
        "published_date": raw.get("published_date") or raw.get("published_at"),
    }


def bias_control(markdown: str, evidence_store: dict, findings: dict[str, dict | None]) -> dict:
    ids = [item for item in citation_ids(_body(markdown)) if item in evidence_store]
    sources = Counter(source_identity(_normalized_evidence(evidence_store[item])) for item in ids)
    concentration = max(sources.values(), default=0) / len(ids) if ids else 1.0

    technologies: dict[str, set[str]] = defaultdict(set)
    for result in findings.values():
        for item in (result or {}).get("records", []) + (result or {}).get("claims", []):
            tech = item.get("technology")
            for evidence_id in item.get("evidence_ids", []):
                if tech in {"sw", "hw"}:
                    technologies[evidence_id].add(tech)
                elif tech == "both":
                    technologies[evidence_id].update(("sw", "hw"))
    counts = Counter(tech for evidence_id in ids for tech in technologies.get(evidence_id, ()))
    low, high = sorted((counts.get("sw", 0), counts.get("hw", 0)))
    ratio = high / low if low else (float("inf") if high else 1.0)
    details = [f"최대 출처 집중도 {concentration:.2f}", f"SW/HW claim 인용 수 {counts.get('sw', 0)}/{counts.get('hw', 0)}"]
    section_failures = []
    for perspective, title in SECTIONS.items():
        body = _section(markdown, title)
        cited_lines = [
            line.strip() for line in body.splitlines()
            if line.strip().startswith(("- ", "* ", "|")) and citation_ids(line)
        ]
        if len(cited_lines) < 3:
            continue
        section_ids = [
            evidence_id
            for line in cited_lines
            for evidence_id in citation_ids(line)
            if evidence_id in evidence_store
        ]
        section_sources = Counter(
            source_identity(_normalized_evidence(evidence_store[evidence_id]))
            for evidence_id in section_ids
        )
        share = max(section_sources.values(), default=0) / len(section_ids) if section_ids else 0.0
        if share > SECTION_SOURCE_FAILURE_THRESHOLD:
            section_failures.append(f"{title}: 단일 출처 편중 {share:.2f}")
        elif share > SECTION_SOURCE_WARNING_THRESHOLD:
            details.append(f"경고: {title}: 단일 출처 편중 {share:.2f}")
        section_tech = {
            tech for evidence_id in section_ids for tech in technologies.get(evidence_id, ())
        }
        if section_tech in ({"sw"}, {"hw"}):
            other = ({"sw", "hw"} - section_tech).pop()
            # 다른 기술의 근거가 없다는 사실이 공백으로 명시돼 있으면 편향이 아니라 근거 부재다
            # (live 1차: ITME는 공개 이해관계자 반응이 없어 4.3이 SW만 다뤘고 gaps에 기록돼 있었다).
            if _absence_documented(body, (findings.get(perspective) or {}), other):
                details.append(f"경고: {title}: 한 기술만 다룸({other.upper()} 근거 부재가 공백으로 명시됨)")
            else:
                section_failures.append(f"{title}: 한 기술만 다룸")
    details.extend(section_failures)
    for tech in ("sw", "hw"):
        counter = any(
            evidence_store.get(evidence_id, {}).get("stance") == "counter"
            for evidence_id, mapped in technologies.items() if tech in mapped
        )
        if not counter:
            mentioned = "counter" in markdown.casefold() or "반대" in markdown or "반론" in markdown
            details.append(f"경고: {tech} counter stance 근거 없음" + ("(한계점 언급)" if mentioned else "(한계점 미언급)"))
    passed = bool(ids) and concentration <= 0.4 and ratio <= 2.5 and not section_failures
    source_score = min(1.0, 0.4 / concentration) if concentration else 1.0
    balance_score = min(1.0, 2.5 / ratio) if ratio not in {0, float("inf")} else (1.0 if ratio == 0 else 0.0)
    section_score = 0.0 if section_failures else 1.0
    return {"passed": passed, "score": min(source_score, balance_score, section_score), "details": details}


ABSENCE_MARKERS = ("확인하지 못", "찾지 못", "판단 보류", "근거 부족", "미확인", "not_found", "not_assessed", "silent")


def _absence_documented(section_body: str, findings: dict, technology: str) -> bool:
    """해당 기술의 근거 부재가 관점 결과 gaps나 섹션 본문에 명시됐는지 본다."""
    if any(gap.get("technology") in {technology, "both"} for gap in findings.get("gaps") or []):
        return True
    return any(marker in section_body for marker in ABSENCE_MARKERS)


def _section(markdown: str, title: str) -> str:
    match = re.search(rf"^##\s+{re.escape(title)}\s*$", markdown, re.MULTILINE)
    if not match:
        return ""
    tail = markdown[match.end():]
    end = re.search(r"^#{1,2}\s+", tail, re.MULTILINE)
    return tail[:end.start()] if end else tail


def coverage(markdown: str) -> dict:
    """4개 관점(4.1~4.4)이 각각 인용 근거를 가진 실질 내용으로 다뤄졌는지 본다.

    정책: 근거 부재를 공백으로 명시했더라도 인용이 하나도 없는 관점은 '포괄'로 보지 않는다.
    과제의 관점 커버리지는 다관점 평가가 실제로 이뤄졌는지를 묻는 항목이라, 근거 0건을 통과시키면
    취지에 어긋난다(live 4차: stakeholder 근거 0건 → 불합격 유지, 재작업 예산 소진 후 needs_review).
    """
    missing = []
    for perspective, title in SECTIONS.items():
        body = _section(markdown, title)
        content_lines = [line.strip() for line in body.splitlines() if line.strip()]
        # 판정과 인용이 있는 줄은 "미확인" 같은 한계 문구를 함께 담아도 실질 내용으로 본다
        # (live 3차: TRL 줄이 '운영 사례 미확인' 한계를 포함해 technical 누락으로 오판됨).
        only_unconfirmed = bool(content_lines) and all(
            any(phrase in line for phrase in META_PHRASES) and not citation_ids(line)
            for line in content_lines
        )
        meaningful = bool(content_lines) and not only_unconfirmed
        if not meaningful or not citation_ids(body):
            missing.append(perspective)
    return {
        "passed": not missing,
        "score": (4 - len(missing)) / 4,
        "details": [] if not missing else ["누락 관점: " + ", ".join(missing)],
        "missing_perspectives": missing,
    }


def run_checks(
    markdown: str,
    evidence_store: dict,
    findings: dict[str, dict | None],
    report_violations: list[str] | None = None,
) -> dict[str, dict]:
    return {
        "groundedness": groundedness(markdown, evidence_store, report_violations),
        "neutrality": neutrality(markdown),
        "bias_control": bias_control(markdown, evidence_store, findings),
        "coverage": coverage(markdown),
    }

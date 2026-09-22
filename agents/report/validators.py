"""보고서의 결정적 품질 검사.

구조·참조·수치·금지 표현은 LLM 판정에 맡기지 않는다. 검사는 섹션 단위로 실행되어
수정 대상도 해당 섹션으로 한정된다.
"""

from __future__ import annotations

import re

from agents.report.references import source_identity
from agents.report.state import SECTION_ORDER, SectionDraft, SectionId, ValidationIssue


REQUIRED_HEADINGS = (
    "# SUMMARY",
    "# 1. 분석 배경",
    "# 2. 기술 선정",
    "# 3. 기술 개요",
    "# 4. 관점별 평가",
    "## 4.1 기술 성숙도(TRL)",
    "## 4.2 시장성",
    "## 4.3 이해관계자",
    "## 4.4 도메인 적용",
    "# 5. 시사점",
    "## 5.1 비교 매트릭스",
    "## 5.2 적용 조건 대조표",
    "## 5.3 관점 간 상충 지점",
    "## 5.4 공유 근거 및 보완 관계",
    "## 5.5 남은 확인 과제",
    "# 6. 한계점",
    "# REFERENCE",
)

PROHIBITED_COMPARISON = (
    "승자",
    "추천",
    "압도",
    "월등",
    "우위",
    "열위",
    "더 낫다",
    "최고의 기술",
    "최선의 기술",
)

_MEASUREMENT = re.compile(
    r"(?<![\w.-])\d+(?:\.\d+)?\s*(?:%|×|x(?=\s|$)|배|GB/s|GB|TB|MB|ms|μs|초|달러|원)",
    re.IGNORECASE,
)
_CITATION = re.compile(r"〔근거:\s*([^〕]+)〕")


def issue(
    code: str, message: str, section_id: SectionId | None, blocking: bool = True
) -> ValidationIssue:
    return {
        "code": code,
        "message": message,
        "section_id": section_id,
        "blocking": blocking,
    }


def extract_measurements(text: str) -> set[str]:
    return {re.sub(r"\s+", "", value).lower() for value in _MEASUREMENT.findall(text or "")}


def extract_citations(text: str) -> list[str]:
    found: list[str] = []
    for group in _CITATION.findall(text or ""):
        for value in group.split(","):
            evidence_id = value.strip()
            if evidence_id and evidence_id not in found:
                found.append(evidence_id)
    return found


def validate_input(context: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    store = context["evidence_store"]
    for claim in context["claims"].values():
        unknown = [eid for eid in claim.get("evidence_ids", []) if eid not in store]
        if unknown:
            issues.append(
                issue(
                    "unknown_evidence",
                    f"{claim['claim_id']}가 없는 evidence를 참조함: {', '.join(unknown)}",
                    None,
                )
            )
        if claim.get("basis") in {"direct", "direct_evidence"} and not claim.get("evidence_ids"):
            issues.append(
                issue(
                    "direct_claim_without_evidence",
                    f"{claim['claim_id']}의 직접 근거 ID가 비어 있음",
                    None,
                )
            )
    synthesis = context.get("synthesis") or {}
    for field in ("matrix", "comparison_matrix", "cross_findings", "relations", "contrast_table"):
        for index, record in enumerate(synthesis.get(field, []), 1):
            unknown_claims = [cid for cid in record.get("claim_ids", []) if cid not in context["claims"]]
            unknown_evidence = [eid for eid in record.get("evidence_ids", []) if eid not in store]
            if unknown_claims:
                issues.append(
                    issue(
                        "unknown_synthesis_claim",
                        f"synthesis.{field}[{index}]가 없는 claim을 참조함: {', '.join(unknown_claims)}",
                        None,
                    )
                )
            if unknown_evidence:
                issues.append(
                    issue(
                        "unknown_synthesis_evidence",
                        f"synthesis.{field}[{index}]가 없는 evidence를 참조함: {', '.join(unknown_evidence)}",
                        None,
                    )
                )
    return issues


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。])\s+|\n", text or "") if part.strip()]


def _metric_issues(section_id: SectionId, markdown: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for sentence in _sentences(markdown):
        folded = sentence.casefold().replace(" ", "")
        if "93.3" in sentence:
            if "deepseek67b" not in folded or "mla만으로" in folded or "mla단독" in folded:
                issues.append(issue("metric_93_3_context", "93.3%에는 DeepSeek 67B 대비 전체 비교 조건이 필요함", section_id))
        if "5.76" in sentence and not re.search(r"8\s*[×x]\s*h800|8개\s*h800", sentence, re.IGNORECASE):
            issues.append(issue("metric_5_76_context", "5.76×에는 8×H800 조건이 필요함", section_id))
        if "35.7" in sentence and "최대" not in sentence:
            issues.append(issue("metric_35_7_context", "35.7%는 최대값으로 표시해야 함", section_id))
        if "1.81" in sentence:
            required = ("ttft", "turn5", "recomputation")
            if not all(token in folded for token in required):
                issues.append(issue("metric_1_81_context", "1.81×에는 turn 5, TTFT, recomputation baseline 조건이 필요함", section_id))
        if "42.5" in sentence:
            if "mla" in folded or not any(token in folded for token in ("훈련", "training")):
                issues.append(issue("metric_42_5_attribution", "42.5%를 MLA 효과로 귀속할 수 없음", section_id))
    return issues


def validate_section(draft: SectionDraft, context: dict) -> list[ValidationIssue]:
    section_id = draft["section_id"]
    if section_id == "reference":
        return []
    issues: list[ValidationIssue] = []
    store = context["evidence_store"]
    claims = context["claims"]

    heading_level = "#" if section_id in {
        "summary",
        "background",
        "technology_selection",
        "technology_overview",
        "limitations",
    } else "##"
    expected_heading = f"{heading_level} {draft['title']}"
    if not draft["markdown"].lstrip().startswith(expected_heading):
        issues.append(
            issue(
                "section_heading",
                f"섹션이 지정 제목으로 시작하지 않음: {expected_heading}",
                section_id,
            )
        )

    unknown_evidence = [eid for eid in draft["evidence_ids"] if eid not in store]
    if unknown_evidence:
        issues.append(issue("unknown_evidence", f"섹션이 없는 evidence를 참조함: {', '.join(unknown_evidence)}", section_id))
    unknown_claims = [cid for cid in draft["claim_ids"] if cid not in claims]
    if unknown_claims:
        issues.append(issue("unknown_claim", f"섹션이 없는 claim을 참조함: {', '.join(unknown_claims)}", section_id))

    inline_ids = extract_citations(draft["markdown"])
    if set(inline_ids) != set(draft["evidence_ids"]):
        issues.append(issue("citation_manifest_mismatch", "본문 citation과 섹션 evidence_ids가 일치하지 않음", section_id))
    for evidence_id in inline_ids:
        if evidence_id not in store:
            issues.append(issue("unknown_inline_evidence", f"본문에 없는 evidence ID가 있음: {evidence_id}", section_id))

    source_text = "\n".join(
        [
            " ".join(
                [claims[cid].get("statement", "")]
                + list(claims[cid].get("conditions", []))
                + [claims[cid].get("uncertainty", "")]
            )
            for cid in draft["claim_ids"]
            if cid in claims
        ]
        + [
            store[eid].get("excerpt", "")
            for eid in draft["evidence_ids"]
            if eid in store
        ]
    )
    grounded = extract_measurements(source_text)
    ungrounded = sorted(extract_measurements(draft["markdown"]) - grounded)
    if ungrounded:
        issues.append(issue("numeric_grounding", f"근거/claim에 없는 측정값: {', '.join(ungrounded)}", section_id))

    for expression in PROHIBITED_COMPARISON:
        if expression in draft["markdown"]:
            issues.append(issue("prohibited_comparison", f"금지된 서열·권고 표현: {expression}", section_id))

    if context.get("not_found_present") and re.search(r"(?:의견|근거|사례|자료).{0,12}(?:없다|존재하지 않는다)", draft["markdown"]):
        issues.append(issue("not_found_generalization", "not_found를 실제 부재로 일반화함", section_id))

    issues.extend(_metric_issues(section_id, draft["markdown"]))
    return issues


def validate_report(report: dict, context: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    markdown = report.get("markdown", "")
    positions = [markdown.find(heading) for heading in REQUIRED_HEADINGS]
    if any(position < 0 for position in positions):
        missing = [heading for heading, position in zip(REQUIRED_HEADINGS, positions) if position < 0]
        issues.append(issue("required_sections", f"필수 목차 누락: {', '.join(missing)}", None))
    elif positions != sorted(positions):
        issues.append(issue("section_order", "보고서 목차 순서가 규격과 다름", None))
    if not markdown.lstrip().startswith("# SUMMARY"):
        issues.append(issue("summary_first", "SUMMARY가 보고서 첫 장이 아님", None))
    expected_tail = f"- {report['references'][-1]}" if report.get("references") else "자료 미확인"
    if not markdown.rstrip().endswith(expected_tail):
        issues.append(issue("reference_last", "REFERENCE가 보고서 마지막이 아님", None))

    cited = report.get("cited_evidence_ids", [])
    if len(cited) != len(set(cited)):
        issues.append(issue("duplicate_citation_id", "cited_evidence_ids에 중복이 있음", None))
    unknown = [eid for eid in cited if eid not in context["evidence_store"]]
    if unknown:
        issues.append(issue("unknown_report_evidence", f"최종 보고서가 없는 evidence를 참조함: {', '.join(unknown)}", None))

    identities = [source_identity(context["evidence_store"][eid]) for eid in cited if eid in context["evidence_store"]]
    if len(report.get("references", [])) != len(set(identities)):
        issues.append(issue("reference_dedupe", "REFERENCE가 source 단위로 중복 제거되지 않음", None))

    degraded = {name: status for name, status in context["upstream_statuses"].items() if status in {"partial", "failed"}}
    limitation_text = next(
        (section["markdown"] for section in report.get("sections", []) if section.get("title") == "6. 한계점"),
        "",
    )
    for name, status in degraded.items():
        if name not in limitation_text or status not in limitation_text:
            issues.append(issue("upstream_status_hidden", f"{name}의 {status} 상태가 한계점에 노출되지 않음", "limitations"))
    return issues


def blocking(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return [item for item in issues if item["blocking"]]

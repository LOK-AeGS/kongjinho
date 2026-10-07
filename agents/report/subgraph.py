"""보고서 생성 흐름: 입력 정규화 → 본문 작성 → 최종화 사이클 → 부분 수정."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from agents.report.budget import (
    MAX_PAGE_GUARD_ROUNDS,
    OVER_BUDGET_TOLERANCE,
    PAGE_LIMIT,
    SECTION_BUDGETS,
    Budgets,
    balanced_pick,
    body_length,
    clip,
    default_budgets,
    estimate_pages,
    omitted_note,
    round_robin_pick,
    shrink,
)
from agents.report.prompts import build_section_prompt
from agents.report.references import collect_references, number_citations, reference_numbers
from agents.report.state import (
    BODY_SECTION_ORDER,
    MAX_REPORT_REVISIONS,
    SECTION_ORDER,
    NormalizedClaim,
    NormalizedEvidence,
    NormalizedInput,
    ReportAgentDeps,
    SectionDraft,
    SectionId,
)
from agents.report.validators import (
    blocking,
    extract_citations,
    validate_input,
    validate_budget,
    validate_report,
    validate_section,
)


SECTION_TITLES: dict[SectionId, str] = {
    "summary": "SUMMARY",
    "background": "1. 분석 배경",
    "technology_selection": "2. 기술 선정",
    "technology_overview": "3. 기술 개요",
    "trl": "4.1 기술 성숙도(TRL)",
    "market": "4.2 시장성",
    "stakeholder": "4.3 이해관계자",
    "domain": "4.4 도메인 적용",
    "comparison_matrix": "5.1 비교 매트릭스",
    "conditions": "5.2 적용 조건 대조표",
    "conflicts": "5.3 관점 간 상충 지점",
    "shared_and_complement": "5.4 공유 근거 및 보완 관계",
    "open_questions": "5.5 남은 확인 과제",
    "limitations": "6. 한계점",
    "reference": "REFERENCE",
}


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _normalize_evidence(key: str, raw: dict) -> NormalizedEvidence | None:
    evidence_id = str(raw.get("evidence_id") or raw.get("id") or key or "").strip()
    if not evidence_id:
        return None
    page = raw.get("page")
    section = raw.get("section")
    locator = raw.get("locator") or raw.get("page_or_locator")
    if not locator:
        locator = ", ".join(str(value) for value in (page, section) if value not in (None, ""))
    return {
        "evidence_id": evidence_id,
        "document_id": raw.get("document_id") or raw.get("doc_id"),
        "source_type": str(raw.get("source_type") or "other"),
        "title": str(raw.get("title") or ""),
        "author_or_organization": str(raw.get("author_or_organization") or raw.get("author_or_org") or ""),
        "url": str(raw.get("url") or ""),
        "published_date": raw.get("published_date") or raw.get("published_at"),
        "accessed_date": str(raw.get("accessed_date") or raw.get("accessed_at") or ""),
        "locator": str(locator or ""),
        "excerpt": str(raw.get("excerpt") or raw.get("quote") or ""),
        "stance": str(raw.get("stance") or "neutral"),
        "content_hash": str(raw.get("content_hash") or raw.get("content_sha256") or ""),
    }


def _normalize_claim(raw: dict, perspective: str, fallback_id: str) -> NormalizedClaim | None:
    claim_id = str(raw.get("claim_id") or raw.get("id") or fallback_id).strip()
    statement = str(raw.get("text") or raw.get("statement") or raw.get("findings") or raw.get("explanation") or "").strip()
    if not claim_id or not statement:
        return None
    technology_ids = raw.get("technology_ids")
    if technology_ids is None and raw.get("technology_id"):
        technology_ids = [raw["technology_id"]]
    if technology_ids is None and raw.get("technology"):
        technology_ids = ["sw", "hw"] if raw["technology"] == "both" else [raw["technology"]]
    limitations = raw.get("limitations") or []
    return {
        "claim_id": claim_id,
        "perspective": perspective,
        "technology_ids": list(technology_ids or []),
        "topic": str(raw.get("topic") or raw.get("criterion") or ""),
        "statement": statement,
        "basis": str(raw.get("basis") or "unknown"),
        "evidence_ids": [str(item) for item in raw.get("evidence_ids", [])],
        "conditions": [str(item) for item in raw.get("conditions", [])],
        "uncertainty": str(raw.get("uncertainty") or "; ".join(str(item) for item in limitations)),
    }


def _gap_text(raw) -> str:
    if not isinstance(raw, dict):
        return str(raw)
    parts = [raw.get("technology"), raw.get("perspective"), raw.get("criterion"), raw.get("reason")]
    missing = raw.get("missing_evidence") or []
    if missing:
        parts.append("미확인 근거: " + ", ".join(str(item) for item in missing))
    return " / ".join(str(item) for item in parts if item)


def _completion(findings: dict | None) -> tuple[str, list[str]]:
    if not findings:
        return "failed", ["상위 결과가 전달되지 않음"]
    status = str(findings.get("status") or "complete")
    gaps = [_gap_text(item) for item in _as_list(findings.get("gaps"))]
    return status, list(dict.fromkeys(item for item in gaps if item))


def normalize_state(state: dict) -> NormalizedInput:
    """확정 AppState를 보고서 작성용 정규형에 읽기 전용으로 투영한다."""
    request = deepcopy(state.get("request") or {})
    technologies = deepcopy(state.get("selected_tech") or {})
    domain = str(state.get("domain") or request.get("scope") or "")
    as_of_date = str(request.get("as_of") or "")
    initial_revisions = int((state.get("retries") or {}).get("report", 0) or 0)

    findings: dict[str, dict | None] = {
        "technical": state.get("technical_findings"),
        "market": state.get("market_findings"),
        "stakeholder": state.get("stakeholder_findings"),
        "domain": state.get("domain_findings"),
    }

    evidence_store: dict[str, NormalizedEvidence] = {}
    for key, raw in (state.get("evidence_store") or {}).items():
        normalized = _normalize_evidence(str(key), raw)
        if normalized:
            evidence_store[normalized["evidence_id"]] = normalized
    for result in findings.values():
        if not result:
            continue
        for index, raw in enumerate(result.get("evidence", [])):
            normalized = _normalize_evidence(str(index), raw)
            if normalized:
                evidence_store.setdefault(normalized["evidence_id"], normalized)

    claims: dict[str, NormalizedClaim] = {}
    for perspective, result in findings.items():
        if not result:
            continue
        raw_claims = result.get("claims") or []
        for index, raw in enumerate(raw_claims, 1):
            claim = _normalize_claim(raw, perspective, f"{perspective}:claim:{index:03d}")
            if claim:
                claims.setdefault(claim["claim_id"], claim)

    synthesis = deepcopy(state.get("synthesis") or {})
    for field in ("summary_claims",):
        for index, raw in enumerate(synthesis.get(field, []), 1):
            claim = _normalize_claim(raw, "synthesis", f"synthesis:{field}:{index:03d}")
            if claim:
                claims.setdefault(claim["claim_id"], claim)

    statuses: dict[str, str] = {}
    gaps: list[str] = []
    for perspective, result in findings.items():
        status, result_gaps = _completion(result)
        statuses[perspective] = status
        gaps.extend(f"{perspective}: {value}" for value in result_gaps)
    status, result_gaps = _completion(synthesis or None)
    statuses["synthesis"] = status
    gaps.extend(f"synthesis: {value}" for value in result_gaps)

    not_found = any(ev.get("stance") == "not_found" for ev in evidence_store.values())
    for result in findings.values():
        if result and any(item.get("status") == "not_found" for item in result.get("search_outcomes", [])):
            not_found = True

    return {
        "as_of_date": as_of_date,
        "domain": domain,
        "technologies": technologies,
        "initial_revision_rounds": initial_revisions,
        "max_revision_rounds": MAX_REPORT_REVISIONS,
        "findings": findings,
        "claims": claims,
        "evidence_store": evidence_store,
        "synthesis": synthesis,
        "upstream_statuses": statuses,
        "upstream_gaps": list(dict.fromkeys(gaps)),
        "not_found_present": not_found,
    }


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _heading(section_id: SectionId) -> str:
    level = "#" if section_id in {"summary", "background", "technology_selection", "technology_overview", "limitations", "reference"} else "##"
    return f"{level} {SECTION_TITLES[section_id]}"


def _counter_claim(context: dict):
    store = context["evidence_store"]
    return lambda item: any(
        (store.get(evidence_id) or {}).get("stance") == "counter"
        for evidence_id in item.get("evidence_ids", [])
    )


def _usable_claims(context: dict, perspective: str) -> list[dict]:
    store = context["evidence_store"]
    result = []
    for claim in context["claims"].values():
        if claim.get("perspective") != perspective:
            continue
        evidence_ids = claim.get("evidence_ids", [])
        if any(eid not in store for eid in evidence_ids):
            continue
        if claim.get("basis") in {"direct", "direct_evidence"} and not evidence_ids:
            continue
        result.append(claim)
    return result


def select_claims(context: dict, perspective: str, limit: int) -> tuple[list[dict], int]:
    """관점 claim을 SW·HW 균형과 반대 근거를 지키며 limit개 고른다. (선택, 전체 수)"""
    claims = _usable_claims(context, perspective)
    return balanced_pick(claims, limit, is_counter=_counter_claim(context)), len(claims)


def _usable_records(context: dict, perspective: str, *, trl_only: bool = False) -> list[dict]:
    result = context["findings"].get(perspective) or {}
    records = []
    for record in result.get("records", []):
        criterion = str(record.get("criterion") or "")
        if trl_only and "trl" not in criterion.casefold() and "성숙" not in criterion:
            continue
        ids = record.get("evidence_ids", [])
        if any(eid not in context["evidence_store"] for eid in ids):
            continue
        records.append(record)
    return records


def select_records(
    context: dict, perspective: str, limit: int, *, trl_only: bool = False
) -> tuple[list[dict], int]:
    records = _usable_records(context, perspective, trl_only=trl_only)
    return balanced_pick(records, limit, is_counter=_counter_claim(context)), len(records)


def _matrix_groups(context: dict) -> dict[tuple[str, str], dict[str, dict]]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for cell in context["synthesis"].get("matrix") or []:
        key = (str(cell.get("perspective") or ""), str(cell.get("criterion") or ""))
        grouped.setdefault(key, {})[str(cell.get("technology") or "")] = cell
    return grouped


def select_matrix_keys(context: dict, limit: int) -> tuple[list[tuple[str, str]], int]:
    """비교 매트릭스 행을 관점별로 번갈아 골라 네 관점이 모두 남게 한다."""
    keys = list(_matrix_groups(context))
    return round_robin_pick(keys, limit, key=lambda item: item[0]), len(keys)


def _linked_rows(context: dict, rows: list[dict]) -> list[tuple[dict, list[str], list[str]]]:
    """claim·evidence 참조가 모두 실재하는 행만 (행, claim_ids, evidence_ids)로 돌려준다."""
    linked = []
    for row in rows:
        ids = [cid for cid in row.get("claim_ids", []) if cid in context["claims"]]
        direct_ids = [eid for eid in row.get("evidence_ids", []) if eid in context["evidence_store"]]
        if (row.get("claim_ids") and len(ids) != len(row["claim_ids"])) or (
            row.get("evidence_ids") and len(direct_ids) != len(row["evidence_ids"])
        ):
            continue
        evidence_ids = _unique(direct_ids + [eid for cid in ids for eid in context["claims"][cid].get("evidence_ids", [])])
        linked.append((row, ids, evidence_ids))
    return linked


def select_relations(
    context: dict, accepted: set[str], limit: int
) -> tuple[list[tuple[dict, list[str], list[str]]], int]:
    """설명이 있는 행을, 상충을 일치보다, 미해소를 해소보다 앞에 두고 limit개 고른다."""
    rows = context["synthesis"].get("cross_findings") or context["synthesis"].get("relations") or []
    linked = [item for item in _linked_rows(context, rows) if item[0].get("kind") in accepted]
    ranked = sorted(
        linked,
        key=lambda item: (
            0 if (item[0].get("explanation") or item[0].get("reason")) else 1,
            0 if item[0].get("kind") == "conflict" else 1,
            0 if item[0].get("resolution") == "unresolved" else 1,
        ),
    )
    return ranked[:limit], len(linked)


def select_contrast(context: dict, limit: int) -> tuple[list[tuple[dict, list[str], list[str]]], int]:
    linked = _linked_rows(context, context["synthesis"].get("contrast_table") or [])
    picked = round_robin_pick(
        linked, limit, key=lambda item: str(item[0].get("criterion") or "").split("/", 1)[0]
    )
    return picked, len(linked)


def select_gaps(context: dict, limit: int) -> tuple[list[str], int]:
    gaps = list(context["upstream_gaps"])
    for request in context["synthesis"].get("retry_requests", []):
        gaps.append(str(request.get("reason") if isinstance(request, dict) else request))
    gaps = _unique(gaps)
    return round_robin_pick(gaps, limit, key=lambda item: item.split(":", 1)[0]), len(gaps)


class DeterministicSectionWriter:
    """상위 구조화 결과를 그대로 배치하는 API-key-free 재현용 writer.

    섹션 예산(budget.py)을 넘지 않을 때까지 항목 수를 줄여 렌더링한다.
    분량 가드는 예산을 넘긴 LLM 섹션을 이 writer의 출력으로 교체한다.
    """

    receives_full_context = True

    def __init__(self, budgets: Budgets | None = None) -> None:
        self.budgets = budgets or default_budgets()

    def write(self, section_id: SectionId, context: dict) -> SectionDraft:
        return self._render(section_id, context)

    def repair(self, section_id: SectionId, draft: SectionDraft, issues: list[dict], context: dict) -> SectionDraft:
        # 외부 초안에 문제가 있어도 상위 claim만 사용하는 결정적 출력으로 해당 섹션만 교체한다.
        return self._render(section_id, context)

    def _render(self, section_id: SectionId, context: dict) -> SectionDraft:
        # 하위 클래스가 __init__을 건너뛰어도 기본 예산으로 동작한다.
        budget = (getattr(self, "budgets", None) or SECTION_BUDGETS)[section_id]
        draft = self._build(section_id, context, budget.max_items)
        for limit in range(budget.max_items - 1, 0, -1):
            if body_length(draft["markdown"]) <= budget.max_chars:
                break
            draft = self._build(section_id, context, limit)
        return draft

    def _claim_lines(self, claims: list[dict]) -> tuple[list[str], list[str], list[str]]:
        claim_ids: list[str] = []
        evidence_ids: list[str] = []
        lines: list[str] = []
        for claim in claims:
            ids = claim.get("evidence_ids", [])
            citation = f" 〔근거: {', '.join(ids)}〕" if ids else ""
            details: list[str] = []
            if claim.get("conditions"):
                details.append("조건: " + clip("; ".join(claim["conditions"]), 120))
            if claim.get("uncertainty"):
                details.append("불확실성: " + clip(claim["uncertainty"], 120))
            suffix = f" ({' / '.join(details)})" if details else ""
            lines.append(f"- {clip(claim['statement'])}{suffix}{citation}")
            claim_ids.append(claim["claim_id"])
            evidence_ids.extend(ids)
        return lines, claim_ids, evidence_ids

    def _claim_draft(
        self, section_id: SectionId, claims: list[dict], empty: str, total: int | None = None
    ) -> SectionDraft:
        lines, claim_ids, evidence_ids = self._claim_lines(claims)
        if not lines:
            lines = [empty]
        note = omitted_note(total if total is not None else len(claims), len(claims))
        if note:
            lines.append(note)
        return {
            "section_id": section_id,
            "title": SECTION_TITLES[section_id],
            "markdown": _heading(section_id) + "\n\n" + "\n".join(lines),
            "claim_ids": _unique(claim_ids),
            "evidence_ids": _unique(evidence_ids),
        }

    def _record_line(self, record: dict, context: dict) -> tuple[str, list[str]]:
        ids = list(record.get("evidence_ids", []))
        value = record.get("value") or record.get("assessment") or "판단 보류"
        citation = f" 〔근거: {', '.join(ids)}〕" if ids else ""
        limitations = clip("; ".join(str(item) for item in record.get("limitations", [])), 120)
        suffix = f" / {limitations}" if limitations else ""
        line = (
            f"- {record.get('technology', '기술')}: {record.get('criterion', '')} "
            f"{value} — {clip(record.get('findings', ''), 200)}{suffix}{citation}"
        )
        return line, ids

    def _perspective(self, section_id: SectionId, context: dict, limit: int) -> SectionDraft:
        claims, total = select_claims(context, section_id, limit)
        draft = self._claim_draft(
            section_id, claims, f"- {SECTION_TITLES[section_id]}에 인용 가능한 결과가 확인되지 않았다.", total
        )
        if len(claims) >= limit:
            return draft
        # claim이 적은 관점(예: 시장)은 판정 기록으로 채워 관점 간 분량 차이를 줄인다.
        records, _ = select_records(context, section_id, limit - len(claims))
        if not records:
            return draft
        lines, evidence_ids = [], []
        for record in records:
            line, ids = self._record_line(record, context)
            lines.append(line)
            evidence_ids.extend(ids)
        body = draft["markdown"].split("\n\n", 1)[1]
        if not claims:
            body = ""
        draft["markdown"] = _heading(section_id) + "\n\n" + "\n".join(filter(None, [body, *lines]))
        draft["evidence_ids"] = _unique(draft["evidence_ids"] + evidence_ids)
        return draft

    def _build(self, section_id: SectionId, context: dict, limit: int) -> SectionDraft:
        if section_id == "summary":
            synthesis_claims = _usable_claims(context, "synthesis")
            base = self._claim_draft(section_id, synthesis_claims[:limit], "- 종합 결과에서 인용 가능한 핵심 문장이 확인되지 않았다.")
            relation_lines: list[str] = []
            relation_claims: list[str] = []
            relation_evidence: list[str] = []
            seen_kinds: set[str] = set()
            rows = context["synthesis"].get("cross_findings") or context["synthesis"].get("relations") or []
            for row, claim_ids, evidence_ids in _linked_rows(context, rows):
                kind = str(row.get("kind") or "")
                if kind not in {"agreement", "conflict"} or kind in seen_kinds:
                    continue
                label = "주요 일치" if kind == "agreement" else "주요 상충"
                cause = f" / 원인 유형: {row['conflict_type']}" if row.get("conflict_type") else ""
                citation = f" 〔근거: {', '.join(evidence_ids)}〕" if evidence_ids else ""
                explanation = clip(row.get("explanation") or row.get("reason") or "설명 미입력")
                relation_lines.append(f"- {label}: {explanation}{cause}{citation}")
                relation_claims.extend(claim_ids)
                relation_evidence.extend(evidence_ids)
                seen_kinds.add(kind)
            scope = f"평가 범위: {context['domain'] or '미지정'} / 조사 기준일: {context['as_of_date'] or '미지정'}"
            degraded = [f"{name}={status}" for name, status in context["upstream_statuses"].items() if status != "complete"]
            tail = "\n" + "\n".join(relation_lines) if relation_lines else ""
            tail += "\n\n- " + scope + "\n- 공개 정보로 확인 가능한 범위 안에서 작성했다."
            if degraded:
                tail += "\n- 판단 보류 상태: " + ", ".join(degraded)
            elif context["upstream_gaps"]:
                tail += "\n- 판단 보류 항목: " + context["upstream_gaps"][0]
            base["markdown"] += tail
            base["claim_ids"] = _unique(base["claim_ids"] + relation_claims)
            base["evidence_ids"] = _unique(base["evidence_ids"] + relation_evidence)
            return base
        if section_id == "background":
            claims = _usable_claims(context, "technical")
            return self._claim_draft(section_id, claims[:limit], "- 인용 가능한 분석 배경 자료가 확인되지 않았다.")
        if section_id == "technology_selection":
            lines = []
            for side in ("sw", "hw"):
                tech = context["technologies"].get(side, {})
                name = tech.get("name") or side
                reason = tech.get("selection_reason") or "선정 이유 미입력"
                lines.append(f"- {side.upper()}: {name} — {reason}")
            return self._plain(section_id, lines)
        if section_id == "technology_overview":
            claims, total = select_claims(context, "technical", limit)
            return self._claim_draft(section_id, claims, "- 인용 가능한 기술 개요가 확인되지 않았다.", total)
        if section_id == "trl":
            records, total = select_records(context, "technical", limit, trl_only=True)
            lines, evidence_ids = [], []
            for record in records:
                line, ids = self._record_line(record, context)
                lines.append(line)
                evidence_ids.extend(ids)
            note = omitted_note(total, len(records))
            draft = self._plain(section_id, (lines or ["- 공개 정보 기반 TRL 기록이 확인되지 않았다."]) + ([note] if note else []))
            draft["evidence_ids"] = _unique(evidence_ids)
            return draft
        if section_id in {"market", "stakeholder", "domain"}:
            return self._perspective(section_id, context, limit)
        if section_id == "comparison_matrix":
            return self._matrix(context, limit)
        if section_id == "conditions":
            return self._conditions(context, limit)
        if section_id in {"conflicts", "shared_and_complement"}:
            return self._relations(section_id, context, limit)
        if section_id == "open_questions":
            gaps, total = select_gaps(context, limit)
            note = omitted_note(total, len(gaps))
            lines = [f"- {clip(value, 200)}" for value in gaps] or ["- 자료 미확인"]
            return self._plain(section_id, lines + ([note] if note else []))
        if section_id == "limitations":
            # upstream 상태 노출은 검증 필수 항목이라 예산과 무관하게 남긴다.
            required = ["- 모든 평가는 공개 정보로 확인 가능한 범위에 한정된다."]
            for name, status in context["upstream_statuses"].items():
                if status in {"partial", "failed"}:
                    required.append(f"- {name} upstream 상태는 {status}이며 관련 공백을 최종 판단에 반영해야 한다.")
            if context.get("not_found_present"):
                required.append("- not_found는 기록된 검색 범위에서 자료를 확인하지 못한 상태이며 실제 부재를 뜻하지 않는다.")
            optional = [f"- {clip(value, 200)}" for value in context["synthesis"].get("limitations", [])]
            optional.append("- 근거 수준과 적용 조건을 함께 표시하고 반대 방향 근거의 미확인을 남겨 확증편향을 줄였다.")
            room = max(0, limit - len(required))
            return self._plain(section_id, _unique(required + optional[:room]))
        raise ValueError(f"지원하지 않는 섹션: {section_id}")

    def _plain(self, section_id: SectionId, lines: list[str]) -> SectionDraft:
        return {
            "section_id": section_id,
            "title": SECTION_TITLES[section_id],
            "markdown": _heading(section_id) + "\n\n" + "\n".join(lines),
            "claim_ids": [],
            "evidence_ids": [],
        }

    def _matrix(self, context: dict, limit: int) -> SectionDraft:
        grouped = _matrix_groups(context)
        keys, total = select_matrix_keys(context, limit)
        lines = ["| 관점 | 기준 | SW | HW |", "|---|---|---|---|"]
        evidence_ids: list[str] = []
        for perspective, criterion in keys:
            pair = grouped[(perspective, criterion)]
            row_evidence = self._record_evidence(context, perspective, criterion)
            citation = f" 〔근거: {', '.join(row_evidence)}〕" if row_evidence else ""
            sw = pair.get("sw") or {}
            hw = pair.get("hw") or {}
            sw_text = str(sw.get("assessment") or "자료 미확인")
            hw_text = str(hw.get("assessment") or "자료 미확인")
            if sw.get("value"):
                sw_text += f" ({sw['value']})"
            if hw.get("value"):
                hw_text += f" ({hw['value']})"
            lines.append(
                f"| {perspective} | {criterion} | {sw_text} | {hw_text}{citation} |"
            )
            evidence_ids.extend(row_evidence)
        if len(lines) == 2:
            lines.append("| 자료 미확인 | 자료 미확인 | 자료 미확인 | 자료 미확인 |")
        note = omitted_note(total, len(keys), "행")
        draft = self._plain("comparison_matrix", lines + (["", note] if note else []))
        draft["evidence_ids"] = _unique(evidence_ids)
        return draft

    def _record_evidence(self, context: dict, perspective: str, criterion: str) -> list[str]:
        findings = context["findings"].get(perspective) or {}
        ids = [
            evidence_id
            for record in findings.get("records", [])
            if str(record.get("criterion") or "") == criterion
            for evidence_id in record.get("evidence_ids", [])
            if evidence_id in context["evidence_store"]
        ]
        return _unique(ids)

    def _conditions(self, context: dict, limit: int) -> SectionDraft:
        rows, total = select_contrast(context, limit)
        lines = ["| 항목 | SW | HW |", "|---|---|---|"]
        claim_ids, evidence_ids = [], []
        for row, ids, row_evidence in rows:
            citation = f" 〔근거: {', '.join(row_evidence)}〕" if row_evidence else ""
            label = row.get("criterion") or row.get("label") or row.get("item") or "조건"
            lines.append(f"| {label} | {row.get('sw', row.get('sw_assessment', ''))} | {row.get('hw', row.get('hw_assessment', ''))}{citation} |")
            claim_ids.extend(ids)
            evidence_ids.extend(row_evidence)
        if len(lines) == 2:
            lines.append("| 자료 미확인 | 자료 미확인 | 자료 미확인 |")
        note = omitted_note(total, len(rows), "행")
        draft = self._plain("conditions", lines + (["", note] if note else []))
        draft["claim_ids"], draft["evidence_ids"] = _unique(claim_ids), _unique(evidence_ids)
        return draft

    def _relations(self, section_id: SectionId, context: dict, limit: int) -> SectionDraft:
        if section_id == "conflicts":
            accepted = {"agreement", "conflict"}
        else:
            accepted = {"complement", "complementarity", "shared_evidence"}
        rows, total = select_relations(context, accepted, limit)
        lines, claim_ids, evidence_ids = [], [], []
        for row, ids, row_evidence in rows:
            citation = f" 〔근거: {', '.join(row_evidence)}〕" if row_evidence else ""
            explanation = clip(row.get("explanation") or row.get("reason") or "설명 미입력")
            resolution = f" / 해소 상태: {row['resolution']}" if row.get("resolution") else ""
            lines.append(f"- {row.get('kind')}: {explanation}{resolution}{citation}")
            claim_ids.extend(ids)
            evidence_ids.extend(row_evidence)
        note = omitted_note(total, len(rows))
        draft = self._plain(section_id, (lines or ["- 구조화된 관계 기록이 확인되지 않았다."]) + ([note] if note else []))
        draft["claim_ids"], draft["evidence_ids"] = _unique(claim_ids), _unique(evidence_ids)
        return draft


def _synthesis_view(section_id: SectionId, context: NormalizedInput, limit: int) -> dict:
    """시사점 계열 섹션에 synthesis 전체 대신 그 섹션에 필요한 행만 선별해 넘긴다."""
    synthesis = context["synthesis"]
    if section_id == "summary":
        relations, _ = select_relations(context, {"agreement", "conflict"}, 2)
        return {
            "summary_claims": [c for c in synthesis.get("summary_claims", [])][:limit],
            "cross_findings": [row for row, _, _ in relations],
            "status": synthesis.get("status"),
        }
    if section_id == "comparison_matrix":
        keys, total = select_matrix_keys(context, limit)
        wanted = set(keys)
        cells = [
            cell for cell in synthesis.get("matrix") or []
            if (str(cell.get("perspective") or ""), str(cell.get("criterion") or "")) in wanted
        ]
        return {"matrix": cells, "omitted": {"total_rows": total, "kept_rows": len(keys)}}
    if section_id == "conditions":
        rows, total = select_contrast(context, limit)
        return {"contrast_table": [row for row, _, _ in rows], "omitted": {"total_rows": total, "kept_rows": len(rows)}}
    if section_id in {"conflicts", "shared_and_complement"}:
        accepted = {"agreement", "conflict"} if section_id == "conflicts" else {"complement", "complementarity", "shared_evidence"}
        rows, total = select_relations(context, accepted, limit)
        return {"cross_findings": [row for row, _, _ in rows], "omitted": {"total": total, "kept": len(rows)}}
    if section_id == "open_questions":
        gaps, total = select_gaps(context, limit)
        return {"gaps": gaps, "omitted": {"total": total, "kept": len(gaps)}}
    # limitations
    return {
        "limitations": list(synthesis.get("limitations", []))[:limit],
        "imbalance": synthesis.get("imbalance", []),
    }


def section_payload(
    section_id: SectionId, context: NormalizedInput, budgets: Budgets | None = None
) -> dict:
    """외부 writer에 전체 evidence_store 대신 해당 섹션의 최소 입력만 제공한다.

    결정적 writer와 같은 선별 함수로 항목을 max_items개까지 줄이고, 분량 상한을 함께 넘긴다.
    """
    budget = (budgets or SECTION_BUDGETS)[section_id]
    limit = budget.max_items
    perspective = {
        "background": "technical",
        "technology_overview": "technical",
        "trl": "technical",
        "market": "market",
        "stakeholder": "stakeholder",
        "domain": "domain",
    }.get(section_id)
    payload = {
        "section_id": section_id,
        "section_title": SECTION_TITLES[section_id],
        "required_heading": _heading(section_id),
        "as_of_date": context["as_of_date"],
        "domain": context["domain"],
        "technologies": context["technologies"],
        "budget": {"max_chars": budget.max_chars, "max_items": limit},
    }
    if perspective:
        source = context["findings"].get(perspective) or {}
        claims, total_claims = select_claims(context, perspective, limit)
        records, total_records = select_records(
            context, perspective, limit, trl_only=section_id == "trl"
        )
        payload["findings"] = {
            "perspective": perspective,
            "status": source.get("status"),
            "claims": [] if section_id == "trl" else claims,
            "records": [] if section_id in {"background", "technology_overview"} else records,
            "gaps": list(source.get("gaps") or [])[:limit],
        }
        payload["omitted"] = {
            "claims": {"total": total_claims, "kept": len(claims)},
            "records": {"total": total_records, "kept": len(records)},
        }
    elif section_id in {"summary", "comparison_matrix", "conditions", "conflicts", "shared_and_complement", "open_questions", "limitations"}:
        payload["synthesis"] = _synthesis_view(section_id, context, limit)
        payload["upstream_statuses"] = context["upstream_statuses"]
        if section_id in {"summary", "limitations"}:
            payload["upstream_gaps"] = context["upstream_gaps"][:limit]
    claim_ids = set()
    raw = str(payload)
    for claim_id in context["claims"]:
        if claim_id in raw:
            claim_ids.add(claim_id)
    direct_evidence_ids = [
        evidence_id for evidence_id in context["evidence_store"] if evidence_id in raw
    ]
    evidence_ids = _unique(
        direct_evidence_ids
        + [
            eid
            for claim_id in claim_ids
            for eid in context["claims"][claim_id].get("evidence_ids", [])
        ]
    )
    payload["evidence"] = {eid: context["evidence_store"][eid] for eid in evidence_ids if eid in context["evidence_store"]}
    return payload


def _writer_context(
    section_id: SectionId,
    context: NormalizedInput,
    *,
    include_normalized: bool,
    budgets: Budgets | None = None,
) -> dict:
    payload = section_payload(section_id, context, budgets)
    public = {"payload": payload, "prompt": build_section_prompt(section_id, payload)}
    # 결정적 재현 writer만 정규형 전체를 읽는다. 일반 LLM writer에는 최소 입력만 노출한다.
    return {**context, **public} if include_normalized else public


def _resolve_writer(deps: ReportAgentDeps):
    if deps.writer is not None:
        return deps.writer
    if deps.generation_mode == "deterministic":
        return DeterministicSectionWriter()
    if deps.generation_mode == "llm":
        from agents.report.writer import create_llm_writer

        return create_llm_writer(deps)
    raise ValueError(f"지원하지 않는 보고서 생성 모드: {deps.generation_mode}")


def _include_normalized(writer, deps: ReportAgentDeps) -> bool:
    return bool(
        deps.writer_receives_full_context
        or getattr(writer, "receives_full_context", False)
    )


def _generation_metadata(writer, deps: ReportAgentDeps) -> dict:
    if isinstance(writer, DeterministicSectionWriter):
        return {"mode": "deterministic", "provider": None, "model": None}
    return {
        "mode": "custom" if deps.writer is not None else "llm",
        "provider": getattr(writer, "provider", deps.model_provider),
        "model": getattr(writer, "model", deps.model),
        "temperature": getattr(writer, "temperature", deps.temperature),
    }


def _assemble(sections: dict[SectionId, SectionDraft]) -> str:
    parts = [sections["summary"]["markdown"], sections["background"]["markdown"], sections["technology_selection"]["markdown"], sections["technology_overview"]["markdown"]]
    parts.append("# 4. 관점별 평가")
    parts.extend(sections[name]["markdown"] for name in ("trl", "market", "stakeholder", "domain"))
    parts.append("# 5. 시사점")
    parts.extend(sections[name]["markdown"] for name in ("comparison_matrix", "conditions", "conflicts", "shared_and_complement", "open_questions"))
    parts.append(sections["limitations"]["markdown"])
    parts.append(sections["reference"]["markdown"])
    return "\n\n".join(parts).strip() + "\n"


def _link_citations(
    sections: dict[SectionId, SectionDraft], context: NormalizedInput
) -> dict[SectionId, SectionDraft]:
    """claim → evidence ID 연결을 섹션 생성과 분리해 확정한다."""
    linked = deepcopy(sections)
    for draft in linked.values():
        claim_evidence = [
            evidence_id
            for claim_id in draft["claim_ids"]
            if claim_id in context["claims"]
            for evidence_id in context["claims"][claim_id].get("evidence_ids", [])
        ]
        expected = _unique(list(draft["evidence_ids"]) + claim_evidence)
        present = set(extract_citations(draft["markdown"]))
        missing = [evidence_id for evidence_id in expected if evidence_id not in present]
        if missing:
            draft["markdown"] += f"\n\n〔근거: {', '.join(missing)}〕"
        draft["evidence_ids"] = expected
    return linked


def finalize_report(
    *,
    body_sections: dict[SectionId, SectionDraft],
    context: NormalizedInput,
    writer,
    deps: ReportAgentDeps,
    input_issues: list[dict],
    revisions_used: int,
    summary_override: SectionDraft | None = None,
    budgets: Budgets | None = None,
) -> dict:
    """SUMMARY 작성부터 품질 검사와 통과 판정까지 한 사이클에서 수행한다."""
    budgets = budgets or default_budgets()
    steps = ["summary", "citations", "references", "quality", "decision"]
    issues = list(input_issues)
    summary_context = _writer_context(
        "summary",
        context,
        include_normalized=_include_normalized(writer, deps),
    )
    fallback_context = _writer_context("summary", context, include_normalized=True)
    if summary_override is None:
        try:
            summary = writer.write("summary", summary_context)
        except Exception as exc:
            summary = DeterministicSectionWriter().write("summary", fallback_context)
            issues.append({
                "code": "writer_error",
                "message": f"summary 생성기 실패: {type(exc).__name__}",
                "section_id": "summary",
                "blocking": True,
            })
        if deps.on_section_written:
            deps.on_section_written("summary")
    else:
        summary = summary_override

    sections: dict[SectionId, SectionDraft] = {"summary": summary, **deepcopy(body_sections)}
    sections = _link_citations(sections, context)
    used_evidence_ids = _unique(
        evidence_id
        for section_id in SECTION_ORDER
        if section_id in sections
        for evidence_id in sections[section_id]["evidence_ids"]
    )
    references, reference_records = collect_references(
        used_evidence_ids, context["evidence_store"]
    )
    sections["reference"] = {
        "section_id": "reference",
        "title": SECTION_TITLES["reference"],
        "markdown": "# REFERENCE\n\n"
        + "\n".join([f"- {line}" for line in references] or ["자료 미확인"]),
        "claim_ids": [],
        "evidence_ids": used_evidence_ids,
    }

    markdown = _assemble(sections)
    degraded = any(
        status in {"partial", "failed"} for status in context["upstream_statuses"].values()
    )
    report = {
        "summary": sections["summary"]["markdown"].split("\n", 1)[-1].strip(),
        "sections": [deepcopy(sections[name]) for name in SECTION_ORDER if name != "reference"],
        "markdown": markdown,
        "cited_evidence_ids": used_evidence_ids,
        "references": references,
        "quality_status": "passed",
        "completion": {
            "status": "partial" if degraded else "complete",
            "revision_rounds_used": revisions_used,
            "gaps": list(context["upstream_gaps"]),
            "errors": [],
        },
    }
    for section_id in SECTION_ORDER:
        if section_id != "reference":
            issues.extend(validate_section(sections[section_id], context))
        issues.extend(validate_budget(sections[section_id], budgets[section_id]))
    issues.extend(validate_report(report, context))
    unresolved = _unique(item["message"] for item in blocking(issues))
    invalid_sections = _unique(
        item["section_id"] for item in blocking(issues) if item["section_id"] is not None
    )
    if not unresolved:
        decision = "passed"
    elif invalid_sections and revisions_used < context["max_revision_rounds"]:
        decision = "repair"
    else:
        decision = "needs_review"
    if decision != "passed":
        report["quality_status"] = "needs_review"
        report["completion"]["status"] = "partial"
        report["completion"]["errors"] = unresolved
    return {
        "decision": decision,
        "steps": steps,
        "report": report,
        "sections": sections,
        "references": reference_records,
        "issues": issues,
        "invalid_sections": invalid_sections,
    }


def _pdf_titles(context: NormalizedInput) -> tuple[str, str]:
    technology_names = [
        str(context["technologies"].get(side, {}).get("name") or "").strip()
        for side in ("sw", "hw")
    ]
    compared = " · ".join(name for name in technology_names if name)
    title = f"{compared} 기술 비교 평가 보고서" if compared else "기술 비교 평가 보고서"
    subtitle_parts = [
        context["domain"],
        f"조사 기준일 {context['as_of_date']}" if context["as_of_date"] else "",
    ]
    return title, " · ".join(part for part in subtitle_parts if part)


def _render_pages(report: dict, context: NormalizedInput, deps: ReportAgentDeps) -> dict:
    """제출용 표기(번호 인용)로 렌더링하고 장수를 센다. PDF 경로가 없으면 글자 수로 추정한다."""
    numbers = reference_numbers(report["cited_evidence_ids"], context["evidence_store"])
    display = number_citations(report["markdown"], numbers)
    if deps.pdf_output_path:
        from agents.report.pdf import export_report_pdf

        title, subtitle = _pdf_titles(context)
        path, pages = export_report_pdf(display, deps.pdf_output_path, title=title, subtitle=subtitle)
        return {"pages": pages, "method": "pdf", "display": display, "pdf_path": str(path)}
    return {"pages": estimate_pages(display), "method": "estimate", "display": display, "pdf_path": None}


def _enforce_page_limit(
    final: dict,
    *,
    body_sections: dict[SectionId, SectionDraft],
    context: NormalizedInput,
    writer,
    deps: ReportAgentDeps,
    input_issues: list[dict],
    revisions_used: int,
) -> tuple[dict, dict, list[dict]]:
    """보고서가 PAGE_LIMIT장을 넘으면 섹션을 결정적 렌더로 줄인다. LLM은 다시 부르지 않는다.

    1) 분량 상한을 넘긴 섹션을 같은 예산의 결정적 렌더로 교체한다.
    2) 넘긴 섹션이 없으면 B·C 등급 예산을 SHRINK_FACTOR만큼 줄여 다시 렌더링한다.
    교체 후보가 근거·수치 검증을 통과하지 못하면 교체하지 않는다(분량보다 정확성 우선).
    최대 MAX_PAGE_GUARD_ROUNDS회 후에도 넘으면 page_limit_exceeded를 남기고 멈춘다.
    """
    budgets = default_budgets()
    replaced: set[SectionId] = set()
    log: list[dict] = []
    summary = final["sections"]["summary"]
    rendered = _render_pages(final["report"], context, deps)
    for round_index in range(MAX_PAGE_GUARD_ROUNDS + 1):
        log.append({
            "round": round_index,
            "pages": rendered["pages"],
            "method": rendered["method"],
            "replaced": sorted(replaced),
            "shrunk": round_index > 0 and budgets != SECTION_BUDGETS,
        })
        if rendered["pages"] <= PAGE_LIMIT or round_index == MAX_PAGE_GUARD_ROUNDS:
            break
        sections = final["sections"]
        targets = [
            section_id
            for section_id in ("summary", *BODY_SECTION_ORDER)
            if section_id not in replaced
            and body_length(sections[section_id]["markdown"])
            > budgets[section_id].max_chars * OVER_BUDGET_TOLERANCE
        ]
        if not targets:
            budgets = shrink(budgets)
            targets = [sid for sid in BODY_SECTION_ORDER if budgets[sid].tier in {"B", "C"}]
        fallback = DeterministicSectionWriter(budgets)
        for section_id in targets:
            candidate = fallback.write(
                section_id,
                _writer_context(section_id, context, include_normalized=True, budgets=budgets),
            )
            if blocking(validate_section(candidate, context)):
                continue
            if section_id == "summary":
                summary = candidate
            else:
                body_sections[section_id] = candidate
            replaced.add(section_id)
        final = finalize_report(
            body_sections=body_sections,
            context=context,
            writer=writer,
            deps=deps,
            input_issues=input_issues,
            revisions_used=revisions_used,
            summary_override=summary,
            budgets=budgets,
        )
        summary = final["sections"]["summary"]
        rendered = _render_pages(final["report"], context, deps)
    if rendered["pages"] > PAGE_LIMIT:
        message = f"보고서가 {PAGE_LIMIT}장 제한을 넘음: {rendered['pages']}장"
        final["issues"].append({
            "code": "page_limit_exceeded", "message": message, "section_id": None, "blocking": True,
        })
        final["report"]["quality_status"] = "needs_review"
        final["report"]["completion"]["status"] = "partial"
        final["report"]["completion"]["errors"].append(message)
    return final, rendered, log


def run_report(state: dict, deps: ReportAgentDeps | None = None) -> dict:
    deps = deps or ReportAgentDeps()
    context = normalize_state(state)
    writer = _resolve_writer(deps)
    input_issues: list[dict] = list(validate_input(context))
    body_sections: dict[SectionId, SectionDraft] = {}

    # 사진의 본문 작성 단계: SUMMARY와 REFERENCE는 아직 만들지 않는다.
    for section_id in BODY_SECTION_ORDER:
        fallback_context = _writer_context(section_id, context, include_normalized=True)
        writer_context = _writer_context(
            section_id,
            context,
            include_normalized=_include_normalized(writer, deps),
        )
        try:
            body_sections[section_id] = writer.write(section_id, writer_context)
        except Exception as exc:
            body_sections[section_id] = DeterministicSectionWriter().write(
                section_id, fallback_context
            )
            input_issues.append({
                "code": "writer_error",
                "message": f"{section_id} 생성기 실패: {type(exc).__name__}",
                "section_id": section_id,
                "blocking": True,
            })
        if deps.on_section_written:
            deps.on_section_written(section_id)

    revisions_used = context["initial_revision_rounds"]
    repair_log: list[str] = []
    summary_override: SectionDraft | None = None
    while True:
        final = finalize_report(
            body_sections=body_sections,
            context=context,
            writer=writer,
            deps=deps,
            input_issues=input_issues,
            revisions_used=revisions_used,
            summary_override=summary_override,
        )
        if final["decision"] != "repair":
            break

        invalid_body = [
            section_id
            for section_id in final["invalid_sections"]
            if section_id in BODY_SECTION_ORDER
        ]
        targets = invalid_body or (["summary"] if "summary" in final["invalid_sections"] else [])
        if not targets:
            break
        repaired = False
        for section_id in targets:
            if revisions_used >= context["max_revision_rounds"]:
                break
            draft = final["sections"][section_id]
            section_issues = [
                item for item in final["issues"] if item["section_id"] == section_id
            ]
            writer_context = _writer_context(
                section_id,
                context,
                include_normalized=_include_normalized(writer, deps),
            )
            fallback_context = _writer_context(section_id, context, include_normalized=True)
            try:
                updated = writer.repair(section_id, draft, section_issues, writer_context)
            except Exception:
                updated = DeterministicSectionWriter().repair(
                    section_id, draft, section_issues, fallback_context
                )
            if section_id == "summary":
                summary_override = updated
            else:
                body_sections[section_id] = updated
                summary_override = None
            revisions_used += 1
            repair_log.append(section_id)
            repaired = True
        if not repaired:
            break

    final, rendered, page_guard = _enforce_page_limit(
        final,
        body_sections=body_sections,
        context=context,
        writer=writer,
        deps=deps,
        input_issues=input_issues,
        revisions_used=revisions_used,
    )
    report = final["report"]
    report["completion"]["revision_rounds_used"] = revisions_used
    report["page_count"] = rendered["pages"]
    report["page_count_method"] = rendered["method"]
    report["page_limit"] = PAGE_LIMIT
    report["page_guard"] = page_guard
    report["display_markdown"] = rendered["display"]
    if rendered["pdf_path"]:
        report["pdf_path"] = rendered["pdf_path"]
    checked_claim_ids = _unique(
        claim_id
        for section_id in SECTION_ORDER
        if section_id in final["sections"]
        for claim_id in final["sections"][section_id]["claim_ids"]
    )
    report_sections = {
        name: final["sections"][name]["markdown"] for name in SECTION_ORDER
    }
    # 제출본(report.md·PDF)은 번호 인용, 추적용 원본은 evidence ID 인용.
    report_sections["final_markdown"] = rendered["display"]
    report_sections["final_markdown_with_ids"] = report["markdown"]
    return {
        "report": report,
        "report_sections": report_sections,
        "references": final["references"],
        "issues": final["issues"],
        "repair_log": repair_log,
        "checked_claim_ids": checked_claim_ids,
        "finalization_steps": final["steps"],
        "generation": _generation_metadata(writer, deps),
    }

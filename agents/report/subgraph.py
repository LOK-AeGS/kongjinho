"""보고서 생성 흐름: 입력 정규화 → 본문 작성 → 최종화 사이클 → 부분 수정."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from agents.report.prompts import build_section_prompt
from agents.report.references import collect_references
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


class DeterministicSectionWriter:
    """상위 구조화 결과를 그대로 배치하는 API-key-free 재현용 writer."""

    receives_full_context = True

    def write(self, section_id: SectionId, context: dict) -> SectionDraft:
        return self._render(section_id, context)

    def repair(self, section_id: SectionId, draft: SectionDraft, issues: list[dict], context: dict) -> SectionDraft:
        # 외부 초안에 문제가 있어도 상위 claim만 사용하는 결정적 출력으로 해당 섹션만 교체한다.
        return self._render(section_id, context)

    def _claims(self, context: dict, perspective: str) -> list[dict]:
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

    def _claim_draft(self, section_id: SectionId, claims: list[dict], empty: str) -> SectionDraft:
        claim_ids: list[str] = []
        evidence_ids: list[str] = []
        lines: list[str] = []
        for claim in claims:
            ids = claim.get("evidence_ids", [])
            citation = f" 〔근거: {', '.join(ids)}〕" if ids else ""
            details: list[str] = []
            if claim.get("conditions"):
                details.append("조건: " + "; ".join(claim["conditions"]))
            if claim.get("uncertainty"):
                details.append("불확실성: " + claim["uncertainty"])
            suffix = f" ({' / '.join(details)})" if details else ""
            lines.append(f"- {claim['statement']}{suffix}{citation}")
            claim_ids.append(claim["claim_id"])
            evidence_ids.extend(ids)
        if not lines:
            lines = [empty]
        return {
            "section_id": section_id,
            "title": SECTION_TITLES[section_id],
            "markdown": _heading(section_id) + "\n\n" + "\n".join(lines),
            "claim_ids": _unique(claim_ids),
            "evidence_ids": _unique(evidence_ids),
        }

    def _render(self, section_id: SectionId, context: dict) -> SectionDraft:
        if section_id == "summary":
            synthesis_claims = self._claims(context, "synthesis")
            base = self._claim_draft(section_id, synthesis_claims[:5], "- 종합 결과에서 인용 가능한 핵심 문장이 확인되지 않았다.")
            relation_lines: list[str] = []
            relation_claims: list[str] = []
            relation_evidence: list[str] = []
            seen_kinds: set[str] = set()
            rows = context["synthesis"].get("cross_findings") or context["synthesis"].get("relations") or []
            for row in rows:
                kind = str(row.get("kind") or "")
                if kind not in {"agreement", "conflict"} or kind in seen_kinds:
                    continue
                claim_ids = [cid for cid in row.get("claim_ids", []) if cid in context["claims"]]
                direct_ids = [eid for eid in row.get("evidence_ids", []) if eid in context["evidence_store"]]
                evidence_ids = _unique(direct_ids + [eid for cid in claim_ids for eid in context["claims"][cid].get("evidence_ids", [])])
                if (row.get("claim_ids") and len(claim_ids) != len(row["claim_ids"])) or (row.get("evidence_ids") and len(direct_ids) != len(row["evidence_ids"])):
                    continue
                label = "주요 일치" if kind == "agreement" else "주요 상충"
                cause = f" / 원인 유형: {row['conflict_type']}" if row.get("conflict_type") else ""
                citation = f" 〔근거: {', '.join(evidence_ids)}〕" if evidence_ids else ""
                relation_lines.append(f"- {label}: {row.get('explanation') or row.get('reason') or '설명 미입력'}{cause}{citation}")
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
            return self._claim_draft(section_id, self._claims(context, "technical")[:3], "- 인용 가능한 분석 배경 자료가 확인되지 않았다.")
        if section_id == "technology_selection":
            lines = []
            for side in ("sw", "hw"):
                tech = context["technologies"].get(side, {})
                name = tech.get("name") or side
                reason = tech.get("selection_reason") or "선정 이유 미입력"
                lines.append(f"- {side.upper()}: {name} — {reason}")
            return self._plain(section_id, lines)
        if section_id == "technology_overview":
            return self._claim_draft(section_id, self._claims(context, "technical"), "- 인용 가능한 기술 개요가 확인되지 않았다.")
        if section_id == "trl":
            result = context["findings"].get("technical") or {}
            lines, evidence_ids = [], []
            records = [
                record
                for record in result.get("records", [])
                if "trl" in str(record.get("criterion") or "").casefold()
                or "성숙" in str(record.get("criterion") or "")
            ]
            for record in records:
                ids = [eid for eid in record.get("evidence_ids", []) if eid in context["evidence_store"]]
                if len(ids) != len(record.get("evidence_ids", [])):
                    continue
                value = record.get("value") or record.get("assessment") or "판단 보류"
                citation = f" 〔근거: {', '.join(ids)}〕" if ids else ""
                limitations = "; ".join(str(item) for item in record.get("limitations", []))
                suffix = f" / {limitations}" if limitations else ""
                lines.append(
                    f"- {record.get('technology', '기술')}: {record.get('criterion', 'TRL')} "
                    f"{value} — {record.get('findings', '')}{suffix}{citation}"
                )
                evidence_ids.extend(ids)
            draft = self._plain(section_id, lines or ["- 공개 정보 기반 TRL 기록이 확인되지 않았다."])
            draft["evidence_ids"] = _unique(evidence_ids)
            return draft
        if section_id in {"market", "stakeholder", "domain"}:
            return self._claim_draft(section_id, self._claims(context, section_id), f"- {SECTION_TITLES[section_id]}에 인용 가능한 결과가 확인되지 않았다.")
        if section_id == "comparison_matrix":
            return self._matrix(context)
        if section_id == "conditions":
            return self._conditions(context)
        if section_id in {"conflicts", "shared_and_complement"}:
            return self._relations(section_id, context)
        if section_id == "open_questions":
            gaps = list(context["upstream_gaps"])
            for request in context["synthesis"].get("retry_requests", []):
                gaps.append(str(request.get("reason") if isinstance(request, dict) else request))
            return self._plain(section_id, [f"- {value}" for value in _unique(gaps)] or ["- 자료 미확인"])
        if section_id == "limitations":
            lines = ["- 모든 평가는 공개 정보로 확인 가능한 범위에 한정된다."]
            for name, status in context["upstream_statuses"].items():
                if status in {"partial", "failed"}:
                    lines.append(f"- {name} upstream 상태는 {status}이며 관련 공백을 최종 판단에 반영해야 한다.")
            lines.extend(f"- {value}" for value in context["synthesis"].get("limitations", []))
            if context.get("not_found_present"):
                lines.append("- not_found는 기록된 검색 범위에서 자료를 확인하지 못한 상태이며 실제 부재를 뜻하지 않는다.")
            lines.append("- 근거 수준과 적용 조건을 함께 표시하고 반대 방향 근거의 미확인을 남겨 확증편향을 줄였다.")
            return self._plain(section_id, _unique(lines))
        raise ValueError(f"지원하지 않는 섹션: {section_id}")

    def _plain(self, section_id: SectionId, lines: list[str]) -> SectionDraft:
        return {
            "section_id": section_id,
            "title": SECTION_TITLES[section_id],
            "markdown": _heading(section_id) + "\n\n" + "\n".join(lines),
            "claim_ids": [],
            "evidence_ids": [],
        }

    def _matrix(self, context: dict) -> SectionDraft:
        cells = context["synthesis"].get("matrix") or []
        lines = ["| 관점 | 기준 | SW | HW |", "|---|---|---|---|"]
        evidence_ids: list[str] = []
        grouped: dict[tuple[str, str], dict[str, dict]] = {}
        for cell in cells:
            key = (str(cell.get("perspective") or ""), str(cell.get("criterion") or ""))
            grouped.setdefault(key, {})[str(cell.get("technology") or "")] = cell
        for (perspective, criterion), pair in grouped.items():
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
        draft = self._plain("comparison_matrix", lines)
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

    def _conditions(self, context: dict) -> SectionDraft:
        rows = context["synthesis"].get("contrast_table") or []
        lines = ["| 항목 | SW | HW |", "|---|---|---|"]
        claim_ids, evidence_ids = [], []
        for row in rows:
            ids = [cid for cid in row.get("claim_ids", []) if cid in context["claims"]]
            direct_ids = [eid for eid in row.get("evidence_ids", []) if eid in context["evidence_store"]]
            row_evidence = _unique(direct_ids + [eid for cid in ids for eid in context["claims"][cid].get("evidence_ids", [])])
            if (row.get("claim_ids") and len(ids) != len(row["claim_ids"])) or (row.get("evidence_ids") and len(direct_ids) != len(row["evidence_ids"])):
                continue
            citation = f" 〔근거: {', '.join(row_evidence)}〕" if row_evidence else ""
            label = row.get("criterion") or row.get("label") or row.get("item") or "조건"
            lines.append(f"| {label} | {row.get('sw', row.get('sw_assessment', ''))} | {row.get('hw', row.get('hw_assessment', ''))}{citation} |")
            claim_ids.extend(ids)
            evidence_ids.extend(row_evidence)
        if len(lines) == 2:
            lines.append("| 자료 미확인 | 자료 미확인 | 자료 미확인 |")
        draft = self._plain("conditions", lines)
        draft["claim_ids"], draft["evidence_ids"] = _unique(claim_ids), _unique(evidence_ids)
        return draft

    def _relations(self, section_id: SectionId, context: dict) -> SectionDraft:
        rows = context["synthesis"].get("cross_findings") or context["synthesis"].get("relations") or []
        if section_id == "conflicts":
            accepted = {"agreement", "conflict"}
        else:
            accepted = {"complement", "complementarity", "shared_evidence"}
        lines, claim_ids, evidence_ids = [], [], []
        for row in rows:
            if row.get("kind") not in accepted:
                continue
            ids = [cid for cid in row.get("claim_ids", []) if cid in context["claims"]]
            direct_ids = [eid for eid in row.get("evidence_ids", []) if eid in context["evidence_store"]]
            row_evidence = _unique(direct_ids + [eid for cid in ids for eid in context["claims"][cid].get("evidence_ids", [])])
            if (row.get("claim_ids") and len(ids) != len(row["claim_ids"])) or (row.get("evidence_ids") and len(direct_ids) != len(row["evidence_ids"])):
                continue
            citation = f" 〔근거: {', '.join(row_evidence)}〕" if row_evidence else ""
            explanation = row.get("explanation") or row.get("reason") or "설명 미입력"
            resolution = f" / 해소 상태: {row['resolution']}" if row.get("resolution") else ""
            lines.append(f"- {row.get('kind')}: {explanation}{resolution}{citation}")
            claim_ids.extend(ids)
            evidence_ids.extend(row_evidence)
        draft = self._plain(section_id, lines or ["- 구조화된 관계 기록이 확인되지 않았다."])
        draft["claim_ids"], draft["evidence_ids"] = _unique(claim_ids), _unique(evidence_ids)
        return draft


def section_payload(section_id: SectionId, context: NormalizedInput) -> dict:
    """외부 writer에 전체 evidence_store 대신 해당 섹션의 최소 입력만 제공한다."""
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
    }
    if perspective:
        payload["findings"] = context["findings"].get(perspective)
    elif section_id in {"summary", "comparison_matrix", "conditions", "conflicts", "shared_and_complement", "open_questions", "limitations"}:
        payload["synthesis"] = context["synthesis"]
        payload["upstream_statuses"] = context["upstream_statuses"]
        payload["upstream_gaps"] = context["upstream_gaps"]
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
    section_id: SectionId, context: NormalizedInput, *, include_normalized: bool
) -> dict:
    payload = section_payload(section_id, context)
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
) -> dict:
    """SUMMARY 작성부터 품질 검사와 통과 판정까지 한 사이클에서 수행한다."""
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

    report = final["report"]
    report["completion"]["revision_rounds_used"] = revisions_used
    checked_claim_ids = _unique(
        claim_id
        for section_id in SECTION_ORDER
        if section_id in final["sections"]
        for claim_id in final["sections"][section_id]["claim_ids"]
    )
    report_sections = {
        name: final["sections"][name]["markdown"] for name in SECTION_ORDER
    }
    report_sections["final_markdown"] = report["markdown"]
    if deps.pdf_output_path:
        from agents.report.pdf import export_markdown_pdf

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
        pdf_path = export_markdown_pdf(
            report["markdown"],
            deps.pdf_output_path,
            title=title,
            subtitle=" · ".join(part for part in subtitle_parts if part),
        )
        report["pdf_path"] = str(pdf_path)
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

"""기술조사 서브그래프 전용 State."""

from __future__ import annotations

from typing import TypedDict


class TechnicalLocalState(TypedDict):
    request: dict
    selected_tech: dict
    corpus_manifest: list[dict]
    candidate_evidence: list[dict]
    coverage: dict
    missing_criteria: list[str]
    search_round: int
    needs_search_retry: bool
    search_logs: list[dict]
    gaps: list[str]
    errors: list[str]
    extraction: dict
    evidence_store: dict[str, dict]
    reference_violations: list[str]
    trl_records: list[dict]
    gate_traces: dict[str, list[dict]]
    revision_round: int
    needs_revision: bool
    quality_report: dict
    technical_findings: dict
    run_meta: dict


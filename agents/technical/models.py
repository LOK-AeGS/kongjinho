"""OpenAI Structured Outputs용 엄격한 모델."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Technology = Literal["sw", "hw"]
Criterion = Literal[
    "mechanism",
    "application_scope",
    "performance",
    "limitations",
    "validation_environment",
]


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(description="입력에 제공된 candidate_id")
    quote: str = Field(description="candidate 원문에 연속해서 존재하는 짧은 직접 인용")


class DraftRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technology: Technology
    criterion: Criterion
    assessment: Literal["supported", "conditional", "unsupported", "unknown"]
    findings: str = Field(max_length=240)
    evidence_refs: list[EvidenceReference]
    conditions: list[str]
    limitations: list[str]
    source_scope: Literal["direct", "class", "mixed"]


class DraftClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technology: Technology
    text: str = Field(max_length=240)
    evidence_refs: list[EvidenceReference]
    conditions: list[str]
    limitations: list[str]


class MetricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technology: Technology
    metric: str
    value: str
    baseline: str | None
    hardware: str | None
    model: str | None
    context: str | None
    is_maximum: bool
    is_upper_bound: bool
    attributable_to_selected_technology: bool
    evidence_refs: list[EvidenceReference]


class ReadinessObservation(BaseModel):
    """LLM은 관측 사실만 구조화하고 TRL 숫자는 만들지 않는다."""

    model_config = ConfigDict(extra="forbid")
    technology: Technology
    summary: str = Field(max_length=240)
    artifact_level: Literal[
        "principle", "concept", "mechanism", "component", "system_prototype", "final_system", "unknown"
    ]
    environment: Literal["literature", "lab", "relevant", "production_equivalent", "operational", "unknown"]
    activity: Literal["analysis", "experiment", "pilot", "qualification", "deployment", "sustained_operation", "unknown"]
    evidence_level: Literal["forecast", "announcement", "pilot", "production", "unknown"]
    source_scope: Literal["direct", "class"]
    real_llm: bool
    real_accelerator: bool
    representative_workload: bool
    representative_scale: bool
    operational_requirements_validated: bool
    evidence_refs: list[EvidenceReference]
    blocking_facts: list[str]
    missing_facts: list[str]


class TechnicalExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[DraftRecord]
    claims: list[DraftClaim]
    metrics: list[MetricRecord]
    readiness_observations: list[ReadinessObservation]
    gaps: list[str]


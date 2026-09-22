from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict

TechnologyID = Literal["sw", "hw"]
Group = Literal["competitor", "operator", "supplier", "investor"]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technology_id: TechnologyID
    target_name: str
    target_scope: Literal["selected_technology", "technology_family", "other"]
    group: Group
    speaker: str
    affiliation: str | None
    stance: Literal["positive", "negative", "conditional", "neutral", "unknown"]
    evidence_stance: Literal["support", "counter", "neutral"]
    domain_relevance: Literal["datacenter", "other", "unknown"]
    statement: str
    source_url: str
    source_title: str
    source_type: Literal["paper", "patent", "official_web", "news", "community", "other"]
    published_date: str | None
    primary_or_secondary: Literal["primary", "secondary"]
    direct_or_proxy: Literal["direct", "proxy"]
    page_or_locator: str
    quote: str  # 원문 snapshot의 해당 block에서 그대로 복사
    conditions: list[str]
    uncertainty: str
    bias_notes: list[str]


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Observation]
    gaps: list[str]


class ResearchBatch(TypedDict):
    pages: dict[str, dict]
    search_logs: list[dict]
    errors: list[str]


class StakeholderState(TypedDict):
    request: dict
    technical_findings: dict | None
    queries: list[dict]
    batches: list[ResearchBatch]
    extraction: dict
    rounds: int
    revision_rounds: int
    rejected: list[str]
    gaps: list[str]
    errors: list[str]
    result: dict

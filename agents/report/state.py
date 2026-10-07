"""보고서 에이전트 내부 계약.

부모 Graph의 확정 AppState를 보고서 작성에 필요한 읽기 전용 정규형으로 투영한다.
공통 State 타입을 이 모듈에 다시 정의하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, TypedDict


SectionId = Literal[
    "summary",
    "background",
    "technology_selection",
    "technology_overview",
    "trl",
    "market",
    "stakeholder",
    "domain",
    "comparison_matrix",
    "conditions",
    "conflicts",
    "shared_and_complement",
    "open_questions",
    "limitations",
    "reference",
]


SECTION_ORDER: tuple[SectionId, ...] = (
    "summary",
    "background",
    "technology_selection",
    "technology_overview",
    "trl",
    "market",
    "stakeholder",
    "domain",
    "comparison_matrix",
    "conditions",
    "conflicts",
    "shared_and_complement",
    "open_questions",
    "limitations",
    "reference",
)

BODY_SECTION_ORDER: tuple[SectionId, ...] = tuple(
    section_id for section_id in SECTION_ORDER if section_id not in {"summary", "reference"}
)

# 확정 AppState에는 별도의 report 수정 한도 필드가 없다. 설계서의 에이전트별 최대 2회를 적용한다.
MAX_REPORT_REVISIONS = 2


class SectionDraft(TypedDict):
    section_id: SectionId
    title: str
    markdown: str
    claim_ids: list[str]
    evidence_ids: list[str]


class ValidationIssue(TypedDict):
    code: str
    message: str
    section_id: SectionId | None
    blocking: bool


class NormalizedEvidence(TypedDict, total=False):
    evidence_id: str
    document_id: str | None
    source_type: str
    title: str
    author_or_organization: str
    url: str
    published_date: str | None
    accessed_date: str
    locator: str
    excerpt: str
    stance: str
    content_hash: str


class NormalizedClaim(TypedDict, total=False):
    claim_id: str
    perspective: str
    technology_ids: list[str]
    topic: str
    statement: str
    basis: str
    evidence_ids: list[str]
    conditions: list[str]
    uncertainty: str


class NormalizedInput(TypedDict):
    as_of_date: str
    domain: str
    technologies: dict[str, dict]
    initial_revision_rounds: int
    max_revision_rounds: int
    findings: dict[str, dict | None]
    claims: dict[str, NormalizedClaim]
    evidence_store: dict[str, NormalizedEvidence]
    synthesis: dict
    upstream_statuses: dict[str, str]
    upstream_gaps: list[str]
    not_found_present: bool
    quality_feedback: list[str]
    unsupported_items: list[dict]  # 직전 품질 평가가 '인용 근거가 뒷받침하지 않음'으로 지목한 문장(재작성 때만)


class SectionWriter(Protocol):
    """기본 LLM writer와 테스트 writer가 지켜야 하는 작은 주입 경계."""

    def write(self, section_id: SectionId, context: dict) -> SectionDraft: ...

    def repair(
        self,
        section_id: SectionId,
        draft: SectionDraft,
        issues: list[ValidationIssue],
        context: dict,
    ) -> SectionDraft: ...


@dataclass(frozen=True)
class ReportAgentDeps:
    # 기본 실행은 LLM writer를 사용한다. API 없는 테스트/재현 실행만 deterministic을 명시한다.
    generation_mode: Literal["llm", "deterministic"] = "llm"
    writer: SectionWriter | None = None
    # 이미 만든 LangChain chat model을 주입할 수 있다. 없으면 model 설정으로 초기화한다.
    llm: object | None = None
    # 2026-10-07 비교(같은 State로 1회씩): gpt-4o-mini 지어낸 인용 6건·64초, gpt-5.1 2건·86초·9쪽,
    # gpt-4.1 2건·34초·7쪽 → gpt-4.1. 품질 Judge는 생성 모델과 달라야 해서 gpt-4o를 쓴다.
    model: str = "gpt-4.1"
    model_provider: str = "openai"
    temperature: float = 0.0
    on_section_written: Callable[[SectionId], None] | None = None
    # 테스트용 결정적 writer처럼 정규형 전체가 필요한 경우에만 명시적으로 연다.
    # 일반 LLM writer에는 prompt와 섹션 payload만 전달한다.
    writer_receives_full_context: bool = False
    # 지정하면 final Markdown을 PDF로 저장한다. 공통 State에는 경로 대신 run_meta로 기록한다.
    pdf_output_path: str | Path | None = None

"""KV cache 다관점 평가 State. Python 3.10+, 외부 의존성 없음.

TypedDict는 타입 계약이며 런타임 검증기가 아니다.
각 노드는 자신이 소유한 최상위 키만 부분 업데이트로 반환한다.
"""

from copy import deepcopy
from datetime import date
from typing import Literal, TypedDict

TechnologyID = Literal["sw", "hw"]
AgentID = Literal["technical", "market", "stakeholder", "domain", "synthesis", "report"]


class Technology(TypedDict):
    name: str
    selection_reason: str
    seed_urls: list[str]


class Request(TypedDict):
    sw: Technology
    hw: Technology
    domains: list[str]
    as_of_date: str  # YYYY-MM-DD
    language: str  # 예: ko
    max_search_rounds: int  # 최초 검색 포함
    max_revision_rounds: int  # 최초 작성 이후 수정 횟수


class CorpusDocument(TypedDict):
    document_id: str
    title: str
    source_url: str
    local_path: str
    total_pages: int
    indexed_pages: list[int]  # 1-based; 선정된 페이지의 총합 <= 200


class CorpusManifest(TypedDict):
    documents: list[CorpusDocument]
    embedding_model: str  # 선정 완료한 오픈소스 모델 ID
    embedding_selection_reason: str
    vector_index_uri: str
    keyword_index_uri: str


class Evidence(TypedDict):
    evidence_id: str  # 실행 내 고유 ID: technical:ev:001 등
    document_id: str | None  # 웹 검색 근거는 None 가능
    source_type: Literal["paper", "patent", "official_web", "news", "community", "other"]
    title: str
    author_or_organization: str
    url: str
    published_date: str | None  # 미확인 날짜를 임의 생성하지 않음
    accessed_date: str
    page: int | None
    section: str | None
    excerpt: str  # 판단을 뒷받침하는 짧은 원문


class Claim(TypedDict):
    claim_id: str  # technical:claim:001 등
    technology_ids: list[TechnologyID]
    topic: str  # 원리, 시장 규모, 채택, 전력, 개발자 반응 등
    statement: str
    basis: Literal["direct_evidence", "inference", "unknown"]
    evidence_ids: list[str]
    conditions: list[str]  # 모델/HW/문맥 길이/배치/평가 환경 등
    uncertainty: str  # 근거 부족·적용 범위·추론 한계


class AgentCompletion(TypedDict):
    status: Literal["complete", "partial", "failed"]
    search_rounds_used: int
    revision_rounds_used: int
    gaps: list[str]
    errors: list[str]


class Findings(TypedDict):
    claims: list[Claim]
    evidence: list[Evidence]  # 해당 에이전트가 확보한 근거
    completion: AgentCompletion


class TRLEstimate(TypedDict):
    technology_id: TechnologyID
    level: int | None  # 1~9; 근거 부족 시 None
    rationale: str
    evidence_ids: list[str]
    caveat: str  # 공개 정보 기반 추정임을 명시


class TechnicalFindings(Findings):
    trl_estimates: list[TRLEstimate]
    comparison_caveats: list[str]


class AdoptionCase(TypedDict):
    technology_id: TechnologyID
    organization: str
    stage: Literal["forecast", "announcement", "pilot", "production", "unknown"]
    claim_ids: list[str]


class MarketFindings(Findings):
    adoption_cases: list[AdoptionCase]


class StakeholderPosition(TypedDict):
    technology_id: TechnologyID
    target_name: str
    target_scope: Literal["selected_technology", "technology_family", "other"]
    group: Literal["competitor", "operator", "supplier", "investor"]
    speaker: str
    affiliation: str | None
    stance: Literal["positive", "negative", "conditional", "neutral", "unknown"]
    evidence_stance: Literal["support", "counter", "neutral"]
    claim_ids: list[str]
    bias_notes: list[str]


class StakeholderFindings(Findings):
    positions: list[StakeholderPosition]


class DomainFit(TypedDict):
    technology_id: TechnologyID
    domain: str
    requirements: list[str]
    assessment: Literal["suitable", "conditional", "unsuitable", "unknown"]
    claim_ids: list[str]
    limitations: list[str]


class DomainFindings(Findings):
    fits: list[DomainFit]


class PerspectiveRelation(TypedDict):
    kind: Literal["agreement", "conflict", "complementarity"]
    explanation: str
    claim_ids: list[str]  # 앞 단계 에이전트의 주장 ID 참조
    applicable_conditions: list[str]


class ComparisonRow(TypedDict):
    perspective: Literal["technical", "market", "stakeholder", "domain"]
    criterion: str
    sw_assessment: str
    hw_assessment: str
    claim_ids: list[str]


class Synthesis(TypedDict):
    comparison_matrix: list[ComparisonRow]
    relations: list[PerspectiveRelation]
    implications: list[Claim]  # 근거는 상위 에이전트 evidence ID 참조
    limitations: list[str]
    completion: AgentCompletion


class ReportSection(TypedDict):
    title: str
    markdown: str
    claim_ids: list[str]
    evidence_ids: list[str]


class Report(TypedDict):
    summary: str
    sections: list[ReportSection]
    markdown: str  # SUMMARY·REFERENCE 포함 전체 본문
    cited_evidence_ids: list[str]  # 실제 인용한 근거만
    references: list[str]  # 인용 근거와 연결된 형식화 참고문헌
    markdown_path: str | None
    pdf_path: str | None
    quality_status: Literal["passed", "needs_review"]
    completion: AgentCompletion


class PipelineState(TypedDict):
    run_id: str
    request: Request
    corpus_manifest: CorpusManifest
    technical_findings: TechnicalFindings | None
    market_findings: MarketFindings | None
    stakeholder_findings: StakeholderFindings | None
    domain_findings: DomainFindings | None
    synthesis: Synthesis | None
    report: Report | None


class AgentLocalState(TypedDict):
    """각 에이전트 내부 서브그래프용; 부모 State와 별도 관리."""
    agent_id: AgentID
    questions: list[str]
    retrieved_evidence: list[Evidence]
    draft_claims: list[Claim]
    search_rounds_used: int
    revision_rounds_used: int
    max_search_rounds: int
    max_revision_rounds: int
    evidence_sufficient: bool
    quality_passed: bool
    gaps: list[str]
    errors: list[str]
    next_action: Literal["search", "analyze", "revise", "finish"]


def initial_state(run_id: str, request: Request, corpus: CorpusManifest) -> PipelineState:
    """초기 상태 생성 및 입력의 핵심 제약 검증. 전체 런타임 스키마 검증은 별도."""
    if not run_id.strip():
        raise ValueError("run_id가 필요합니다.")
    date.fromisoformat(request["as_of_date"])
    if not request["domains"]:
        raise ValueError("평가 도메인이 최소 1개 필요합니다.")
    if request["max_search_rounds"] < 1 or request["max_revision_rounds"] < 0:
        raise ValueError("검색은 1회 이상, 수정 한도는 0회 이상이어야 합니다.")
    for side in ("sw", "hw"):
        if not request[side]["name"].strip() or not request[side]["selection_reason"].strip():
            raise ValueError(f"{side} 기술명과 선정 사유가 필요합니다.")
    ids = [doc["document_id"] for doc in corpus["documents"]]
    if len(ids) != len(set(ids)):
        raise ValueError("document_id는 고유해야 합니다.")
    count = 0
    for doc in corpus["documents"]:
        pages = doc["indexed_pages"]
        if doc["total_pages"] < 1 or not pages or len(pages) != len(set(pages)):
            raise ValueError("유효한 페이지 수와 중복 없는 선정 페이지 목록이 필요합니다.")
        if any(p < 1 or p > doc["total_pages"] for p in pages):
            raise ValueError("선정 페이지가 문서 범위를 벗어났습니다.")
        count += len(pages)
    if not 1 <= count <= 200:
        raise ValueError("RAG 문서는 총 1~200페이지로 구성해야 합니다.")
    return {
        "run_id": run_id,
        "request": deepcopy(request),
        "corpus_manifest": deepcopy(corpus),
        "technical_findings": None,
        "market_findings": None,
        "stakeholder_findings": None,
        "domain_findings": None,
        "synthesis": None,
        "report": None,
    }

# state.py
from __future__ import annotations

from typing import Annotated, Literal, TypeVar, TypedDict


# =========================================================
# 1. 공통 Literal 타입
# =========================================================

Technology = Literal["sw", "hw"]
Perspective = Literal[
    "technical",
    "market",
    "stakeholder",
    "domain",
]

Basis = Literal[
    "direct",
    "inferred",
    "unknown",
    "not_applicable",
]

EvidenceLevel = Literal[
    "forecast",
    "announcement",
    "pilot",
    "production",
    "unknown",
]

Scope = Literal[
    "direct",
    "class",
    "mixed",
]

Stance = Literal[
    "support",
    "counter",
    "neutral",
    "not_found",
]

CompletionStatus = Literal[
    "complete",
    "partial",
    "failed",
]


# =========================================================
# 2. 사용자 입력
# =========================================================

class RequestSpec(TypedDict):
    as_of: str
    language: Literal["ko", "en"]
    scope: str
    max_search_rounds: int


class TechSpec(TypedDict):
    name: str
    short_name: str
    technology: Technology
    approach: str
    source_ids: list[str]
    selection_reason: str


# =========================================================
# 3. 문서 및 근거
# =========================================================

class DocMeta(TypedDict):
    doc_id: str
    title: str
    revision: str | None
    source_type: str
    url: str | None
    local_path: str | None
    page_count: int | None
    sha256: str
    license: str | None


class Evidence(TypedDict):
    id: str
    claim_id: str

    doc_id: str | None
    title: str
    author_or_org: str

    source_type: str
    primary_or_secondary: Literal["primary", "secondary"]
    direct_or_proxy: Literal["direct", "proxy"]

    url: str | None
    published_at: str | None
    accessed_at: str

    page_or_locator: str
    quote: str

    stance: Stance
    evidence_level: EvidenceLevel

    metric_tag: str | None
    perspective: Perspective
    content_hash: str


# =========================================================
# 4. 주장 및 미확인 항목
# =========================================================

class Claim(TypedDict):
    claim_id: str
    technology: Technology | Literal["both"]
    perspective: Perspective

    text: str
    evidence_ids: list[str]
    conditions: list[str]
    limitations: list[str]


class Gap(TypedDict):
    technology: Technology | Literal["both"]
    perspective: Perspective
    criterion: str
    reason: str
    missing_evidence: list[str]


# =========================================================
# 5. 관점별 공통 판정 스키마
# =========================================================

class VerdictRecord(TypedDict):
    technology: Technology
    perspective: Perspective
    criterion: str

    # 모든 관점이 공유하는 축
    basis: Basis
    evidence_level: EvidenceLevel
    scope: Scope
    stance_counts: dict[str, int]
    evidence_ids: list[str]

    # 관점별 고유 판정
    assessment: str
    assessment_vocab: str
    value: str | None

    findings: str
    limitations: list[str]


class PerspectiveFindings(TypedDict):
    perspective: Perspective
    status: CompletionStatus

    records: list[VerdictRecord]
    claims: list[Claim]
    gaps: list[Gap]
    limitations: list[str]

    # TRL 입력 불일치와 근거 변경 추적
    input_evidence_ids: list[str]


# =========================================================
# 6. 평가 종합 결과
# =========================================================

class MatrixCell(TypedDict):
    technology: Technology
    perspective: Perspective
    criterion: str

    basis: Basis
    assessment: str
    assessment_vocab: str
    value: str | None

    n_evidence: int
    evidence_level: EvidenceLevel
    stance_counts: dict[str, int]
    scope: Scope


class CrossFinding(TypedDict):
    id: str

    kind: Literal[
        "agreement",
        "conflict",
        "complement",
        "shared_evidence",
    ]

    conflict_type: (
        Literal[
            "evidence_level",
            "scope",
            "condition",
            "numeric",
            "temporal",
            "stance",
            "trl_input",
        ]
        | None
    )

    technology: Technology | Literal["both"]
    record_refs: list[str]
    evidence_ids: list[str]

    detected_by: Literal["rule", "llm"]
    rule_id: str | None

    explanation: str | None
    resolution: Literal["explained", "unresolved"]


class ContrastRow(TypedDict):
    criterion: str
    sw: str | None
    hw: str | None
    relation: Literal[
        "agreement",
        "conflict",
        "complement",
        "independent",
        "unknown",
    ]
    evidence_ids: list[str]


class SynthesisResult(TypedDict):
    status: CompletionStatus
    as_of: str
    input_hash: str

    matrix: list[MatrixCell]
    cross_findings: list[CrossFinding]
    contrast_table: list[ContrastRow]
    summary_claims: list[Claim]

    gaps: list[Gap]
    imbalance: dict[str, object]
    retry_requests: list[dict[str, object]]
    limitations: list[str]

    dropped_sentences: list[dict[str, object]]
    meta: dict[str, object]


# =========================================================
# 7. 보고서 및 품질 검사
# =========================================================

class Reference(TypedDict):
    evidence_id: str
    title: str
    author_or_org: str
    published_at: str | None
    url: str | None
    locator: str


class QualityReport(TypedDict):
    status: Literal["passed", "failed", "needs_review"]
    violations: list[str]
    warnings: list[str]
    checked_claim_ids: list[str]


# =========================================================
# 8. Reducer
# =========================================================

T = TypeVar("T")


def merge_dict_right(
    current: dict[str, T] | None,
    update: dict[str, T] | None,
) -> dict[str, T]:
    """동일 키에서는 새 값을 사용한다."""
    merged = dict(current or {})
    merged.update(update or {})
    return merged


def merge_evidence_store(
    current: dict[str, Evidence] | None,
    update: dict[str, Evidence] | None,
) -> dict[str, Evidence]:
    """
    evidence_id 기준 멱등 병합.

    - 새로운 evidence_id는 추가
    - 동일 evidence_id는 기존 값을 우선 보존
    - 기존 값이 비어 있는 필드만 새 값으로 보완
    """
    merged = dict(current or {})

    for evidence_id, incoming in (update or {}).items():
        if evidence_id not in merged:
            merged[evidence_id] = incoming
            continue

        existing = merged[evidence_id]
        combined = dict(existing)

        for key, value in incoming.items():
            old_value = combined.get(key)

            if old_value in (None, "", [], {}):
                combined[key] = value

        merged[evidence_id] = combined  # type: ignore[assignment]

    return merged


# =========================================================
# 9. 최상위 Parent Graph State
# =========================================================

class AppState(TypedDict):
    # 사용자 입력
    request: RequestSpec
    selected_tech: dict[Technology, TechSpec]
    domain: str

    # RAG 준비 결과
    corpus_manifest: list[DocMeta]

    # 에이전트별 독립 결과
    technical_findings: PerspectiveFindings | None
    market_findings: PerspectiveFindings | None
    stakeholder_findings: PerspectiveFindings | None
    domain_findings: PerspectiveFindings | None

    # ①②③④가 병렬로 갱신할 수 있는 공통 근거 저장소
    evidence_store: Annotated[
        dict[str, Evidence],
        merge_evidence_store,
    ]

    # 평가 종합
    synthesis: SynthesisResult | None

    # 보고서 생성
    report_sections: Annotated[
        dict[str, str],
        merge_dict_right,
    ]
    references: dict[str, Reference]

    # 관점별 실행 기록
    search_log_by_perspective: Annotated[
        dict[str, list[dict[str, object]]],
        merge_dict_right,
    ]

    quality_by_perspective: Annotated[
        dict[str, QualityReport],
        merge_dict_right,
    ]

    retries: Annotated[
        dict[str, int],
        merge_dict_right,
    ]

    run_meta: Annotated[
        dict[str, object],
        merge_dict_right,
    ]


# =========================================================
# 10. 초기 State 생성
# =========================================================

def create_initial_state(
    *,
    request: RequestSpec,
    selected_tech: dict[Technology, TechSpec],
    corpus_manifest: list[DocMeta],
) -> AppState:
    return {
        "request": request,
        "selected_tech": selected_tech,
        "domain": "datacenter_inference",
        "corpus_manifest": corpus_manifest,

        "technical_findings": None,
        "market_findings": None,
        "stakeholder_findings": None,
        "domain_findings": None,

        "evidence_store": {},
        "synthesis": None,

        "report_sections": {},
        "references": {},

        "search_log_by_perspective": {},
        "quality_by_perspective": {},
        "retries": {
            "technical": 0,
            "market": 0,
            "stakeholder": 0,
            "domain": 0,
            "synthesis": 0,
            "report": 0,
        },
        "run_meta": {},
    }
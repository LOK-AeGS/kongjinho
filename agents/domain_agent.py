"""도메인 평가 에이전트: 데이터센터 관점에서 SW/HW 기술의 적합 조건을 평가한다.

부모 그래프에는 `domain_findings` 키 하나만 반환한다(STATE_DESIGN.md 업데이트 규칙).
내부 흐름은 outputs/agent-architecture/mermaid/06-domain-evaluation.mmd 를 따른다:
  질문 생성 -> RAG 검색 -> 적용 조건 검토 -> 근거 충분성 판단 -(부족)-> 질의 보완 -> 재검색
                                                         -(충분/한도)-> 적합성 분석 -> 사실·추론 분리

종합 에이전트로는 산문이 아니라 구조화 레코드를 넘긴다. 이유와 제약은 docs/DOMAIN_AGENT.md 참조.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from prompts.domain import (
    ANALYSIS_SYSTEM,
    ANALYSIS_USER,
    DATACENTER_REQUIREMENTS,
    DOMAIN_NAME,
    QUESTION_SYSTEM,
    QUESTION_USER,
    REFINE_HINT,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
    format_requirements,
)
from state import (
    AgentCompletion,
    AgentLocalState,
    Claim,
    DomainFindings,
    DomainFit,
    Evidence,
    PipelineState,
)

AGENT_ID = "domain"

# 주장 1건은 한 문장으로 제한한다. 길어지면 논점이 섞여 종합 단계에서 분해가 불가능하다.
MAX_STATEMENT_CHARS = 240
MAX_QUESTIONS_PER_ROUND = 6
HITS_PER_QUESTION = 3

# 커뮤니티 출처만으로는 사실 주장을 세우지 않는다(확증편향·출처 신뢰도 관리).
WEAK_SOURCE_TYPES = {"community", "other"}


@dataclass
class SearchHit:
    """검색 결과 1건. state.Evidence로 변환되기 전의 중간 표현이다.

    출처 메타데이터(문서 ID·페이지·URL)를 여기서 확보하지 못하면
    보고서 REFERENCE 단계에서 복원할 방법이 없다.
    """

    title: str
    url: str
    excerpt: str
    source_type: Literal["paper", "patent", "official_web", "news", "community", "other"]
    author_or_organization: str
    document_id: str | None = None
    page: int | None = None
    section: str | None = None
    published_date: str | None = None
    accessed_date: str = field(default_factory=lambda: date.today().isoformat())


class Retriever(Protocol):
    """이 에이전트가 요구하는 검색 계약.

    인덱스 구축(문서 파싱·청크·임베딩)은 팀 공유 RAG 준비 단계가 담당하며,
    이 프로토콜만 만족하면 어떤 구현이든 주입할 수 있다.
    """

    def search(self, query: str, k: int = 4) -> list[SearchHit]: ...


class QuestionPlan(BaseModel):
    """검색 질문 묶음. 기대 효과와 제약을 짝으로 만들게 강제한다."""

    benefit_questions: list[str] = Field(description="기대 효과를 확인하는 질문")
    limitation_questions: list[str] = Field(description="제약·실패 조건을 확인하는 질문")


class SufficiencyVerdict(BaseModel):
    sufficient: bool = Field(description="조건부로라도 판단 가능한지")
    missing_axes: list[str] = Field(description="근거가 비어 있는 요구사항 축")


class DraftClaim(BaseModel):
    # 최종 claim_id는 검증 단계가 부여한다. 모델은 fits에서 되짚을 수 있게 자기 키만 붙인다.
    # 순번(1,2,3…)으로 참조시켰더니 모델이 자리를 세다 어긋나 sw 판정이 hw 주장을 가리켰다.
    claim_key: str = Field(description="이 주장의 짧은 식별자. 예: sw-memory-1")
    technology_ids: list[Literal["sw", "hw"]]
    topic: str
    statement: str = Field(description=f"한 문장, {MAX_STATEMENT_CHARS}자 이내")
    basis: Literal["direct_evidence", "inference", "unknown"]
    evidence_ids: list[str] = Field(description="제공된 근거 목록에 실재하는 ID만")
    conditions: list[str] = Field(description="측정·적용 전제(모델 크기, 문맥 길이, HW 등)")
    uncertainty: str


class DraftFit(BaseModel):
    technology_id: Literal["sw", "hw"]
    requirements: list[str]
    assessment: Literal["suitable", "conditional", "unsuitable", "unknown"]
    claim_keys: list[str] = Field(
        description="이 판정을 뒷받침하는 claims의 claim_key. 같은 technology_id의 주장만 적는다"
    )
    limitations: list[str]


class DomainAnalysis(BaseModel):
    claims: list[DraftClaim]
    fits: list[DraftFit]


class DomainLocalState(AgentLocalState):
    """팀 AgentLocalState에 도메인 평가 전용 입력·중간 결과를 더한 서브그래프 상태."""

    sw_name: str
    hw_name: str
    technical_summary: str
    condition_notes: list[str]
    fits: list[DomainFit]


@dataclass
class DomainAgentDeps:
    """State 밖에서 관리하는 런타임 의존성(STATE_DESIGN.md 저장 경계)."""

    llm: object
    retriever: Retriever


def _evidence_key(hit: SearchHit) -> tuple:
    """중복 판정 키.

    URL·페이지만으로는 같은 페이지에서 나온 서로 다른 청크가 하나로 합쳐져 근거가 소실된다.
    발췌 앞부분까지 포함해야 문서 검색에서도 청크 단위로 구분된다.
    """
    return (hit.url, hit.page, hit.excerpt[:120])


def _to_evidence(hit: SearchHit, index: int) -> Evidence:
    return {
        "evidence_id": f"{AGENT_ID}:ev:{index:03d}",
        "document_id": hit.document_id,
        "source_type": hit.source_type,
        "title": hit.title,
        "author_or_organization": hit.author_or_organization,
        "url": hit.url,
        "published_date": hit.published_date,
        "accessed_date": hit.accessed_date,
        "page": hit.page,
        "section": hit.section,
        "excerpt": hit.excerpt,
    }


def _digest(evidence: list[Evidence], excerpt_chars: int = 400) -> str:
    """LLM에 넘길 근거 요약. 원문 전체가 아니라 ID와 짧은 발췌만 넘겨 희석을 막는다."""
    lines = []
    for ev in evidence:
        where = f" p.{ev['page']}" if ev["page"] else ""
        lines.append(
            f"[{ev['evidence_id']}] ({ev['source_type']}) {ev['title']}{where}\n"
            f"  {ev['excerpt'][:excerpt_chars]}"
        )
    return "\n".join(lines) if lines else "(수집된 근거 없음)"


def summarize_technical(state: PipelineState) -> str:
    """기술 조사 결과를 압축한다. 선행 에이전트 산문을 그대로 넘기면 내용이 희석된다."""
    findings = state.get("technical_findings")
    if not findings:
        return "(기술 조사 결과 없음 - 도메인 평가는 자체 검색 근거만으로 수행)"
    lines = []
    for claim in findings.get("claims", [])[:12]:
        techs = "/".join(claim["technology_ids"])
        lines.append(f"- ({techs}) {claim['topic']}: {claim['statement']}")
    for trl in findings.get("trl_estimates", []):
        level = trl["level"] if trl["level"] is not None else "판단보류"
        lines.append(f"- TRL({trl['technology_id']}): {level} / {trl['caveat']}")
    return "\n".join(lines) if lines else "(기술 조사 주장 없음)"


def _plan_questions(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """요구사항 축마다 기대 효과와 제약을 짝으로 묻는 검색 질문을 만든다."""
    refine = ""
    if state["gaps"]:
        refine = REFINE_HINT.format(gaps="\n".join(f"- {g}" for g in state["gaps"]))
    planner = deps.llm.with_structured_output(QuestionPlan)
    plan = planner.invoke(
        [
            ("system", QUESTION_SYSTEM),
            (
                "human",
                QUESTION_USER.format(
                    domain=DOMAIN_NAME,
                    requirements=format_requirements(),
                    sw_name=state["sw_name"],
                    hw_name=state["hw_name"],
                    technical_summary=state["technical_summary"],
                    refine_hint=refine,
                    n_questions=MAX_QUESTIONS_PER_ROUND,
                ),
            ),
        ]
    )
    # 한쪽만 생성되면 검색이 편향되므로 두 묶음을 교차로 배치해 검색 예산을 균등 배분한다.
    paired: list[str] = []
    for benefit, limitation in zip(plan.benefit_questions, plan.limitation_questions):
        paired.extend([benefit, limitation])
    leftover = plan.benefit_questions[len(plan.limitation_questions) :] + (
        plan.limitation_questions[len(plan.benefit_questions) :]
    )
    questions = (paired + leftover)[:MAX_QUESTIONS_PER_ROUND]
    return {"questions": questions, "next_action": "search"}


def _retrieve(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """주입된 검색 도구로 근거를 모은다. 같은 출처는 재검색해도 같은 ID를 유지한다."""
    known = {
        (ev["url"], ev["page"], ev["excerpt"][:120]) for ev in state["retrieved_evidence"]
    }
    collected = list(state["retrieved_evidence"])
    errors = list(state["errors"])

    for question in state["questions"]:
        try:
            hits = deps.retriever.search(question, k=HITS_PER_QUESTION)
        except Exception as exc:  # 검색 실패가 그래프를 중단시키지 않게 한다
            errors.append(f"검색 실패({question[:40]}): {exc}")
            continue
        for hit in hits:
            key = _evidence_key(hit)
            if key in known:
                continue
            known.add(key)
            collected.append(_to_evidence(hit, len(collected) + 1))

    return {
        "retrieved_evidence": collected,
        "search_rounds_used": state["search_rounds_used"] + 1,
        "errors": errors,
    }


def _review_conditions(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """실험 환경과 데이터센터 운영 환경의 차이를 뽑아둔다.

    논문 측정값을 그대로 도메인 적합성 근거로 쓰지 않기 위한 단계다.
    """
    if not state["retrieved_evidence"]:
        return {"condition_notes": []}

    class ConditionNotes(BaseModel):
        notes: list[str] = Field(description="측정 전제와 데이터센터 환경의 차이")

    reviewer = deps.llm.with_structured_output(ConditionNotes)
    result = reviewer.invoke(
        [
            (
                "system",
                "수집된 근거에서 측정·실험 전제(모델 크기, 문맥 길이, 배치, GPU, 연구용 셋업 여부)를 찾아"
                " 데이터센터 운영 환경과 다른 점만 간결히 적으세요. 근거에 없으면 추측하지 마세요.",
            ),
            ("human", _digest(state["retrieved_evidence"])),
        ]
    )
    return {"condition_notes": result.notes}


def _check_sufficiency(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """요구사항 축이 비었는지, 한쪽 방향 근거만 모였는지 확인한다."""
    if not state["retrieved_evidence"]:
        return {
            "evidence_sufficient": False,
            "gaps": ["수집된 근거가 없어 모든 요구사항 축을 판단할 수 없음"],
        }
    judge = deps.llm.with_structured_output(SufficiencyVerdict)
    verdict = judge.invoke(
        [
            ("system", SUFFICIENCY_SYSTEM),
            (
                "human",
                SUFFICIENCY_USER.format(
                    requirements=format_requirements(),
                    evidence_digest=_digest(state["retrieved_evidence"]),
                ),
            ),
        ]
    )
    return {
        "evidence_sufficient": verdict.sufficient,
        "gaps": [] if verdict.sufficient else verdict.missing_axes,
    }


def _route_after_check(state: DomainLocalState) -> str:
    """근거가 부족하고 검색 예산이 남았을 때만 재검색한다(무한 루프 방지)."""
    if state["evidence_sufficient"]:
        return "analyze"
    if state["search_rounds_used"] < state["max_search_rounds"]:
        return "plan_questions"
    return "analyze"


def _analyze(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """조건별 적합성을 구조화 출력으로 받는다. 이 결과가 종합 에이전트의 입력이 된다."""
    if not state["retrieved_evidence"]:
        return {"draft_claims": [], "fits": [], "quality_passed": False, "next_action": "finish"}

    analyst = deps.llm.with_structured_output(DomainAnalysis)
    condition_hint = "\n".join(f"- {n}" for n in state["condition_notes"])
    analysis = analyst.invoke(
        [
            ("system", ANALYSIS_SYSTEM.format(max_chars=MAX_STATEMENT_CHARS)),
            (
                "human",
                ANALYSIS_USER.format(
                    domain=DOMAIN_NAME,
                    requirements=format_requirements(),
                    sw_name=state["sw_name"],
                    hw_name=state["hw_name"],
                    technical_summary=state["technical_summary"],
                    evidence_digest=_digest(state["retrieved_evidence"])
                    + (f"\n\n[실험 환경과 데이터센터의 차이]\n{condition_hint}" if condition_hint else ""),
                ),
            ),
        ]
    )
    claims, fits, gaps = _validate(analysis, state["retrieved_evidence"], state["condition_notes"])
    return {
        "draft_claims": claims,
        "fits": fits,
        "gaps": state["gaps"] + gaps,
        "quality_passed": bool(claims),
        "next_action": "finish",
    }


def _validate(
    analysis: DomainAnalysis, evidence: list[Evidence], condition_notes: list[str]
) -> tuple[list[Claim], list[DomainFit], list[str]]:
    """근거 추적 계약을 코드로 검증한다. TypedDict는 런타임 검증을 하지 않는다.

    STATE_DESIGN.md의 검사 항목: direct_evidence는 유효 근거 참조, 추론은 전제·한계 명시.
    """
    by_id = {ev["evidence_id"]: ev for ev in evidence}
    gaps: list[str] = []
    claims: list[Claim] = []
    claim_id_by_key: dict[str, str] = {}
    techs_by_key: dict[str, list[str]] = {}

    for draft in analysis.claims:
        valid_ids = [eid for eid in draft.evidence_ids if eid in by_id]
        basis = draft.basis

        # 존재하지 않는 근거를 인용하면 사실 주장으로 인정하지 않는다.
        if basis == "direct_evidence" and not valid_ids:
            basis = "unknown"
            gaps.append(f"근거 없이 사실로 제시된 주장을 판단 보류로 내림: {draft.statement[:60]}")
        # 약한 출처만 있는 경우도 마찬가지로 승격하지 않는다.
        elif basis == "direct_evidence" and all(
            by_id[eid]["source_type"] in WEAK_SOURCE_TYPES for eid in valid_ids
        ):
            basis = "inference"
            gaps.append(f"공신력 낮은 출처만 확보되어 추론으로 낮춤: {draft.statement[:60]}")

        uncertainty = draft.uncertainty.strip()
        conditions = [c for c in draft.conditions if c.strip()]
        if basis == "inference":
            if not conditions:
                conditions = condition_notes[:2] or ["적용 전제가 근거에 명시되지 않음"]
            if not uncertainty:
                uncertainty = "근거에서 유추한 판단으로 적용 범위가 제한됨"
        if basis == "unknown" and not uncertainty:
            uncertainty = "판단을 뒷받침할 근거를 확보하지 못함"

        claim_id = f"{AGENT_ID}:claim:{len(claims) + 1:03d}"
        claim_id_by_key[draft.claim_key] = claim_id
        techs_by_key[draft.claim_key] = draft.technology_ids
        claims.append(
            {
                "claim_id": claim_id,
                "technology_ids": draft.technology_ids,
                "topic": draft.topic,
                "statement": draft.statement.strip()[:MAX_STATEMENT_CHARS],
                "basis": basis,
                "evidence_ids": valid_ids,
                "conditions": conditions,
                "uncertainty": uncertainty,
            }
        )

    fits: list[DomainFit] = []
    for draft_fit in analysis.fits:
        linked = []
        for key in draft_fit.claim_keys:
            if key not in claim_id_by_key:
                continue
            # 기술이 어긋난 연결은 끊는다. sw 판정이 hw 주장을 가리키는 교차 참조가 실제로 나왔고,
            # ID만 유효하면 통과하는 검사로는 이 오류가 드러나지 않는다.
            if draft_fit.technology_id not in techs_by_key[key]:
                gaps.append(
                    f"{draft_fit.technology_id} 판정이 다른 기술의 주장({key})을 참조해 연결 해제"
                )
                continue
            linked.append(claim_id_by_key[key])
        assessment = draft_fit.assessment
        # 뒷받침 주장이 하나도 없으면 적합/부적합 판정을 유지하지 않는다.
        if not linked and assessment != "unknown":
            assessment = "unknown"
            gaps.append(f"연결된 주장이 없어 {draft_fit.technology_id} 판정을 보류로 내림")
        fits.append(
            {
                "technology_id": draft_fit.technology_id,
                "domain": DOMAIN_NAME,
                "requirements": draft_fit.requirements or list(DATACENTER_REQUIREMENTS),
                "assessment": assessment,
                "claim_ids": linked,
                "limitations": draft_fit.limitations,
            }
        )

    evaluated = {f["technology_id"] for f in fits}
    for missing in {"sw", "hw"} - evaluated:
        gaps.append(f"{missing} 기술에 대한 도메인 판단을 생성하지 못함")

    return claims, fits, gaps


def _build_subgraph(deps: DomainAgentDeps):
    """mermaid 06-domain-evaluation 흐름을 그대로 노드로 옮긴다."""
    graph = StateGraph(DomainLocalState)
    graph.add_node("plan_questions", lambda s: _plan_questions(s, deps))
    graph.add_node("retrieve", lambda s: _retrieve(s, deps))
    graph.add_node("review_conditions", lambda s: _review_conditions(s, deps))
    graph.add_node("check_sufficiency", lambda s: _check_sufficiency(s, deps))
    graph.add_node("analyze", lambda s: _analyze(s, deps))

    graph.add_edge(START, "plan_questions")
    graph.add_edge("plan_questions", "retrieve")
    graph.add_edge("retrieve", "review_conditions")
    graph.add_edge("review_conditions", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency", _route_after_check, ["plan_questions", "analyze"]
    )
    graph.add_edge("analyze", END)
    return graph.compile()


def build_domain_agent(deps: DomainAgentDeps):
    """graph_wiring.build_graph(domain=...)에 주입할 노드 함수를 만든다."""

    subgraph = _build_subgraph(deps)

    def domain_node(state: PipelineState) -> dict:
        request = state["request"]
        local: DomainLocalState = {
            "agent_id": AGENT_ID,
            "questions": [],
            "retrieved_evidence": [],
            "draft_claims": [],
            "search_rounds_used": 0,
            "revision_rounds_used": 0,
            "max_search_rounds": request["max_search_rounds"],
            "max_revision_rounds": request["max_revision_rounds"],
            "evidence_sufficient": False,
            "quality_passed": False,
            "gaps": [],
            "errors": [],
            "next_action": "search",
            "sw_name": request["sw"]["name"],
            "hw_name": request["hw"]["name"],
            "technical_summary": summarize_technical(state),
            "condition_notes": [],
            "fits": [],
        }

        try:
            result = subgraph.invoke(local)
        except Exception as exc:
            # 복구해 진행하는 경우에만 failed로 남긴다(STATE_DESIGN.md).
            return {
                "domain_findings": _findings(
                    claims=[], evidence=[], fits=[], status="failed",
                    search_rounds=0, gaps=[], errors=[f"도메인 평가 서브그래프 실패: {exc}"],
                )
            }

        status: Literal["complete", "partial", "failed"]
        if result["quality_passed"] and not result["gaps"]:
            status = "complete"
        elif result["quality_passed"]:
            status = "partial"
        else:
            status = "failed"

        return {
            "domain_findings": _findings(
                claims=result["draft_claims"],
                evidence=result["retrieved_evidence"],
                fits=result["fits"],
                status=status,
                search_rounds=result["search_rounds_used"],
                gaps=result["gaps"],
                errors=result["errors"],
            )
        }

    return domain_node


def _findings(
    *,
    claims: list[Claim],
    evidence: list[Evidence],
    fits: list[DomainFit],
    status: Literal["complete", "partial", "failed"],
    search_rounds: int,
    gaps: list[str],
    errors: list[str],
) -> DomainFindings:
    completion: AgentCompletion = {
        "status": status,
        "search_rounds_used": search_rounds,
        "revision_rounds_used": 0,
        "gaps": gaps,
        "errors": errors,
    }
    return {
        "claims": claims,
        "evidence": evidence,
        "completion": completion,
        "fits": fits,
    }


def make_deps(retriever: Retriever, model: str = "gpt-4o-mini") -> DomainAgentDeps:
    """팀 공유 검색 도구를 주입해 실행 구성을 만든다. LLM은 GPT 고정."""
    from langchain.chat_models import init_chat_model

    return DomainAgentDeps(
        llm=init_chat_model(model, model_provider="openai", temperature=0),
        retriever=retriever,
    )


__all__ = [
    "AGENT_ID",
    "DomainAgentDeps",
    "Retriever",
    "SearchHit",
    "build_domain_agent",
    "make_deps",
    "summarize_technical",
]

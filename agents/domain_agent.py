"""도메인 평가 에이전트: 웹 검색 기반 RAG로 데이터센터 관점의 적합 조건을 평가한다.

검색 대상은 팀 문서 풀이 아니라 공신력 있는 웹 출처(논문·표준·벤더 공식·주요 매체)다.
스니펫만으로는 적용 조건을 알 수 없어 본문을 가져오고, 길이에 따라 처리가 갈린다.
짧은 문서는 그대로 근거가 되고, 긴 문서(논문 등)만 청킹·색인해 필요한 부분을 뽑는다.
임베딩이 쓰이는 지점은 후자 하나뿐이다.

품질 검사는 두 층으로 나뉜다. 인용 대조·수치·날짜·참조 무결성은 결정적 guard 가 맡고,
coverage 와 neutrality 만 LLM judge 가 본다. 표현 린터가 근거 없는 승자·추천·압도 표현을 막는다.

부모 그래프에는 domain_findings 와 도메인 전용 부속 키만 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from prompts.domain import (
    ANALYSIS_SYSTEM,
    ANALYSIS_USER,
    DATACENTER_REQUIREMENTS,
    DOMAIN_NAME,
    PROMPT_VERSION,
    QUESTION_SYSTEM,
    QUESTION_USER,
    REFINE_HINT,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
    format_requirements,
)
from quality.guard import GuardReport, run_guard
from quality.judge import JudgeResult, run_judge
from quality.linter import LintReport, lint_claims
from rag.evidence import Evidence, SearchLogEntry, merge_evidence, normalize_text
from rag.fetch import fetch_document, pick_quote
from rag.index import Chunk, build_index, chunk_parts

AGENT_ID = "domain"
PERSPECTIVE = "domain"

MAX_STATEMENT_CHARS = 240
MAX_QUESTIONS_PER_ROUND = 6
RESULTS_PER_QUESTION = 4
CHUNKS_PER_QUESTION = 3
# 라운드당 본문 수집 상한. 수집·파싱·임베딩이 문서 수에 비례해 늘어나므로 묶어 둔다.
MAX_DOCUMENTS_PER_ROUND = 10
# 과제 제약. 웹 문서는 사전에 페이지를 고를 수 없어 수집하면서 누적으로 막는다.
PAGE_BUDGET = 200

WEAK_SOURCE_TYPES = {"news", "other"}


class QuestionPlan(BaseModel):
    benefit_questions: list[str] = Field(description="기대 효과를 확인하는 질문")
    limitation_questions: list[str] = Field(description="제약·실패 조건을 확인하는 질문")


class SufficiencyVerdict(BaseModel):
    sufficient: bool = Field(description="조건부로라도 판단 가능한지")
    missing_axes: list[str] = Field(description="근거가 비어 있는 요구사항 축")


class DraftClaim(BaseModel):
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
    claim_keys: list[str] = Field(description="같은 technology_id 주장의 claim_key 만")
    limitations: list[str]


class DomainAnalysis(BaseModel):
    claims: list[DraftClaim]
    fits: list[DraftFit]


class DomainLocalState(TypedDict):
    """도메인 관점 서브그래프 상태. 다른 관점의 중간 결론은 들어오지 않는다."""

    sw_name: str
    hw_name: str
    technical_summary: str
    as_of_date: str
    questions: list[str]
    evidence_store: dict[str, dict]  # evidence_id -> Evidence dict (멱등 병합)
    source_texts: dict[str, str]  # url -> 정규화 원문 (인용 대조용)
    search_log: list[dict]
    pages_used: int
    condition_notes: list[str]
    claims: list[dict]
    fits: list[dict]
    search_rounds_used: int
    max_search_rounds: int
    evidence_sufficient: bool
    gaps: list[str]
    errors: list[str]


@dataclass
class DomainAgentDeps:
    """State 밖 런타임 의존성. JSON 직렬화 대상이 아니다."""

    llm: object
    search_provider: object  # search(query) -> (list[SearchResult], SearchLogEntry)
    embedding_model: str | None = None  # None 이면 긴 문서도 BM25 단독으로 처리
    fetch_cache_dir: Path | None = None
    embedding_run_info: dict | None = field(default=None, init=False)


def project_input(state: dict) -> dict:
    """부모 State에서 이 관점이 볼 것만 추린다.

    시장·이해관계자 관점의 중간 결론을 같이 넘기면 도메인 판단이 그쪽 결론에 물든다.
    선행 단계인 기술 조사 결과와 요청 정보만 통과시킨다.
    """
    request = state["request"]
    return {
        "sw_name": request["sw"]["name"],
        "hw_name": request["hw"]["name"],
        "as_of_date": request["as_of_date"],
        "max_search_rounds": request["max_search_rounds"],
        "technical_summary": summarize_technical(state.get("technical_findings")),
    }


def summarize_technical(findings: dict | None) -> str:
    """선행 결과를 압축한다. 산문을 그대로 넘기면 내용이 희석된다."""
    if not findings:
        return "(기술 조사 결과 없음 - 자체 검색 근거만으로 평가)"
    lines = [
        f"- ({'/'.join(c['technology_ids'])}) {c['topic']}: {c['statement']}"
        for c in findings.get("claims", [])[:12]
    ]
    for trl in findings.get("trl_estimates", []):
        level = trl["level"] if trl["level"] is not None else "판단보류"
        lines.append(f"- TRL({trl['technology_id']}): {level} / {trl['caveat']}")
    return "\n".join(lines) or "(기술 조사 주장 없음)"


def _digest(
    evidence_store: dict[str, dict], excerpt_chars: int = 400
) -> tuple[str, dict[str, str]]:
    """LLM에 넘길 근거 요약과 표시 라벨 매핑을 만든다.

    실제 evidence_id 는 출처 신원 해시(domain:ev:100e7ab6d6e1)라 모델이 그대로 옮겨 적지
    못한다. 실측에서 이 형태로 넘겼더니 주장 6건 전부가 근거를 인용하지 못했다.
    그래서 E1, E2 같은 짧은 라벨로 보여주고 여기서 실제 ID로 되돌린다.
    """
    lines: list[str] = []
    label_to_id: dict[str, str] = {}
    for position, ev in enumerate(evidence_store.values(), start=1):
        label = f"E{position}"
        label_to_id[label] = ev["evidence_id"]
        lines.append(
            f"[{label}] ({ev['source_type']}) {ev['title'][:70]} / {ev['locator']}\n"
            f"  {ev['quote'][:excerpt_chars]}"
        )
    return ("\n".join(lines) if lines else "(수집된 근거 없음)"), label_to_id


def _plan_questions(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """요구사항 축마다 기대 효과와 제약을 짝으로 묻게 해 검색 편향을 막는다."""
    refine = (
        REFINE_HINT.format(gaps="\n".join(f"- {g}" for g in state["gaps"]))
        if state["gaps"]
        else ""
    )
    plan = deps.llm.with_structured_output(QuestionPlan).invoke(
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
    paired: list[str] = []
    for benefit, limitation in zip(plan.benefit_questions, plan.limitation_questions):
        paired.extend([benefit, limitation])
    leftover = (
        plan.benefit_questions[len(plan.limitation_questions) :]
        + plan.limitation_questions[len(plan.benefit_questions) :]
    )
    return {"questions": (paired + leftover)[:MAX_QUESTIONS_PER_ROUND]}


def _evidence_from_chunk(chunk: Chunk, source_type: str, published: str | None,
                         checksum: str, *, query: str) -> Evidence:
    return Evidence.build(
        source_type=source_type,
        title=chunk.title,
        author_or_organization=chunk.url.split("/")[2] if "//" in chunk.url else chunk.url,
        url=chunk.url,
        quote=pick_quote(chunk.text),
        locator=chunk.locator,
        retrieved_via="fetched_document",
        published_date=published,
        content_sha256=checksum,
        search_query=query,
    )


def _retrieve(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """웹 검색 → 본문 수집 → (긴 문서만) 색인 → 근거 생성.

    긴 문서를 문서마다 따로 색인하면 임베딩 패스가 문서 수만큼 반복돼 크게 느려진다.
    그래서 이 라운드에서 모은 긴 문서 청크를 한 번에 색인하고 질문별로 상위를 고른다.
    같은 출처를 다시 만나도 evidence_id 가 같아 저장소 병합은 멱등하다.
    """
    store: dict[str, dict] = {}
    source_texts = dict(state["source_texts"])
    logs: list[dict] = []
    errors = list(state["errors"])
    pages_used = state["pages_used"]
    seen_urls = set(source_texts)

    # 1) 질문별로 검색해 후보 URL을 모은다(중복 URL은 처음 만난 질문에 귀속).
    candidates: list[tuple[str, object]] = []  # (question, SearchResult)
    for question in state["questions"]:
        try:
            results, log = deps.search_provider.search(question)
        except Exception as exc:
            errors.append(f"검색 실패({question[:40]}): {type(exc).__name__}: {exc}")
            continue
        logs.append(log.__dict__ if isinstance(log, SearchLogEntry) else dict(log))
        for result in results[:RESULTS_PER_QUESTION]:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            candidates.append((question, result))

    # 2) 본문을 가져온다. 페이지 예산과 문서 수 상한을 여기서 건다.
    long_chunks: list[Chunk] = []
    meta_by_url: dict[str, tuple[str, str | None, str]] = {}  # url -> (source_type, published, sha)
    for question, result in candidates[:MAX_DOCUMENTS_PER_ROUND]:
        if pages_used >= PAGE_BUDGET:
            errors.append(f"200페이지 한도 도달로 수집 중단 (현재 {pages_used}p)")
            break
        document = fetch_document(result.url, cache_dir=deps.fetch_cache_dir)
        if document.error:
            errors.append(f"본문 수집 실패({result.url[:60]}): {document.error}")
            continue

        # 한도를 넘기는 문서는 통째로 버린다. 앞서 "넘으면 중단"으로 두었더니 209/200 이 됐다.
        if pages_used + document.page_count > PAGE_BUDGET:
            errors.append(
                f"200페이지 한도 초과로 제외({result.url[:50]}, {document.page_count}p)"
            )
            continue
        pages_used += document.page_count
        source_texts[result.url] = normalize_text(document.full_text)
        meta_by_url[result.url] = (
            result.source_type, result.published_date, document.content_sha256
        )
        chunks = chunk_parts(document.parts, result.url, document.title or result.title)
        if not chunks:
            continue

        if document.is_long:
            long_chunks.extend(chunks)  # 색인은 아래에서 한 번만
        else:
            # 짧은 문서는 색인할 이유가 없다. 첫 조각을 그대로 근거로 쓴다.
            evidence = _evidence_from_chunk(chunks[0], *meta_by_url[result.url], query=question)
            store[evidence.evidence_id] = evidence.to_dict()

    # 3) 긴 문서 청크를 통합 색인해 질문별 상위를 고른다.
    if long_chunks:
        for position, chunk in enumerate(long_chunks):
            chunk.chunk_index = position
        index = build_index(long_chunks, embedding_model=deps.embedding_model)
        if index.dense is not None and deps.embedding_run_info is None:
            deps.embedding_run_info = index.dense.run_info.to_dict()
        for question in state["questions"]:
            for chunk, _score in index.rank(question, CHUNKS_PER_QUESTION):
                meta = meta_by_url.get(chunk.url)
                if meta is None:
                    continue
                evidence = _evidence_from_chunk(chunk, *meta, query=question)
                store[evidence.evidence_id] = evidence.to_dict()

    return {
        "evidence_store": merge_evidence(state["evidence_store"], store),
        "source_texts": source_texts,
        "search_log": state["search_log"] + logs,
        "pages_used": pages_used,
        "search_rounds_used": state["search_rounds_used"] + 1,
        "errors": errors,
    }


def _review_conditions(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """논문 측정값을 그대로 도메인 근거로 쓰지 않도록 실험 전제를 먼저 뽑아둔다."""
    if not state["evidence_store"]:
        return {"condition_notes": []}

    class ConditionNotes(BaseModel):
        notes: list[str] = Field(description="측정 전제와 데이터센터 운영 환경의 차이")

    try:
        result = deps.llm.with_structured_output(ConditionNotes).invoke(
            [
                (
                    "system",
                    "수집된 근거에서 측정·실험 전제(모델 크기, 문맥 길이, 배치, GPU, 연구용 셋업 여부)를"
                    " 찾아 데이터센터 운영 환경과 다른 점만 간결히 적으세요. 근거에 없으면 추측하지 마세요.",
                ),
                ("human", _digest(state["evidence_store"])[0]),
            ]
        )
        return {"condition_notes": result.notes}
    except Exception as exc:
        return {"condition_notes": [], "errors": state["errors"] + [f"조건 검토 실패: {exc}"]}


def _check_sufficiency(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    if not state["evidence_store"]:
        return {
            "evidence_sufficient": False,
            "gaps": ["수집된 근거가 없어 모든 요구사항 축을 판단할 수 없음"],
        }
    try:
        verdict = deps.llm.with_structured_output(SufficiencyVerdict).invoke(
            [
                ("system", SUFFICIENCY_SYSTEM),
                (
                    "human",
                    SUFFICIENCY_USER.format(
                        requirements=format_requirements(),
                        evidence_digest=_digest(state["evidence_store"])[0],
                    ),
                ),
            ]
        )
    except Exception as exc:
        return {"evidence_sufficient": True, "errors": state["errors"] + [f"충분성 판단 실패: {exc}"]}
    return {
        "evidence_sufficient": verdict.sufficient,
        "gaps": [] if verdict.sufficient else verdict.missing_axes,
    }


def _route_after_check(state: DomainLocalState) -> str:
    if state["evidence_sufficient"]:
        return "analyze"
    if state["search_rounds_used"] < state["max_search_rounds"]:
        return "plan_questions"
    return "analyze"


def _analyze(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    if not state["evidence_store"]:
        return {"claims": [], "fits": []}
    condition_hint = "\n".join(f"- {n}" for n in state["condition_notes"])
    digest, label_to_id = _digest(state["evidence_store"])
    try:
        analysis = deps.llm.with_structured_output(DomainAnalysis).invoke(
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
                        evidence_digest=digest
                        + (
                            f"\n\n[실험 환경과 데이터센터의 차이]\n{condition_hint}"
                            if condition_hint
                            else ""
                        ),
                    ),
                ),
            ]
        )
    except Exception as exc:
        return {"claims": [], "fits": [], "errors": state["errors"] + [f"분석 실패: {exc}"]}

    claims, fits, gaps = _shape(
        analysis, state["evidence_store"], state["condition_notes"], label_to_id
    )
    return {"claims": claims, "fits": fits, "gaps": state["gaps"] + gaps}


def _shape(
    analysis: DomainAnalysis,
    evidence_store: dict[str, dict],
    condition_notes: list[str],
    label_to_id: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """모델 출력을 최종 스키마로 옮기면서 근거 강도를 조정한다.

    여기서 하는 일은 강등뿐이고, 사실 검증은 guard 가 따로 한다.
    """
    gaps: list[str] = []
    claims: list[dict] = []
    id_by_key: dict[str, str] = {}
    techs_by_key: dict[str, list[str]] = {}

    labels = label_to_id or {}
    for draft in analysis.claims:
        # 모델은 E1 같은 라벨로 인용한다. 실제 ID 를 그대로 적은 경우도 함께 받아준다.
        resolved = [labels.get(token.strip(), token.strip()) for token in draft.evidence_ids]
        valid = [eid for eid in dict.fromkeys(resolved) if eid in evidence_store]
        basis = draft.basis
        if basis == "direct_evidence" and not valid:
            basis = "unknown"
            gaps.append(f"근거 없이 사실로 제시된 주장을 판단 보류로 내림: {draft.statement[:50]}")
        elif basis == "direct_evidence" and all(
            evidence_store[eid]["source_type"] in WEAK_SOURCE_TYPES for eid in valid
        ):
            basis = "inference"
            gaps.append(f"매체 보도만 근거여서 추론으로 낮춤: {draft.statement[:50]}")

        conditions = [c for c in draft.conditions if c.strip()]
        uncertainty = draft.uncertainty.strip()
        if basis == "inference":
            conditions = conditions or condition_notes[:2] or ["적용 전제가 근거에 명시되지 않음"]
            uncertainty = uncertainty or "근거에서 유추한 판단으로 적용 범위가 제한됨"
        if basis == "unknown":
            uncertainty = uncertainty or "판단을 뒷받침할 근거를 확보하지 못함"

        claim_id = f"{AGENT_ID}:claim:{len(claims) + 1:03d}"
        id_by_key[draft.claim_key] = claim_id
        techs_by_key[draft.claim_key] = draft.technology_ids
        claims.append(
            {
                "claim_id": claim_id,
                "technology_ids": draft.technology_ids,
                "topic": draft.topic,
                "statement": draft.statement.strip()[:MAX_STATEMENT_CHARS],
                "basis": basis,
                "evidence_ids": valid,
                "conditions": conditions,
                "uncertainty": uncertainty,
            }
        )

    fits: list[dict] = []
    for draft_fit in analysis.fits:
        linked = []
        for key in draft_fit.claim_keys:
            if key not in id_by_key:
                continue
            if draft_fit.technology_id not in techs_by_key[key]:
                gaps.append(f"{draft_fit.technology_id} 판정이 다른 기술 주장({key})을 참조해 연결 해제")
                continue
            linked.append(id_by_key[key])
        assessment = draft_fit.assessment
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

    for missing in {"sw", "hw"} - {f["technology_id"] for f in fits}:
        gaps.append(f"{missing} 기술에 대한 도메인 판단을 생성하지 못함")
    return claims, fits, gaps


def _apply_quality(
    state: DomainLocalState, deps: DomainAgentDeps
) -> tuple[list[dict], list[dict], GuardReport, LintReport, JudgeResult, list[str]]:
    """guard(결정적) → linter(표현) → judge(질적) 순서로 적용한다."""
    gaps = list(state["gaps"])
    guard = run_guard(
        claims=state["claims"],
        evidence_store=state["evidence_store"],
        source_texts=state["source_texts"],
        as_of_date=state["as_of_date"],
    )
    # 주장 자체에 걸린 위반과, 그 주장이 인용한 근거에 걸린 위반을 모두 본다.
    # 근거가 원문 대조에 실패했는데 그것을 인용한 주장이 사실로 남으면 검사의 의미가 없다.
    claim_level = {"citation", "numeric_unit", "reference_integrity"}
    evidence_level = {"quote", "locator", "date"}
    flagged_claims = {v.target_id for v in guard.violations if v.check in claim_level}
    flagged_evidence = {v.target_id for v in guard.violations if v.check in evidence_level}

    claims = []
    for claim in state["claims"]:
        tainted = [eid for eid in claim["evidence_ids"] if eid in flagged_evidence]
        if claim["basis"] == "direct_evidence" and (claim["claim_id"] in flagged_claims or tainted):
            reason = (
                "결정적 검사에서 근거 대조에 실패함"
                if claim["claim_id"] in flagged_claims
                else f"인용한 근거가 검사에 실패함({tainted[:2]})"
            )
            claim = {**claim, "basis": "inference"}
            claim["uncertainty"] = claim["uncertainty"] or reason
            gaps.append(f"{claim['claim_id']}: guard 위반으로 추론으로 낮춤 - {reason}")
        claims.append(claim)

    lint = lint_claims(claims)
    blocked = lint.blocking_ids
    if blocked:
        gaps.extend(f"{cid}: 평가 표현으로 차단됨" for cid in sorted(blocked))
    claims = [c for c in claims if c["claim_id"] not in blocked]

    kept_ids = {c["claim_id"] for c in claims}
    fits = [
        {**f, "claim_ids": [cid for cid in f["claim_ids"] if cid in kept_ids]}
        for f in state["fits"]
    ]
    for fit in fits:
        if not fit["claim_ids"] and fit["assessment"] != "unknown":
            fit["assessment"] = "unknown"
            gaps.append(f"{fit['technology_id']}: 남은 주장이 없어 판정 보류")

    judge = run_judge(
        deps.llm,
        requirements=format_requirements(),
        claims=claims,
        fits=fits,
        model_name=getattr(deps.llm, "model_name", "unknown"),
    )
    gaps.extend(f"coverage 미달 축: {axis}" for axis in judge.missing_axes)
    return claims, fits, guard, lint, judge, gaps


def _build_subgraph(deps: DomainAgentDeps):
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
    graph.add_conditional_edges("check_sufficiency", _route_after_check, ["plan_questions", "analyze"])
    graph.add_edge("analyze", END)
    return graph.compile()


def build_domain_agent(deps: DomainAgentDeps):
    """graph_wiring.build_graph(domain=...) 에 주입할 노드 함수를 만든다."""
    subgraph = _build_subgraph(deps)

    def domain_node(state: dict) -> dict:
        projected = project_input(state)
        local: DomainLocalState = {
            **projected,
            "questions": [],
            "evidence_store": {},
            "source_texts": {},
            "search_log": [],
            "pages_used": 0,
            "condition_notes": [],
            "claims": [],
            "fits": [],
            "search_rounds_used": 0,
            "evidence_sufficient": False,
            "gaps": [],
            "errors": [],
        }
        try:
            result = subgraph.invoke(local)
        except Exception as exc:
            return _output(
                claims=[], fits=[], evidence_store={}, search_log=[], pages_used=0,
                guard=None, lint=None, judge=None, search_rounds=0, gaps=[],
                errors=[f"도메인 평가 서브그래프 실패: {type(exc).__name__}: {exc}"],
                status="failed",
            )

        claims, fits, guard, lint, judge, gaps = _apply_quality(result, deps)
        if not claims:
            status = "failed"
        elif gaps or not guard.passed or not judge.passed:
            status = "partial"
        else:
            status = "complete"

        return _output(
            claims=claims, fits=fits, evidence_store=result["evidence_store"],
            search_log=result["search_log"], pages_used=result["pages_used"],
            guard=guard, lint=lint, judge=judge,
            search_rounds=result["search_rounds_used"], gaps=gaps, errors=result["errors"],
            status=status,
        )

    return domain_node


def _output(*, claims, fits, evidence_store, search_log, pages_used, guard, lint, judge,
            search_rounds, gaps, errors, status) -> dict:
    """부모 State 업데이트. 관점별 키로 분리해 병렬 분기에서 충돌하지 않게 한다."""
    return {
        "domain_findings": {
            "claims": claims,
            "fits": fits,
            "cited_evidence_ids": sorted({eid for c in claims for eid in c["evidence_ids"]}),
            "completion": {
                "status": status,
                "search_rounds_used": search_rounds,
                "revision_rounds_used": 0,
                "pages_used": pages_used,
                "gaps": gaps,
                "errors": errors,
            },
        },
        # evidence_store 는 관점 간 공유 채널이라 dict 병합 리듀서로 합친다.
        "evidence_store": evidence_store,
        "quality_by_perspective": {
            PERSPECTIVE: {
                "guard": guard.to_dict() if guard else None,
                "lint": lint.to_dict() if lint else None,
                "judge": judge.to_dict() if judge else None,
                "prompt_version": PROMPT_VERSION,
            }
        },
        "search_log_by_perspective": {PERSPECTIVE: search_log},
    }


__all__ = [
    "AGENT_ID",
    "PAGE_BUDGET",
    "DomainAgentDeps",
    "DomainAnalysis",
    "DraftClaim",
    "DraftFit",
    "build_domain_agent",
    "project_input",
    "summarize_technical",
]

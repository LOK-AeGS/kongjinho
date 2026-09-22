"""도메인 평가 에이전트: 웹 검색 기반 RAG로 데이터센터 관점의 적합 조건을 평가한다.

검색 대상은 팀 문서 풀이 아니라 공신력 있는 웹 출처(논문·표준·벤더 공식·주요 매체)다.
스니펫만으로는 적용 조건을 알 수 없어 본문을 가져오고, 길이에 따라 처리가 갈린다.
짧은 문서는 그대로 근거가 되고, 긴 문서(논문 등)만 청킹·색인해 필요한 부분을 뽑는다.
임베딩이 쓰이는 지점은 후자 하나뿐이다.

품질 검사는 결정적 코드(guard/linter)와 별도 LLM judge로 나누지 않고, analyze 노드
하나의 프롬프트에 흡수했다(v3, PROMPT_VERSION 참고). 남긴 코드 검증은 참조 무결성
(존재하지 않는 근거 라벨 제거) 하나뿐이다 — 이건 품질 판단이 아니라 없으면
evidence_store 조회에서 KeyError로 죽는 방어적 처리라서 남겼다.

부모 State 와의 변환(노드 함수)은 node.py 에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agents.domain.prompts import (
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
from agents.domain.tools.evidence import Evidence, SearchLogEntry, merge_evidence, normalize_text
from agents.domain.tools.fetch import fetch_document, pick_quote
from agents.domain.tools.index import Chunk, build_index, chunk_parts

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

Basis = Literal["direct", "inferred", "unknown", "not_applicable"]


class QuestionPlan(BaseModel):
    benefit_questions: list[str] = Field(description="기대 효과를 확인하는 질문")
    limitation_questions: list[str] = Field(description="제약·실패 조건을 확인하는 질문")


class SufficiencyVerdict(BaseModel):
    sufficient: bool = Field(description="조건부로라도 판단 가능한지")
    missing_axes: list[str] = Field(description="근거가 비어 있는 요구사항 축")


class DraftClaim(BaseModel):
    claim_key: str = Field(description="이 주장의 짧은 식별자. 예: sw-memory-1")
    technology_ids: list[Literal["sw", "hw"]]
    text: str = Field(description=f"한 문장, {MAX_STATEMENT_CHARS}자 이내")
    basis: Basis
    evidence_ids: list[str] = Field(description="제공된 근거 목록의 라벨(E1, E2, ...)만")
    conditions: list[str] = Field(description="측정·적용 전제(모델 크기, 문맥 길이, HW 등)")
    limitations: list[str] = Field(description="이 주장의 한계나 불확실성")


class DraftRecord(BaseModel):
    """요구사항 축 하나 × 기술 하나의 판정. 팀 공통 VerdictRecord 형태에 맞춘다."""

    technology_id: Literal["sw", "hw"]
    criterion: str = Field(description="요구사항 축 문장 그대로")
    basis: Basis
    evidence_level: Literal["forecast", "announcement", "pilot", "production", "unknown"]
    scope: Literal["direct", "class", "mixed"]
    assessment: Literal["suitable", "conditional", "unsuitable", "unknown"]
    assessment_vocab: str = "domain_fit_v1"
    value: str | None = Field(default=None, description="관련 수치, 원문 그대로. 없으면 null")
    findings: str = Field(description=f"판정 근거 요약, {MAX_STATEMENT_CHARS}자 이내")
    limitations: list[str]
    claim_keys: list[str] = Field(description="같은 technology_id 주장의 claim_key 만")


class SelfCheck(BaseModel):
    """guard/linter/judge를 대체하는 자체 점검. node.py 에서 QualityReport로 변환된다."""

    status: Literal["passed", "failed", "needs_review"]
    violations: list[str] = Field(description="근거 없는 인용, 인용문에 없는 수치, 평가적 표현")
    warnings: list[str] = Field(description="sw/hw 서술 불균형 등 차단하지 않는 경고")
    covered_axes: list[str] = Field(description="실제로 근거를 찾아 다룬 요구사항 축")
    missing_axes: list[str] = Field(description="근거가 없어 판단하지 못한 요구사항 축")


class DomainAnalysis(BaseModel):
    claims: list[DraftClaim]
    records: list[DraftRecord]
    self_check: SelfCheck


class DomainLocalState(TypedDict):
    """도메인 관점 서브그래프 상태. 다른 관점의 중간 결론은 들어오지 않는다."""

    sw_name: str
    hw_name: str
    technical_summary: str
    as_of_date: str
    questions: list[str]
    evidence_store: dict[str, dict]  # evidence_id -> Evidence dict (멱등 병합)
    source_texts: dict[str, str]  # url -> 정규화 원문 (참조용, 인용 대조 코드는 더는 없음)
    search_log: list[dict]
    pages_used: int
    condition_notes: list[str]
    claims: list[dict]
    records: list[dict]
    self_check: dict
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


def _digest(
    evidence_store: dict[str, dict], excerpt_chars: int = 400
) -> tuple[str, dict[str, str]]:
    """LLM에 넘길 근거 요약과 표시 라벨 매핑을 만든다.

    실제 evidence_id 는 출처 신원 해시(domain:ev:100e7ab6d6e1)라 모델이 그대로 옮겨 적지
    못한다. 실측에서 이 형태로 넘겼더니 주장 전부가 근거를 인용하지 못했다.
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


_EMPTY_SELF_CHECK = {
    "status": "failed",
    "violations": [],
    "warnings": [],
    "covered_axes": [],
    "missing_axes": list(DATACENTER_REQUIREMENTS),
}


def _analyze(state: DomainLocalState, deps: DomainAgentDeps) -> dict:
    """판정·주장·자체 품질 점검을 한 번의 구조화 출력으로 받는다(PROMPT_VERSION v3)."""
    if not state["evidence_store"]:
        return {"claims": [], "records": [], "self_check": _EMPTY_SELF_CHECK}

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
        return {
            "claims": [], "records": [], "self_check": _EMPTY_SELF_CHECK,
            "errors": state["errors"] + [f"분석 실패: {type(exc).__name__}: {exc}"],
        }

    claims, records, gaps = _shape(analysis, state["evidence_store"], label_to_id)
    self_check = analysis.self_check.model_dump()
    return {
        "claims": claims,
        "records": records,
        "self_check": self_check,
        "gaps": state["gaps"] + gaps + list(self_check.get("missing_axes", [])),
    }


def _shape(
    analysis: DomainAnalysis,
    evidence_store: dict[str, dict],
    label_to_id: dict[str, str],
) -> tuple[list[dict], list[dict], list[str]]:
    """모델 출력을 최종 스키마로 옮긴다. 여기서 하는 일은 참조 무결성 확인뿐이다.

    존재하지 않는 라벨을 조용히 걸러내지 않으면 evidence_store 조회에서 KeyError로
    죽는다. 사실 판단(근거 충분성·표현 적절성)은 모델의 self_check가 맡는다.
    """
    gaps: list[str] = []
    claims: list[dict] = []
    id_by_key: dict[str, str] = {}
    techs_by_key: dict[str, list[str]] = {}

    for draft in analysis.claims:
        # 모델은 E1 같은 라벨로 인용한다. 실제 ID를 그대로 적은 경우도 함께 받아준다.
        resolved = [label_to_id.get(token.strip(), token.strip()) for token in draft.evidence_ids]
        valid = [eid for eid in dict.fromkeys(resolved) if eid in evidence_store]
        if len(valid) < len(set(resolved)):
            gaps.append(f"{draft.claim_key}: 존재하지 않는 근거 참조를 제거함")

        claim_id = f"{AGENT_ID}:claim:{len(claims) + 1:03d}"
        id_by_key[draft.claim_key] = claim_id
        techs_by_key[draft.claim_key] = draft.technology_ids
        claims.append(
            {
                "claim_id": claim_id,
                "technology_ids": draft.technology_ids,
                "text": draft.text.strip()[:MAX_STATEMENT_CHARS],
                "basis": draft.basis,
                "evidence_ids": valid,
                "conditions": [c for c in draft.conditions if c.strip()],
                "limitations": [lim for lim in draft.limitations if lim.strip()],
            }
        )
    claim_by_id = {c["claim_id"]: c for c in claims}

    records: list[dict] = []
    for draft in analysis.records:
        linked_claims: list[str] = []
        for key in draft.claim_keys:
            if key not in id_by_key:
                continue
            if draft.technology_id not in techs_by_key[key]:
                gaps.append(
                    f"{draft.technology_id}/{draft.criterion[:30]}: "
                    f"다른 기술 주장({key})을 참조해 연결 해제"
                )
                continue
            linked_claims.append(id_by_key[key])

        linked_evidence: list[str] = []
        for cid in linked_claims:
            linked_evidence.extend(claim_by_id[cid]["evidence_ids"])

        assessment = draft.assessment
        if not linked_claims and assessment not in ("unknown",):
            assessment = "unknown"
            gaps.append(
                f"{draft.technology_id}/{draft.criterion[:30]}: 연결된 주장이 없어 판정을 보류로 내림"
            )

        records.append(
            {
                "technology_id": draft.technology_id,
                "criterion": draft.criterion,
                "basis": draft.basis,
                "evidence_level": draft.evidence_level,
                "scope": draft.scope,
                "assessment": assessment,
                "assessment_vocab": draft.assessment_vocab,
                "value": draft.value,
                "findings": draft.findings.strip()[:MAX_STATEMENT_CHARS],
                "limitations": [lim for lim in draft.limitations if lim.strip()],
                "claim_ids": linked_claims,
                "evidence_ids": sorted(set(linked_evidence)),
            }
        )

    for missing in {"sw", "hw"} - {r["technology_id"] for r in records}:
        gaps.append(f"{missing} 기술에 대한 판정 레코드를 생성하지 못함")
    return claims, records, gaps


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


__all__ = [
    "AGENT_ID",
    "PERSPECTIVE",
    "PAGE_BUDGET",
    "DomainAgentDeps",
    "DomainAnalysis",
    "DomainLocalState",
    "DraftClaim",
    "DraftRecord",
    "SelfCheck",
    "_build_subgraph",
]

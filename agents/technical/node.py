"""부모 AppState와 기술조사 서브그래프 사이의 단일 어댑터."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache, partial

from .backend import OpenAIAnalyzer
from .config import DEFAULT_REQUEST, DEFAULT_SELECTED_TECH, fixed_input, validate_fixed_input, validate_output_contract
from .corpus import default_source_dir, load_manifest, parse_corpus, validate_corpus
from .retrieval import BGEEmbedder, HybridRetriever
from .subgraph import TechnicalAgentDeps, run_technical
from .web import TavilyProvider


@lru_cache(maxsize=1)
def default_deps() -> tuple[TechnicalAgentDeps, list[dict]]:
    documents = load_manifest()
    validate_corpus(documents)
    chunks = parse_corpus(documents)
    retriever = HybridRetriever(chunks, embedder=BGEEmbedder())
    deps = TechnicalAgentDeps(
        retriever=retriever,
        web_provider=TavilyProvider(cache_dir=default_source_dir().parent / "web_cache"),
        analyzer=OpenAIAnalyzer(),
    )
    manifest = [document.to_parent_meta(default_source_dir()) for document in documents]
    return deps, manifest


def _fixed_parent_input(state: dict) -> tuple[dict, dict]:
    as_of = (state.get("request") or {}).get("as_of", DEFAULT_REQUEST["as_of"])
    request, selected = fixed_input(as_of=as_of)
    supplied = state.get("selected_tech")
    if supplied is not None and supplied != DEFAULT_SELECTED_TECH:
        raise ValueError("기술조사 입력 기술은 DeepSeek-V2 MLA와 ITME로 고정됩니다.")
    validate_fixed_input(request, selected)
    return request, selected


def technical_agent(state: dict, *, deps: TechnicalAgentDeps | None = None, corpus_manifest: list[dict] | None = None) -> dict:
    """부모에는 자기 소유 키인 technical_findings와 evidence_store만 반환한다."""
    try:
        request, selected = _fixed_parent_input(state)
        if deps is None:
            deps, default_manifest = default_deps()
        else:
            default_manifest = []
        final = run_technical(
            request=request,
            selected_tech=selected,
            corpus_manifest=deepcopy(corpus_manifest if corpus_manifest is not None else default_manifest),
            deps=deps,
        )
        findings = deepcopy(final["technical_findings"])
        findings.setdefault("meta", {}).update({
            "search_rounds_used": final["search_round"],
            "revision_rounds_used": final["revision_round"],
            "search_logs": final["search_logs"],
            "quality_report": final["quality_report"],
            "gate_traces": final["gate_traces"],
            "retriever": getattr(deps.retriever, "run_info", {}),
            "analyzer": getattr(deps.analyzer, "run_info", {}),
        })
        result = {"technical_findings": findings, "evidence_store": final["evidence_store"]}
        validate_output_contract(result)
        return result
    except Exception as exc:
        failure = {
            "perspective": "technical", "status": "failed", "records": [], "claims": [],
            "gaps": [{"technology": "both", "perspective": "technical", "criterion": "execution", "reason": f"기술조사 실행 실패 ({type(exc).__name__})", "missing_evidence": ["실행 환경과 입력을 점검해야 함"]}],
            "limitations": [f"기술조사 실행 실패 ({type(exc).__name__})"],
            "input_evidence_ids": [], "meta": {"quality_report": {"status": "failed", "violations": [type(exc).__name__], "warnings": [], "checked_claim_ids": []}},
        }
        result = {"technical_findings": failure, "evidence_store": {}}
        validate_output_contract(result)
        return result


def make_node(deps: TechnicalAgentDeps | None = None, *, corpus_manifest: list[dict] | None = None):
    return partial(technical_agent, deps=deps, corpus_manifest=corpus_manifest)

"""기술조사 에이전트의 고정 입력과 실행 기본값."""

from __future__ import annotations

from copy import deepcopy
from datetime import date


DEFAULT_REQUEST = {
    "as_of": "2026-09-22",
    "language": "ko",
    "scope": "datacenter_inference",
    "max_search_rounds": 2,
}

DEFAULT_SELECTED_TECH = {
    "sw": {
        "name": "DeepSeek-V2 Multi-head Latent Attention",
        "short_name": "MLA",
        "technology": "sw",
        "approach": "model_architecture_kv_compression",
        "source_ids": ["arxiv:2405.04434v5"],
        "selection_reason": "어텐션 구조에서 KV를 저차원 잠재 표현으로 압축",
    },
    "hw": {
        "name": "ITME: Inference Tiered Memory Expansion",
        "short_name": "ITME",
        "technology": "hw",
        "approach": "cxl_hybrid_tiered_memory",
        "source_ids": ["arxiv:2606.12556v2"],
        "selection_reason": "CXL-hybrid 계층으로 데이터센터 KV 상태 저장 공간 확장",
    },
}

MAX_REVISION_ROUNDS = 1
TOP_K_PER_QUERY = 4

OUTPUT_CONTRACT = {
    "top_level": {"technical_findings", "evidence_store"},
    "technical_findings": {"perspective", "status", "records", "claims", "gaps", "limitations", "input_evidence_ids", "meta"},
    "record": {"technology", "perspective", "criterion", "basis", "evidence_level", "scope", "stance_counts", "evidence_ids", "assessment", "assessment_vocab", "value", "findings", "limitations"},
    "claim": {"claim_id", "technology", "perspective", "text", "evidence_ids", "conditions", "limitations"},
    "gap": {"technology", "perspective", "criterion", "reason", "missing_evidence"},
    "evidence": {"id", "claim_id", "doc_id", "title", "author_or_org", "source_type", "primary_or_secondary", "direct_or_proxy", "url", "published_at", "accessed_at", "page_or_locator", "quote", "stance", "evidence_level", "metric_tag", "perspective", "content_hash"},
}


def fixed_input(*, as_of: str | None = None) -> tuple[dict, dict]:
    request = deepcopy(DEFAULT_REQUEST)
    request["as_of"] = as_of or DEFAULT_REQUEST["as_of"] or date.today().isoformat()
    return request, deepcopy(DEFAULT_SELECTED_TECH)


def validate_fixed_input(request: dict, selected_tech: dict) -> None:
    if request.get("scope") != "datacenter_inference":
        raise ValueError("기술조사 범위는 datacenter_inference로 고정됩니다.")
    date.fromisoformat(request["as_of"])
    rounds = request.get("max_search_rounds")
    if not isinstance(rounds, int) or not 1 <= rounds <= 2:
        raise ValueError("max_search_rounds는 1~2여야 합니다.")
    if set(selected_tech) != {"sw", "hw"}:
        raise ValueError("selected_tech에는 sw와 hw가 정확히 한 건씩 있어야 합니다.")
    if selected_tech["sw"].get("short_name") != "MLA":
        raise ValueError("SW 판정 단위는 DeepSeek-V2 MLA로 고정됩니다.")
    if selected_tech["hw"].get("short_name") != "ITME":
        raise ValueError("HW 판정 단위는 ITME 시스템 프로토타입으로 고정됩니다.")


def validate_output_contract(result: dict) -> None:
    if set(result) != OUTPUT_CONTRACT["top_level"]:
        raise ValueError("기술조사 최상위 출력 계약이 변경됐습니다.")
    findings = result["technical_findings"]
    if set(findings) != OUTPUT_CONTRACT["technical_findings"]:
        raise ValueError("technical_findings 출력 계약이 변경됐습니다.")
    for name in ("record", "claim", "gap"):
        key = {"record": "records", "claim": "claims", "gap": "gaps"}[name]
        if any(set(item) != OUTPUT_CONTRACT[name] for item in findings[key]):
            raise ValueError(f"{name} 출력 계약이 변경됐습니다.")
    if any(set(item) != OUTPUT_CONTRACT["evidence"] for item in result["evidence_store"].values()):
        raise ValueError("evidence 출력 계약이 변경됐습니다.")

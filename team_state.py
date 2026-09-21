"""v0.3 팀 계약. 같은 근거는 idempotent reducer로 합친다."""
from copy import deepcopy
from operator import add
from typing import Annotated, Literal, TypedDict


def merge_evidence(left: dict, right: dict) -> dict:
    result = deepcopy(left)
    for key, incoming in right.items():
        if incoming['id'] != key:
            raise ValueError('evidence key/id mismatch')
        old = result.get(key)
        if old:
            for field in ('url', 'page_or_locator', 'quote'):
                if old[field] != incoming[field]:
                    raise ValueError('evidence identity collision')
            # 같은 quote에 여러 claim/관점이 연결될 수 있다. latest + stable tie-break.
            import json
            chosen = max((old, incoming), key=lambda e: (e['accessed_at'], json.dumps(
                {k: v for k, v in e.items() if k not in ('claim_ids', 'perspectives', 'bindings')},
                sort_keys=True, ensure_ascii=False)))
            merged = deepcopy(chosen)
            for field, singular in (('claim_ids', 'claim_id'), ('perspectives', 'perspective')):
                merged[field] = sorted(set(old.get(field, [old[singular]])) | set(incoming.get(field, [incoming[singular]])))
            merged['bindings'] = sorted({json.dumps(b, sort_keys=True, ensure_ascii=False) for b in old.get('bindings', []) + incoming.get('bindings', [])})
            merged['bindings'] = [json.loads(b) for b in merged['bindings']]
            result[key] = merged
        else:
            result[key] = deepcopy(incoming)
    return result


class TechSpec(TypedDict):
    name: str
    selection_reason: str
    seed_urls: list[str]


class SharedEvidence(TypedDict):
    id: str
    claim_id: str
    doc_id: str | None
    title: str
    author_or_org: str
    source_type: str
    primary_or_secondary: Literal['primary', 'secondary']
    direct_or_proxy: Literal['direct', 'proxy']
    url: str
    published_at: str | None
    accessed_at: str
    page_or_locator: str
    quote: str
    stance: Literal['support', 'counter', 'neutral', 'not_found']
    perspective: str
    content_hash: str
    claim_ids: list[str]
    perspectives: list[str]
    bindings: list[dict]  # claim별 stance/perspective 보존
    snapshot_path: str | None


class PerspectiveEval(TypedDict):
    perspective: str
    findings: list[dict]
    evidence_ids: list[str]
    limitations: list[str]
    confidence: float  # 직접 근거가 있는 기술/그룹 비율. 정확도 확률 아님.
    completion: dict
    search_outcomes: list[dict]  # not_found는 사실 근거가 아닌 검색 결과 기록


class EvaluationState(TypedDict, total=False):
    selected_tech: dict[str, TechSpec]
    domain: str
    tech_profiles: dict
    evidence_store: Annotated[dict[str, SharedEvidence], merge_evidence]
    trl_eval: PerspectiveEval | None
    market_eval: PerspectiveEval | None
    stakeholder_eval: PerspectiveEval | None
    domain_eval: PerspectiveEval | None
    quality_by_perspective: dict[str, dict]
    failed_perspectives: list[str]
    retries: dict[str, int]
    conflicts: list[dict]
    agreements: list[str]
    report_sections: dict[str, str]
    references: dict[str, dict]
    errors: Annotated[list[str], add]
    as_of_date: str

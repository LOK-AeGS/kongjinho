"""이해관계자 내부 근거 병합. 원문 검증 후 근거 풀(pool)을 합칠 때 쓴다.

원래 graph/team_state.py(v0.3)에 있던 reducer를 공통 State 교체(graph/state.py AppState)에 맞춰 옮겼다.
부모 State의 evidence_store reducer는 graph.state.merge_evidence_store 이다.
"""
import json
from copy import deepcopy


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

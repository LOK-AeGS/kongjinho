"""원문 검증을 통과한 이해관계자 근거의 멱등 병합."""

import json
from copy import deepcopy


def merge_evidence(left: dict, right: dict) -> dict:
    result = deepcopy(left)
    for key, incoming in right.items():
        if incoming["id"] != key:
            raise ValueError("evidence key/id mismatch")
        old = result.get(key)
        if old:
            for field in ("url", "page_or_locator", "quote"):
                if old[field] != incoming[field]:
                    raise ValueError("evidence identity collision")
            chosen = max((old, incoming), key=lambda evidence: (
                evidence["accessed_at"],
                json.dumps(
                    {k: v for k, v in evidence.items() if k not in ("claim_ids", "perspectives", "bindings")},
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            ))
            merged = deepcopy(chosen)
            for field, singular in (("claim_ids", "claim_id"), ("perspectives", "perspective")):
                merged[field] = sorted(
                    set(old.get(field, [old[singular]]))
                    | set(incoming.get(field, [incoming[singular]]))
                )
            encoded = {
                json.dumps(binding, sort_keys=True, ensure_ascii=False)
                for binding in old.get("bindings", []) + incoming.get("bindings", [])
            }
            merged["bindings"] = [json.loads(binding) for binding in sorted(encoded)]
            result[key] = merged
        else:
            result[key] = deepcopy(incoming)
    return result

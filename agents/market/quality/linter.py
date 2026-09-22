"""표현 린터: 근거 없는 승자·추천·압도 표현만 잡는다. 사실 비교 문장은 통과시킨다.

도메인 에이전트(agents/domain/quality/linter.py)와 같은 목적이지만, "다른 에이전트 폴더를
import하지 않는다"는 팀 규칙에 따라 시장 에이전트 것을 따로 둔다.
"""

from __future__ import annotations

import re

_VERDICT = re.compile(r"(더\s*(우수|뛰어|낫|유리|효과적)|우월|우위|최선|최적의\s*선택|승자|이긴다|앞선다|앞서간다)")
_RECOMMEND = re.compile(r"(권장|추천|채택해야|도입해야|선택해야|사용해야\s*한다|바람직하다)")
_OVERSTATE = re.compile(r"(압도적|월등|현저히\s*우수|비교할\s*수\s*없|단연|획기적|혁신적)")
_PATTERNS = (("verdict", _VERDICT), ("recommendation", _RECOMMEND), ("overstatement", _OVERSTATE))


def lint_statements(statements: list[tuple[str, str]]) -> list[dict]:
    """[(claim_id, statement), ...] 를 받아 위반 목록을 반환한다."""
    hits = []
    for claim_id, text in statements:
        for kind, pattern in _PATTERNS:
            m = pattern.search(text)
            if m:
                hits.append({"claim_id": claim_id, "kind": kind, "matched": m.group(0)})
    return hits

"""표현 린터: 비교 사실은 허용하고, 근거 없는 승자·추천·압도 표현만 막는다.

과제는 우열 판정이 아니라 "관점에 따라 평가가 어떻게 갈리는가"를 보는 것이다.
그렇다고 비교 자체를 막으면 "SW는 93.3% 감소, HW는 1.80배 향상" 같은 사실 진술까지
사라져 보고서가 공허해진다. 그래서 사실 비교는 통과시키고 평가적 결론만 잡는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 승자 판정: 어느 쪽이 낫다는 결론
_VERDICT = re.compile(
    r"(더\s*(우수|뛰어|낫|유리|효과적)|우월|우위|최선|최적의\s*선택|승자|이긴다|앞선다|앞서간다)"
)
# 추천: 행동 지시
_RECOMMEND = re.compile(r"(권장|추천|채택해야|도입해야|선택해야|사용해야\s*한다|바람직하다)")
# 압도: 과장 수식
_OVERSTATE = re.compile(r"(압도적|월등|현저히\s*우수|비교할\s*수\s*없|단연|획기적|혁신적)")

_PATTERNS = (("verdict", _VERDICT), ("recommendation", _RECOMMEND), ("overstatement", _OVERSTATE))


@dataclass
class LintFinding:
    target_id: str
    kind: str  # verdict / recommendation / overstatement
    matched: str
    blocking: bool  # True 면 해당 주장을 통과시키지 않는다
    detail: str


@dataclass
class LintReport:
    findings: list[LintFinding] = field(default_factory=list)

    @property
    def blocking_ids(self) -> set[str]:
        return {f.target_id for f in self.findings if f.blocking}

    def to_dict(self) -> dict:
        return {
            "blocking_count": sum(1 for f in self.findings if f.blocking),
            "warning_count": sum(1 for f in self.findings if not f.blocking),
            "findings": [
                {
                    "target_id": f.target_id,
                    "kind": f.kind,
                    "matched": f.matched,
                    "blocking": f.blocking,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


def lint_claims(claims: list[dict]) -> LintReport:
    """근거가 없는 평가 표현은 차단하고, 근거가 있어도 추천·압도 표현은 경고로 남긴다."""
    report = LintReport()
    for claim in claims:
        claim_id = claim.get("claim_id", "?")
        statement = claim.get("statement") or ""
        has_evidence = bool(claim.get("evidence_ids")) and claim.get("basis") == "direct_evidence"

        for kind, pattern in _PATTERNS:
            match = pattern.search(statement)
            if not match:
                continue
            # 추천과 압도는 근거가 있어도 이 보고서의 목적(우열 판정 아님)에 어긋나 경고로 남긴다.
            blocking = not has_evidence or kind in ("recommendation", "overstatement")
            report.findings.append(
                LintFinding(
                    target_id=claim_id,
                    kind=kind,
                    matched=match.group(0),
                    blocking=blocking,
                    detail=(
                        "근거 없는 평가 표현" if not has_evidence else "우열 판정에 해당하는 표현"
                    ),
                )
            )
    return report

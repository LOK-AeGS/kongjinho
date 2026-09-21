"""결정적 검증: 코드로 확인 가능한 것만 검사한다.

LLM judge 에게 "이 인용이 원문에 있나"를 묻지 않는다. 문자열 대조로 확인 가능한 일을
확률적 판정자에게 맡기면 통과 여부가 실행마다 달라진다. judge 는 coverage·neutrality 처럼
대조로 답할 수 없는 것만 본다.

검사 항목: citation, locator, quote, numeric unit, date, reference integrity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

from rag.evidence import normalize_text
from rag.fetch import has_numeric_with_unit

# 단위가 붙은 수치(측정값)만 대조한다.
# 모든 숫자를 뽑았더니 "DeepSeek-V2" 의 2 를 측정값으로 보고 멀쩡한 주장을 강등시켰다.
# 제품명·버전과 측정값을 구분할 방법은 단위 유무뿐이다.
_MEASUREMENT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|배|[xX]\b|GB|TB|MB|KB|ms|us|ns|초|W\b|kW|MW|TFLOPS|GB/s|TB/s|tokens?/s)",
)
_ANY_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_ISO_DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


@dataclass
class GuardViolation:
    check: str
    target_id: str
    detail: str


@dataclass
class GuardReport:
    violations: list[GuardViolation] = field(default_factory=list)
    checked_claims: int = 0
    checked_evidence: int = 0

    @property
    def passed(self) -> bool:
        return not self.violations

    def by_check(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.check] = counts.get(v.check, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checked_claims": self.checked_claims,
            "checked_evidence": self.checked_evidence,
            "violations": [
                {"check": v.check, "target_id": v.target_id, "detail": v.detail}
                for v in self.violations
            ],
            "counts_by_check": self.by_check(),
        }


def _measurements_in(text: str) -> set[str]:
    """단위가 붙은 수치만. 주장 쪽에서 무엇을 검사할지 고르는 데 쓴다."""
    return set(_MEASUREMENT.findall(text or ""))


def _numbers_in(text: str) -> set[str]:
    """근거 쪽 대조용. 인용문에서는 단위 표기가 생략될 수 있어 모든 숫자를 본다."""
    return set(_ANY_NUMBER.findall(text or ""))


def run_guard(
    *,
    claims: list[dict],
    evidence_store: dict[str, dict],
    source_texts: dict[str, str],
    as_of_date: str,
) -> GuardReport:
    """claims 와 evidence_store 를 대조 검사한다.

    source_texts 는 {url: 정규화된 원문} 이며 quote 대조에 쓴다.
    원문 스냅샷이 없는 근거는 quote 검증을 건너뛰지 않고 위반으로 기록한다.
    """
    report = GuardReport(checked_claims=len(claims), checked_evidence=len(evidence_store))
    today = date.fromisoformat(as_of_date)

    for evidence_id, ev in evidence_store.items():
        # locator: 근거가 문서 어디에서 왔는지 특정할 수 없으면 검증도 반박도 불가능하다.
        if not (ev.get("locator") or "").strip():
            report.violations.append(
                GuardViolation("locator", evidence_id, "locator 가 비어 있음")
            )

        # reference integrity: URL 형식
        parsed = urlparse(ev.get("url") or "")
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            report.violations.append(
                GuardViolation("reference_integrity", evidence_id, f"URL 형식 오류: {ev.get('url')}")
            )

        # quote: 원문에 실제로 있는 문장인지 문자열 대조
        quote = normalize_text(ev.get("quote") or "")
        if not quote:
            report.violations.append(GuardViolation("quote", evidence_id, "인용문이 비어 있음"))
        else:
            source = source_texts.get(ev.get("url") or "")
            if source is None:
                report.violations.append(
                    GuardViolation("quote", evidence_id, "원문 스냅샷이 없어 인용을 대조할 수 없음")
                )
            elif quote not in source:
                report.violations.append(
                    GuardViolation("quote", evidence_id, f"원문에 없는 인용문: {quote[:60]}")
                )

        # date: 형식과 미래 날짜
        published = ev.get("published_date")
        if published:
            if not _ISO_DATE.match(published):
                report.violations.append(
                    GuardViolation("date", evidence_id, f"날짜 형식 오류: {published}")
                )
            else:
                parts = [int(p) for p in published.split("-")]
                published_date = date(parts[0], parts[1] if len(parts) > 1 else 1,
                                      parts[2] if len(parts) > 2 else 1)
                if published_date > today:
                    report.violations.append(
                        GuardViolation("date", evidence_id, f"기준일 이후 발행일: {published}")
                    )

    for claim in claims:
        claim_id = claim.get("claim_id", "?")
        cited = claim.get("evidence_ids") or []

        # citation: 사실 주장은 반드시 근거를 참조한다
        if claim.get("basis") == "direct_evidence" and not cited:
            report.violations.append(
                GuardViolation("citation", claim_id, "근거 없이 direct_evidence 로 표기")
            )

        # reference integrity: 참조한 ID 가 저장소에 실재하는가
        missing = [eid for eid in cited if eid not in evidence_store]
        if missing:
            report.violations.append(
                GuardViolation("reference_integrity", claim_id, f"없는 근거 참조: {missing}")
            )

        # numeric unit: 주장이 제시한 측정값이 인용한 근거에 실제로 있는가.
        # 단위가 붙은 수치만 본다. 제품명·버전의 숫자(DeepSeek-V2 의 2)는 측정값이 아니다.
        statement = claim.get("statement") or ""
        claim_measurements = _measurements_in(statement)
        if claim_measurements:
            quotes = " ".join(
                normalize_text(evidence_store[eid].get("quote", ""))
                for eid in cited
                if eid in evidence_store
            )
            unsupported = claim_measurements - _numbers_in(quotes)
            if unsupported:
                report.violations.append(
                    GuardViolation(
                        "numeric_unit", claim_id, f"인용문에 없는 수치: {sorted(unsupported)}"
                    )
                )
        elif has_numeric_with_unit(statement):
            # 정규식이 놓친 단위 표기. 대조는 못 하지만 기록은 남긴다.
            report.violations.append(
                GuardViolation("numeric_unit", claim_id, "단위를 인식하지 못해 대조하지 못함")
            )

    return report

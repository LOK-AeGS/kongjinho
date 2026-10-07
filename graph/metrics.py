"""에이전트 경계를 넘어 같은 방식으로 측정값을 찾고 비교한다."""

from __future__ import annotations

import re


MEASUREMENT = re.compile(
    r"(?<![\w.-])\d+(?:\.\d+)?\s*(?:%|×|x(?=\s|$)|배|GB/s|GB|TB|MB|ms|μs|초|달러|원)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def extract_measurements(text: str) -> set[str]:
    return {
        re.sub(r"\s+", "", value).lower()
        for value in MEASUREMENT.findall(text or "")
    }


def measurement_values(text: str) -> set[str]:
    """단위 표기가 달라도 원문 숫자를 비교할 수 있도록 숫자 부분만 정규화한다."""
    return {
        match.group(0)
        for measurement in MEASUREMENT.findall(text or "")
        if (match := _NUMBER.search(measurement)) is not None
    }

"""에이전트 경계를 넘어 같은 방식으로 측정값을 찾고 비교한다."""

from __future__ import annotations

import re


# 주장 쪽에서 "측정값"을 찾는 패턴. 영어 원문 표기(5.76 times, 35.7\\%, 2-fold)도 측정값으로 본다.
MEASUREMENT = re.compile(
    r"(?<![\w.-])\d+(?:\.\d+)?\s*(?:\\?%|×|x(?=\s|$)|배|times\b|fold\b|percent\b|GB/s|GB|TB|MB|ms|μs|초|달러|원)",
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


def number_pattern(value: str) -> re.Pattern:
    """'5.76'이 원문에 '5.76', '5 . 76', '5.76×', '5.76 times', '35.7\\%' 어떤 표기로 있어도 찾는다.

    공백은 소수점 주변에서만 허용한다. 숫자 사이까지 허용하면 그림 축 눈금 '1.8 1.6'을 1.81로 읽어,
    1.81이 없는 축 조각을 인용한 주장이 근거 있음으로 통과했다(live).
    """
    integer, _, fraction = value.partition(".")
    body = re.escape(integer) + (rf"\s*\.\s*{re.escape(fraction)}" if fraction else "")
    return re.compile(rf"(?<![\d.]){body}(?![\d])")


def unsupported_values(values: set[str], text: str) -> set[str]:
    """근거 원문에 숫자로 나타나지 않는 값. 단위·표기는 보지 않는다.

    단위가 붙은 형태로만 비교하면 원문의 '5.76 times', '35.7\\%'를 놓쳐 근거가 있는 주장을
    근거 없음으로 판정했다(live 1~5차 기술 조사 수치 21건 중 8건이 이 오탐).
    """
    clean = (text or "").replace("\\", "").replace(",", "")
    return {value for value in values if not number_pattern(value).search(clean)}

"""평가 종합 규칙 상수. 설계서 v0.8 §5.2, §6.3, §8.6.1, §8.6.2 기준.

relations.py(상충 규칙 SX)와 review.py(중립성 검사 C)가 같은 표를 쓴다.
"""

from __future__ import annotations

import re

RULES_VERSION = "synthesis-rules-v1"

PERSPECTIVES = ("technical", "market", "stakeholder", "domain")
TECHNOLOGIES = ("sw", "hw")

# §6.3 증거 수준 순서. 비교에 쓴다.
EVIDENCE_LEVEL_RANK = {"unknown": 0, "forecast": 1, "announcement": 2, "pilot": 3, "production": 4}

# SX4 수치 차이 임계, SX5 시점 차이 임계
NUMERIC_GAP_RATIO = 0.10
TEMPORAL_GAP_DAYS = 365

# 상충 설명에서 쓸 수 있는 규칙 기준값 (SX1 운영 환경 = TRL 7단계, SX4 10%, SX5 12개월·1년·365일)
RULE_THRESHOLD_NUMBERS = {"SX1": ("7",), "SX4": ("10",), "SX5": ("1", "12", "365")}

# 근거 수 불균형 임계 (§6.4: 2배 이상이면 limitations 에 기록)
IMBALANCE_RATIO = 2.0

# C4 / SX3: 조건 필수 수치. 숫자가 나오면 토큰 묶음 중 하나 이상이 같은 문장에 있어야 한다.
# 각 항목: (숫자, 필요한 토큰 묶음 목록(모두 충족해야 함), 설명)
CONDITION_RULES: list[tuple[str, list[tuple[str, ...]], str]] = [
    ("93.3", [("67B",)], "93.3%는 DeepSeek 67B 대비 전체 모델 비교"),
    ("5.76", [("H800",)], "5.76×는 단일 노드 8×H800 조건"),
    ("1.80", [("NVMe-oF",)], "1.80×는 NVMe-oF 기반 disaggregated storage 대비 처리량"),
    ("1.81", [("Recompute", "재계산"), ("turn 5", "TTFT")], "1.81×는 Recompute 대비 TTFT, turn 5 시점"),
    ("35.7", [("CPU-offload", "CPU 오프로드"), ("최대", "up to", "maximum")], "35.7%는 CPU-offload 대비 최대값"),
]

# 기술별 금지 수치 (§5.1, §5.2): 42.5는 MLA 문장, 3.02는 ITME 문장에서 쓰지 않는다.
FORBIDDEN_NUMBERS = {
    "sw": [("42.5", "42.5% 훈련비 절감은 MLA 효과가 아님 (DeepSeekMoE·전체 훈련 효율에 귀속)")],
    "hw": [("3.02", "3.02×는 Ideal GPU memory 상한값이며 ITME 수치가 아님")],
}

# C5 우열 어휘
RANKING_WORDS = ("우수", "더 낫", "승자", "추천", "권장", "압도", "우위", "열위", "월등")

# C6 도메인 밖 키워드
OUT_OF_DOMAIN_WORDS = ("온디바이스", "엣지", "모바일", "on-device", "edge", "mobile")

_NUMBER = re.compile(r"(?<![A-Za-z0-9.\-:_/])(\d+(?:\.\d+)?)(?![A-Za-z0-9:_])")


def numbers_in(text: str) -> list[str]:
    """문장 속 수치. 제품명에 붙은 숫자(V2, H800, 67B, T3.5)와 ID 속 숫자(market:claim:003)는 제외한다."""
    return _NUMBER.findall(text or "")


def number_present(number: str, haystack: str) -> bool:
    """근거 문장에 같은 수치가 있는지. 1.80 과 1.8 을 같은 값으로 본다."""
    target = float(number)
    return any(float(n) == target for n in _NUMBER.findall(haystack or ""))


def condition_violations(text: str, technology: str | None) -> list[str]:
    """C4 / SX3: 조건 필수 수치에 조건 토큰이 빠졌거나, 기술별 금지 수치가 쓰인 경우."""
    found = numbers_in(text)
    problems = []
    for number, groups, why in CONDITION_RULES:
        if any(float(n) == float(number) for n in found):
            if not all(any(token.lower() in text.lower() for token in group) for group in groups):
                problems.append(f"C4 조건 누락: {why}")
    for tech, rules in FORBIDDEN_NUMBERS.items():
        if technology in (tech, "both"):
            for number, why in rules:
                if any(float(n) == float(number) for n in found):
                    problems.append(f"C4 금지 수치: {why}")
    return problems


def trl_bounds(value: str | None) -> tuple[int, int] | None:
    """TRL 값 "4", "4-5", "4–5" 를 (하한, 상한)으로. 해석할 수 없으면 None."""
    if not value:
        return None
    digits = [int(d) for d in re.findall(r"\d", value)]
    if not digits or any(d < 1 or d > 9 for d in digits):
        return None
    return min(digits), max(digits)

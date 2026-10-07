"""보고서 핵심 수치의 조건 규칙과 결정적 주석."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricRule:
    code: str
    metric: re.Pattern
    required: tuple[re.Pattern, ...]
    forbidden: tuple[re.Pattern, ...]
    message: str
    canonical_condition: str | None


def _patterns(*values: str) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(value, re.IGNORECASE) for value in values)


# validator와 annotate_metrics가 이 표를 함께 사용한다. 새 수치 조건은 한 곳에만 추가한다.
METRIC_RULES = (
    MetricRule(
        "metric_93_3_context",
        re.compile(r"93\.3"),
        _patterns(r"deepseek67b"),
        _patterns(r"mla만으로", r"mla단독"),
        "93.3%에는 DeepSeek 67B 대비 전체 비교 조건이 필요함",
        "DeepSeek 67B 대비 DeepSeek-V2 전체 모델 비교",
    ),
    MetricRule(
        "metric_5_76_context",
        re.compile(r"5\.76"),
        _patterns(r"8[×x]h800|8개h800"),
        (),
        "5.76×에는 8×H800 조건이 필요함",
        "8×H800 기준",
    ),
    MetricRule(
        "metric_35_7_context",
        re.compile(r"35\.7"),
        _patterns(r"최대"),
        (),
        "35.7%는 최대값으로 표시해야 함",
        "CPU-offload 대비 최대값",
    ),
    MetricRule(
        "metric_1_81_context",
        re.compile(r"1\.81"),
        _patterns(r"ttft", r"turn5", r"recomputation"),
        (),
        "1.81×에는 turn 5, TTFT, recomputation baseline 조건이 필요함",
        "turn 5 TTFT, recomputation 대비",
    ),
    MetricRule(
        "metric_42_5_attribution",
        re.compile(r"42\.5"),
        _patterns(r"훈련|training"),
        _patterns(r"mla"),
        "42.5%를 MLA 효과로 귀속할 수 없음",
        None,
    ),
)


def metric_violations(text: str) -> list[MetricRule]:
    folded = text.casefold().replace(" ", "")
    return [
        rule for rule in METRIC_RULES
        if rule.metric.search(text)
        and (
            not all(pattern.search(folded) for pattern in rule.required)
            or any(pattern.search(folded) for pattern in rule.forbidden)
        )
    ]


_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+")


def _annotate_sentence(sentence: str) -> str:
    annotations = [
        rule.canonical_condition
        for rule in metric_violations(sentence)
        if rule.canonical_condition
    ]
    if not annotations:
        return sentence
    suffix = " " + " ".join(f"({value})" for value in annotations)
    stripped = sentence.rstrip()
    # validator는 문장 단위로 조건을 찾으므로, 조건을 마침표 뒤가 아니라 그 문장 안에 넣는다.
    if stripped and stripped[-1] in ".!?。":
        return stripped[:-1] + suffix + stripped[-1]
    return stripped + suffix


def _annotate_text(text: str) -> str:
    return " ".join(_annotate_sentence(part) for part in _SENTENCE_END.split(text)) if text.strip() else text


def annotate_metrics(markdown: str) -> str:
    """문장 또는 표 셀에 빠진 canonical 측정 조건을 덧붙인다."""
    lines = []
    for line in (markdown or "").splitlines():
        if line.strip().startswith("|") and line.count("|") >= 2:
            cells = line.split("|")
            cells = [cells[0], *[_annotate_text(cell) for cell in cells[1:-1]], cells[-1]]
            lines.append("|".join(cells))
        else:
            lines.append(_annotate_text(line))
    return "\n".join(lines)

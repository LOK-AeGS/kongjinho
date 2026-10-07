"""보고서 핵심 수치의 조건 규칙과 결정적 주석."""

from __future__ import annotations

import re
from dataclasses import dataclass

from graph.rules import METRIC_RULE_SPECS


@dataclass(frozen=True)
class MetricRule:
    code: str
    metric: re.Pattern
    required: tuple[re.Pattern, ...]
    forbidden: tuple[re.Pattern, ...]
    message: str
    canonical_condition: str | None
    evidence_key: re.Pattern | None


def _patterns(*values: str) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(value, re.IGNORECASE) for value in values)


# 규칙 내용은 graph/rules.py의 METRIC_RULE_SPECS 한 곳에 있다. 여기서는 정규식으로 컴파일만 한다.
METRIC_RULES = tuple(
    MetricRule(
        code=spec["code"],
        metric=re.compile(spec["metric"]),
        required=_patterns(*spec["required"]),
        forbidden=_patterns(*spec["forbidden"]),
        message=spec["message"],
        canonical_condition=spec["condition"],
        evidence_key=re.compile(spec["evidence_key"], re.IGNORECASE) if spec["evidence_key"] else None,
    )
    for spec in METRIC_RULE_SPECS
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
# 조건이 빠졌다는 사실을 설명하는 문장에 그 조건을 덧붙이면 뜻이 모순된다("기준이 누락되어 …(기준 문구)").
_ABOUT_MISSING = re.compile(r"누락|빠져|빠진|명기되지|명시되지|불명확")


def condition_supported(rule: MetricRule, evidence_text: str) -> bool:
    """표준 조건이 인용 근거 원문에 실제로 있는지. 근거에 없는 조건은 덧붙이지 않는다."""
    if rule.evidence_key is None:
        return False
    return bool(rule.evidence_key.search(evidence_text.casefold().replace(" ", "")))


def unsupported_metric_rules(text: str, evidence_text: str) -> list[MetricRule]:
    """조건이 빠졌는데 근거 원문으로도 보정할 수 없는 규칙. 이런 문장은 보고서에 싣지 않는다."""
    return [
        rule
        for rule in metric_violations(text)
        if rule.canonical_condition is None
        or _ABOUT_MISSING.search(text)
        or not condition_supported(rule, evidence_text)
    ]


def _annotate_sentence(sentence: str, evidence_text: str) -> str:
    if _ABOUT_MISSING.search(sentence):
        return sentence
    annotations = [
        rule.canonical_condition
        for rule in metric_violations(sentence)
        if rule.canonical_condition and condition_supported(rule, evidence_text)
    ]
    if not annotations:
        return sentence
    suffix = " " + " ".join(f"({value})" for value in annotations)
    stripped = sentence.rstrip()
    # validator는 문장 단위로 조건을 찾으므로, 조건을 마침표 뒤가 아니라 그 문장 안에 넣는다.
    if stripped and stripped[-1] in ".!?。":
        return stripped[:-1] + suffix + stripped[-1]
    return stripped + suffix


def _annotate_text(text: str, evidence_text: str) -> str:
    if not text.strip():
        return text
    return " ".join(_annotate_sentence(part, evidence_text) for part in _SENTENCE_END.split(text))


def annotate_metrics(markdown: str, evidence_lookup=None) -> str:
    """문장 또는 표 셀에 빠진 표준 측정 조건을 덧붙인다. 단, 그 줄이 인용한 근거 원문에 조건이 있을 때만.

    evidence_lookup(line) -> 그 줄이 인용한 근거 원문 텍스트. 없으면 아무것도 덧붙이지 않는다.
    """
    lines = []
    for line in (markdown or "").splitlines():
        evidence_text = evidence_lookup(line) if evidence_lookup else ""
        if line.strip().startswith("|") and line.count("|") >= 2:
            cells = line.split("|")
            cells = [cells[0], *[_annotate_text(cell, evidence_text) for cell in cells[1:-1]], cells[-1]]
            lines.append("|".join(cells))
        else:
            lines.append(_annotate_text(line, evidence_text))
    return "\n".join(lines)

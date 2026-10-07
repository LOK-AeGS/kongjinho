"""보고서 근거 연결(Groundedness) 판정의 공유 기준.

보고서 검증(agents/report)과 품질 평가(agents/quality)가 같은 "사실 문장" 분류와 인용률 기준을 쓴다.
두 계층의 기준이 달라 live 4차에서 보고서 검증은 위반 0건, 품질 평가는 인용률 0.51로 어긋났다.
"""

from __future__ import annotations

import re

CITATION = re.compile(r"〔근거:\s*([^〕]+)〕")
# 사실 문장 중 인용이 붙은 비율의 하한
MIN_CITATION_RATIO = 0.8


def citation_ids(text: str) -> list[str]:
    return [item.strip() for group in CITATION.findall(text or "") for item in group.split(",") if item.strip()]


META_PHRASES = (
    "확인되지 않았다", "미확인", "분량 제한", "판단 보류", "not_assessed",
    "silent", "unknown", "근거 0건", "찾지 못", "None |", "평가 범위:",
    "조사 기준일", "공개 정보로 확인 가능한 범위", "판단 보류 상태:",
    "근거 부족", "공개 근거에서 확인하지 못",
)
GROUNDEDNESS_EXCLUDED_SECTIONS = (
    "# 2. 기술 선정",
    "## 5.5 남은 확인 과제",
    "# 6. 한계점",
)

def _body(markdown: str) -> str:
    head = markdown.partition("# REFERENCE")[0]
    start = head.find("# SUMMARY")
    return head[start:] if start >= 0 else head


def factual_body_lines(markdown: str) -> list[str]:
    """groundedness 분모가 되는 사실 불릿·표 행을 반환한다."""
    factual = []
    lines = _body(markdown).splitlines()
    excluded_level: int | None = None
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if line in GROUNDEDNESS_EXCLUDED_SECTIONS:
                excluded_level = level
            elif excluded_level is not None and level <= excluded_level:
                excluded_level = None
            continue
        if excluded_level is not None or not line or set(line.replace(" ", "")) <= {"|", "-", ":"}:
            continue
        if not line.startswith(("- ", "* ", "|")):
            continue
        if line.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip().replace(" ", "")
            if separator and set(separator) <= {"|", "-", ":"}:
                continue
        if any(phrase in line for phrase in META_PHRASES):
            continue
        factual.append(line)
    return factual

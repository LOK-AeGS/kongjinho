"""보고서 섹션별 분량 예산과 우선순위 선별.

보고서는 최대 10장(표지 포함)으로 제한된다. 전체를 같은 비율로 줄이지 않고 섹션마다
상한(max_chars)·항목 수(max_items)를 두며, 관점 4개(4.1~4.4)는 비슷한 분량을 받는다.

강제는 세 단계로 이뤄진다.
1. 입력 선별: writer에 넘기기 전에 claim·관계·행을 우선순위대로 max_items개만 남긴다.
2. 프롬프트: 섹션 규칙에 분량 상한을 적는다.
3. 검증·페이지 가드: 상한 초과는 경고로 남기고, PDF가 PAGE_LIMIT를 넘으면
   초과 섹션을 결정적 렌더로 교체하고 B·C 등급 예산을 줄여 다시 렌더링한다.

수치는 2026-09-22 live 실행 보고서(16쪽, 본문 27,846자)를 기준으로 잡은 시작값이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Literal, TypeVar

from agents.report.state import SectionId

Tier = Literal["A", "B", "C", "fixed"]

# 표지 1장을 포함한 최대 장수.
PAGE_LIMIT = 10
# PDF가 없을 때(테스트, --no-pdf) 쓰는 추정치. A4 본문 9.2pt 기준 실측 약 1,700자/장.
CHARS_PER_PAGE = 1700
# 상한을 이 비율까지 넘는 것은 허용한다(문장 단위로 끊기지 않는 경우).
OVER_BUDGET_TOLERANCE = 1.15
# 페이지 가드에서 B·C 등급 예산을 한 번에 줄이는 비율과 최대 반복 횟수.
SHRINK_FACTOR = 0.8
MAX_PAGE_GUARD_ROUNDS = 3


@dataclass(frozen=True)
class SectionBudget:
    max_chars: int
    max_items: int
    tier: Tier
    # 이보다 짧으면 관점 간 분량 불균형으로 경고한다(근거가 없으면 gap을 명시해야 함).
    min_chars: int = 0


SECTION_BUDGETS: dict[SectionId, SectionBudget] = {
    # A: 보호 — 페이지 가드에서도 줄이지 않는다.
    "summary": SectionBudget(1200, 5, "A"),
    "trl": SectionBudget(1100, 4, "A", min_chars=400),
    "market": SectionBudget(1100, 4, "A", min_chars=400),
    "stakeholder": SectionBudget(1100, 4, "A", min_chars=400),
    "domain": SectionBudget(1200, 12, "A", min_chars=400),
    # B: 압축 — 시사점과 한계점
    "comparison_matrix": SectionBudget(1500, 12, "B"),
    "conditions": SectionBudget(1200, 8, "B"),
    "conflicts": SectionBudget(1000, 5, "B"),
    "shared_and_complement": SectionBudget(500, 3, "B"),
    "open_questions": SectionBudget(500, 5, "B"),
    "limitations": SectionBudget(900, 7, "B"),
    # C: 축약 — 배경·개요
    "background": SectionBudget(600, 2, "C"),
    "technology_selection": SectionBudget(400, 2, "fixed"),
    "technology_overview": SectionBudget(1000, 4, "C"),
    # REFERENCE는 본문 인용으로 결정된다(출처 1건 ≈ 200자). 상한은 경고용.
    "reference": SectionBudget(4500, 25, "fixed"),
}

Budgets = dict[SectionId, SectionBudget]


def default_budgets() -> Budgets:
    return dict(SECTION_BUDGETS)


def shrink(budgets: Budgets, factor: float = SHRINK_FACTOR) -> Budgets:
    """B·C 등급만 비율대로 줄인다. 항목 수는 최소 1개를 남긴다."""
    return {
        section_id: (
            replace(
                budget,
                max_chars=int(budget.max_chars * factor),
                max_items=max(1, int(budget.max_items * factor)),
            )
            if budget.tier in {"B", "C"}
            else budget
        )
        for section_id, budget in budgets.items()
    }


_CITATION_MARK = re.compile(r"〔근거:\s*[^〕]+〕")


def body_length(markdown: str) -> int:
    """제목 줄을 뺀 섹션 본문의 제출본 기준 길이.

    제출본은 `〔근거: 긴 evidence ID…〕`를 `[n]`으로 바꿔 찍으므로 인용은 4자로 센다.
    """
    lines = markdown.strip().splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return len(_CITATION_MARK.sub("[00]", "\n".join(lines).strip()))


def estimate_pages(markdown: str) -> int:
    """PDF 없이 장수를 추정한다. 표지 1장 + 본문."""
    return 1 + -(-len(markdown) // CHARS_PER_PAGE)


T = TypeVar("T", bound=dict)


def _technologies(item: dict) -> list[str]:
    ids = item.get("technology_ids")
    if ids:
        return list(ids)
    technology = item.get("technology")
    if technology == "both":
        return ["sw", "hw"]
    return [technology] if technology else []


def balanced_pick(
    items: Iterable[T],
    limit: int,
    *,
    is_counter=lambda item: False,
) -> list[T]:
    """SW·HW가 번갈아 들어가도록 고르고, 기술마다 반대 근거 1건을 먼저 확보한다.

    한 기술의 항목만 앞에 몰려 있어도 분량이 한쪽으로 쏠리지 않게 하고(중립성),
    잘라낸 뒤에도 반대 근거가 남게 한다(확증편향 방지). 원래 순서는 유지한다.
    """
    items = list(items)
    if len(items) <= limit:
        return items
    chosen: set[int] = set()
    for side in ("sw", "hw"):
        for index, item in enumerate(items):
            if len(chosen) >= limit:
                break
            if side in _technologies(item) and is_counter(item):
                chosen.add(index)
                break
    queues = {
        side: [index for index, item in enumerate(items) if side in _technologies(item)]
        for side in ("sw", "hw")
    }
    others = [index for index, item in enumerate(items) if not set(_technologies(item)) & {"sw", "hw"}]
    while len(chosen) < limit and (queues["sw"] or queues["hw"] or others):
        for queue in (queues["sw"], queues["hw"], others):
            while queue and queue[0] in chosen:
                queue.pop(0)
            if queue and len(chosen) < limit:
                chosen.add(queue.pop(0))
    return [item for index, item in enumerate(items) if index in chosen]


def round_robin_pick(items: Iterable[T], limit: int, key) -> list[T]:
    """그룹(관점 등)을 번갈아 돌며 limit개를 고른다. 출력은 원래 순서를 유지한다."""
    items = list(items)
    if len(items) <= limit:
        return items
    queues: dict[object, list[int]] = {}
    for index, item in enumerate(items):
        queues.setdefault(key(item), []).append(index)
    chosen: set[int] = set()
    while len(chosen) < limit:
        for queue in queues.values():
            if queue and len(chosen) < limit:
                chosen.add(queue.pop(0))
    return [item for index, item in enumerate(items) if index in chosen]


def clip(text: str, limit: int = 260) -> str:
    """한 항목이 섹션 예산을 혼자 차지하지 않도록 자른다. 잘랐으면 말줄임표를 붙인다."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def omitted_note(total: int, kept: int, unit: str = "건") -> str:
    """잘라낸 항목 수를 본문에 밝히는 문장. 생략이 없으면 빈 문자열."""
    if total <= kept:
        return ""
    return f"- 분량 제한으로 전체 {total}{unit} 중 {kept}{unit}만 싣고 {total - kept}{unit}은 생략했다(전체 목록은 final_state.json)."

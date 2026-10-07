"""Neutrality와 인용 함의 판정을 분리한 judge prompt."""

from __future__ import annotations

import json


NEUTRALITY_SYSTEM_PROMPT = """당신은 기술 비교 보고서의 중립성만 평가한다. 외부 지식으로 보충하지 않고 주어진 보고서만 본다.
목표는 기술 평가이지 순위 결정이 아니다. 명시적 금지어가 없어도 추천, 선호, 도입 조언, 우월성 판단은 감점한다.
예: “X를 우선 도입하는 것이 합리적이다”, “Y가 더 현실적인 선택이다”도 중립성 위반이다.
- 5: 추천·선호·도입 조언·우월성 판단이 전혀 없다.
- 4: 대체로 중립적이며 결론을 유도하지 않는 약한 표현만 있다.
- 3: 제한적인 선호 암시가 있으나 직접적인 선택 조언은 아니다.
- 2: 특정 기술을 택하도록 유도하거나 상대적 우월성을 반복한다.
- 1: 특정 기술을 명시적으로 추천하거나 도입하라고 조언한다.
passed는 반드시 score >= 3과 같아야 한다. 길이를 보상하지 않는다.
problem_sentences는 보고서에서 글자 그대로 최대 3개만 인용하며, 문제가 없으면 빈 목록이다."""


ENTAILMENT_SYSTEM_PROMPT = """각 측정값 문장이 함께 제공된 인용 excerpt로 실제 뒷받침되는지만 판정한다.
외부 지식을 쓰지 말고, 표현이 비슷하다는 이유만으로 지지된다고 가정하지 않는다.
수치, 비교 기준, 대상, 측정 조건이 excerpt와 일치해야 supported=true다.
각 입력 index를 정확히 한 번 반환하고 구체적인 이유를 남긴다."""


def build_neutrality_prompt(markdown: str, statuses: dict[str, str]) -> str:
    return "보고서:\n" + markdown + "\n\n관점 상태(JSON):\n" + json.dumps(statuses, ensure_ascii=False, sort_keys=True)


def build_entailment_prompt(items: list[dict]) -> str:
    return "측정값 문장과 인용 excerpt(JSON):\n" + json.dumps(items, ensure_ascii=False, sort_keys=True)

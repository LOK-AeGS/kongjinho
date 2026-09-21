"""LLM 질적 평가: coverage 와 neutrality 만 본다.

인용 대조·수치 확인·날짜 형식처럼 코드로 답할 수 있는 것은 guard 가 맡는다.
judge 에게 그것까지 맡기면 같은 입력에 다른 판정이 나와 재현성이 깨진다.
여기서는 대조로 답할 수 없는 두 가지만 묻는다.

- coverage: 도메인 요구사항 축이 실제로 다뤄졌는가, 어느 축이 비었는가
- neutrality: 한쪽을 밀어주는 서술인가, 제약과 기대 효과가 함께 기술됐는가
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

JUDGE_PROMPT_VERSION = "domain-judge/v1"

JUDGE_SYSTEM = """당신은 기술 평가 보고서의 품질 검토자입니다.
아래 두 가지만 평가하고, 사실 확인이나 인용 검증은 하지 않습니다(그건 별도 검사가 담당합니다).

coverage: 제시된 도메인 요구사항 축 중 실제로 다뤄진 축과 비어 있는 축을 가려냅니다.
neutrality: 서술이 한쪽 기술을 밀어주고 있는지 봅니다.
- 기대 효과만 쓰고 제약을 빼놓으면 중립적이지 않습니다.
- 수치를 근거로 한 비교 진술 자체는 중립적입니다. 우열 결론·추천 표현이 문제입니다.
- 두 기술의 서술 분량이나 어조가 크게 기울면 지적하세요."""

JUDGE_USER = """[도메인 요구사항 축]
{requirements}

[생성된 주장]
{claims}

[생성된 판정]
{fits}

coverage 와 neutrality 를 평가하세요."""


class JudgeVerdict(BaseModel):
    covered_axes: list[str] = Field(description="근거와 함께 실제로 다뤄진 요구사항 축")
    missing_axes: list[str] = Field(description="다뤄지지 않았거나 근거가 없는 축")
    neutrality: Literal["neutral", "tilted", "unknown"] = Field(description="중립성 판정")
    neutrality_notes: list[str] = Field(description="한쪽으로 기운 부분. 없으면 빈 목록")


@dataclass
class JudgeResult:
    covered_axes: list[str]
    missing_axes: list[str]
    neutrality: str
    neutrality_notes: list[str]
    model: str
    prompt_version: str = JUDGE_PROMPT_VERSION
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.neutrality == "neutral" and not self.missing_axes

    def to_dict(self) -> dict:
        return {
            "covered_axes": self.covered_axes,
            "missing_axes": self.missing_axes,
            "neutrality": self.neutrality,
            "neutrality_notes": self.neutrality_notes,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "passed": self.passed,
            "error": self.error,
        }


def run_judge(llm, *, requirements: str, claims: list[dict], fits: list[dict], model_name: str) -> JudgeResult:
    """judge 실패가 그래프를 중단시키지 않게 한다. 실패는 unknown 으로 기록하고 진행한다."""
    claim_lines = "\n".join(
        f"- [{c.get('claim_id')}] ({c.get('basis')}) {'/'.join(c.get('technology_ids', []))}: "
        f"{c.get('statement')}"
        for c in claims
    ) or "(없음)"
    fit_lines = "\n".join(
        f"- {f.get('technology_id')}: {f.get('assessment')} / 제약 {f.get('limitations')}"
        for f in fits
    ) or "(없음)"

    try:
        verdict = llm.with_structured_output(JudgeVerdict).invoke(
            [
                ("system", JUDGE_SYSTEM),
                (
                    "human",
                    JUDGE_USER.format(
                        requirements=requirements, claims=claim_lines, fits=fit_lines
                    ),
                ),
            ]
        )
    except Exception as exc:
        return JudgeResult(
            covered_axes=[], missing_axes=[], neutrality="unknown", neutrality_notes=[],
            model=model_name, error=f"{type(exc).__name__}: {exc}",
        )

    return JudgeResult(
        covered_axes=verdict.covered_axes,
        missing_axes=verdict.missing_axes,
        neutrality=verdict.neutrality,
        neutrality_notes=verdict.neutrality_notes,
        model=model_name,
    )

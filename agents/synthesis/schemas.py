"""OpenAIWriter 구조화 출력 스키마 (pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DraftClaim(BaseModel):
    technology: Literal["sw", "hw", "both"]
    text: str = Field(description="한 문장, 240자 이내")
    evidence_ids: list[str]
    conditions: list[str]
    limitations: list[str]


class Explanation(BaseModel):
    finding_id: str
    explanation: str
    resolved: bool


class Draft(BaseModel):
    claims: list[DraftClaim]
    explanations: list[Explanation]

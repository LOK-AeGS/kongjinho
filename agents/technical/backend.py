"""gpt-4.1-nano Structured Outputs 호출 계층."""

from __future__ import annotations

import json
import os

from .models import TechnicalExtraction
from .prompts import ANALYSIS_INSTRUCTIONS, PROMPT_VERSION


class OpenAIAnalyzer:
    def __init__(self, model: str | None = None, client=None, max_output_tokens: int = 8000):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
        self.client = client
        self.max_output_tokens = max_output_tokens

    def _client(self):
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("OpenAI 호출에는 openai 패키지가 필요합니다.") from exc
            self.client = OpenAI(timeout=120, max_retries=1)
        return self.client

    def extract(self, payload: dict, feedback: list[str] | None = None) -> TechnicalExtraction:
        body = dict(payload)
        body["revision_feedback"] = list(feedback or [])
        response = self._client().responses.parse(
            model=self.model,
            instructions=ANALYSIS_INSTRUCTIONS,
            input=json.dumps(body, ensure_ascii=False),
            text_format=TechnicalExtraction,
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        if response.status != "completed" or response.output_parsed is None:
            raise ValueError("technical_extraction_incomplete")
        return response.output_parsed

    @property
    def run_info(self) -> dict:
        return {"model": self.model, "prompt_version": PROMPT_VERSION, "api": "openai.responses.parse"}


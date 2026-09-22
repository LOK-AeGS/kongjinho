"""gpt-4.1-nano Structured Outputs 호출 계층."""

from __future__ import annotations

import json
import os

from .models import TechnicalExtraction
from .prompts import ANALYSIS_INSTRUCTIONS, PROMPT_VERSION


# 이 에이전트는 원문과 글자 단위로 일치하는 인용을 요구한다. 작은 모델은 원문을 옮기지
# 못하고 바꿔 써서 근거가 전부 기각되고 TRL 이 공백으로 남는다. 고정 코퍼스 실측 인용
# 통과율은 gpt-4.1-nano 0%, gpt-4.1-mini 23%, gpt-4.1 84%, gpt-5 91% 였다.
# gpt-5 는 91% 로 가장 정확하지만 한 번 실행에 323초가 걸려 기본값은 gpt-4.1 로 둔다.
DEFAULT_MODEL = "gpt-4.1"

# 추론 모델(gpt-5 계열)은 출력 한도를 추론 토큰으로 먼저 소진하고, 120초 안에 끝내지
# 못한다. 기본값으로는 응답이 잘리거나 타임아웃이 나면서 추출이 통째로 비어버린다.
DEFAULT_MAX_OUTPUT_TOKENS = 16000
DEFAULT_TIMEOUT_SECONDS = 180


class OpenAIAnalyzer:
    def __init__(self, model: str | None = None, client=None, max_output_tokens: int | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.client = client
        self.max_output_tokens = max_output_tokens or int(
            os.getenv("TECHNICAL_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
        )
        self.timeout = float(os.getenv("TECHNICAL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))

    def _client(self):
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("OpenAI 호출에는 openai 패키지가 필요합니다.") from exc
            self.client = OpenAI(timeout=self.timeout, max_retries=1)
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
            detail = getattr(response, "incomplete_details", None)
            raise ValueError(
                f"technical_extraction_incomplete (status={response.status}, detail={detail}, "
                f"model={self.model}, max_output_tokens={self.max_output_tokens})"
            )
        return response.output_parsed

    @property
    def run_info(self) -> dict:
        return {"model": self.model, "prompt_version": PROMPT_VERSION, "api": "openai.responses.parse"}


"""보고서 섹션을 실제 LLM 호출로 작성하는 adapter."""

from __future__ import annotations

import os
from typing import Annotated, Any

try:
    # LangChain/Pydantic의 Python 3.10 호환 structured-output 경로에서 권장한다.
    from typing_extensions import TypedDict
except ModuleNotFoundError:  # 오프라인 최소 테스트 환경
    from typing import TypedDict

from agents.report.prompts import SYSTEM_RULES, build_repair_prompt
from agents.report.state import ReportAgentDeps, SectionDraft, SectionId, ValidationIssue


class LLMSectionOutput(TypedDict):
    """모델이 반환해야 하는 섹션별 구조화 출력."""

    markdown: Annotated[str, "요구된 제목으로 시작하는 한국어 Markdown 섹션"]
    claim_ids: Annotated[
        list[str], "입력 payload에 실제로 존재하며 본문 작성에 사용한 claim_id"
    ]
    evidence_ids: Annotated[
        list[str], "입력 payload에 실제로 존재하며 본문에서 인용한 evidence_id"
    ]


class LLMSectionWriter:
    """LangChain chat model을 SectionWriter 계약에 연결한다."""

    receives_full_context = False

    def __init__(
        self,
        llm: object,
        *,
        model: str,
        provider: str,
        temperature: float,
    ) -> None:
        self.llm = llm
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self._structured = llm.with_structured_output(LLMSectionOutput)

    def _invoke(self, prompt: str) -> LLMSectionOutput:
        result = self._structured.invoke(
            [
                ("system", SYSTEM_RULES),
                ("human", prompt),
            ]
        )
        if not isinstance(result, dict):
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            else:
                raise TypeError("LLM structured output이 dict 형식이 아님")
        return {
            "markdown": str(result.get("markdown") or ""),
            "claim_ids": [str(value) for value in result.get("claim_ids", [])],
            "evidence_ids": [str(value) for value in result.get("evidence_ids", [])],
        }

    @staticmethod
    def _draft(section_id: SectionId, context: dict, output: LLMSectionOutput) -> SectionDraft:
        payload = context["payload"]
        return {
            "section_id": section_id,
            "title": str(payload["section_title"]),
            "markdown": output["markdown"].strip(),
            "claim_ids": list(dict.fromkeys(output["claim_ids"])),
            "evidence_ids": list(dict.fromkeys(output["evidence_ids"])),
        }

    def write(self, section_id: SectionId, context: dict) -> SectionDraft:
        return self._draft(section_id, context, self._invoke(context["prompt"]))

    def repair(
        self,
        section_id: SectionId,
        draft: SectionDraft,
        issues: list[ValidationIssue],
        context: dict,
    ) -> SectionDraft:
        prompt = build_repair_prompt(
            context["prompt"],
            draft=draft,
            issues=issues,
        )
        return self._draft(section_id, context, self._invoke(prompt))


def load_report_environment() -> bool:
    """환경값이 비어 있으면 저장소의 .env에서 OpenAI key를 자동 로드한다."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return bool(os.getenv("OPENAI_API_KEY"))
    load_dotenv(override=not bool(os.getenv("OPENAI_API_KEY")))
    return bool(os.getenv("OPENAI_API_KEY"))


def create_llm_writer(deps: ReportAgentDeps) -> LLMSectionWriter:
    """주입된 client 또는 설정값으로 실제 보고서 LLM writer를 만든다."""
    llm: Any = deps.llm
    if llm is None:
        if not load_report_environment():
            raise RuntimeError(
                "OPENAI_API_KEY가 없습니다. 저장소 루트의 .env에 설정하세요."
            )
        # 오프라인 테스트가 report 모듈을 import하는 것만으로 LangChain을 요구하지 않게 지연 import한다.
        from langchain.chat_models import init_chat_model

        llm = init_chat_model(
            deps.model,
            model_provider=deps.model_provider,
            temperature=deps.temperature,
        )
    return LLMSectionWriter(
        llm,
        model=deps.model,
        provider=deps.model_provider,
        temperature=deps.temperature,
    )

"""중립성 whole-report 판정과 측정값 인용 함의 판정을 수행한다."""

from __future__ import annotations

import os
import re
from decimal import Decimal, InvalidOperation

from agents.quality.checks import citation_ids, factual_body_lines
from agents.quality.prompts import (
    ENTAILMENT_SYSTEM_PROMPT,
    NEUTRALITY_SYSTEM_PROMPT,
    build_entailment_prompt,
    build_neutrality_prompt,
)
from agents.report.validators import _MEASUREMENT


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _usage(raw) -> dict[str, int]:
    usage = dict(getattr(raw, "usage_metadata", None) or {})
    token_usage = dict((getattr(raw, "response_metadata", None) or {}).get("token_usage") or {})
    input_tokens = usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", token_usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens", token_usage.get("total_tokens", input_tokens + output_tokens))
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in ("input_tokens", "output_tokens", "total_tokens")}


def _measurement_value(raw: str) -> str:
    number = raw.replace(",", "")
    number = "".join(character for character in number if character.isdigit() or character == ".")
    try:
        return format(Decimal(number).normalize(), "f")
    except (InvalidOperation, ValueError):
        return number


def _excerpt_window(sentence: str, quote: str, size: int = 400) -> str:
    """문장의 측정값과 같은 quote 측정값 주변을 보여주고, 없으면 선두를 사용한다."""
    wanted = {_measurement_value(match.group()) for match in _MEASUREMENT.finditer(sentence)}
    matched = next(
        (
            match for match in _NUMBER.finditer(quote)
            if _measurement_value(match.group()) in wanted
        ),
        None,
    )
    if matched is None:
        return quote[:size]
    center = (matched.start() + matched.end()) // 2
    start = max(0, center - size // 2)
    end = min(len(quote), start + size)
    start = max(0, end - size)
    window = quote[start:end]
    return ("…" if start else "") + window + ("…" if end < len(quote) else "")


def extract_entailment_items(markdown: str, evidence_store: dict, limit: int = 15) -> list[dict]:
    """인용된 측정값 문장과 최대 두 개의 근거 excerpt를 작은 judge 입력으로 만든다."""
    items = []
    for line in factual_body_lines(markdown):
        if not _MEASUREMENT.search(line):
            continue
        ids = citation_ids(line)
        if not ids:
            continue
        evidence = []
        for evidence_id in ids[:2]:
            raw = evidence_store.get(evidence_id) or {}
            quote = str(raw.get("quote") or raw.get("excerpt") or "")
            excerpt = _excerpt_window(line, quote)
            evidence.append({"evidence_id": evidence_id, "excerpt": excerpt})
        items.append({
            "index": len(items) + 1,
            "sentence": line,
            "evidence_ids": ids,
            "evidence": evidence,
        })
        if len(items) >= limit:
            break
    return items


def _parsed(result: dict) -> dict:
    if result.get("parsing_error") is not None:
        raise ValueError(f"structured output parsing failed: {result['parsing_error']}")
    parsed = result.get("parsed")
    return parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)


def make_judge(model: str | None = None):
    """LangChain/OpenAI를 실제 judge 생성 시점에만 import한다."""
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class CriterionVerdict(BaseModel):
        score: int = Field(ge=1, le=5)
        passed: bool
        reasons: str
        problem_sentences: list[str]

    class EntailmentItemVerdict(BaseModel):
        index: int = Field(ge=1)
        supported: bool
        reason: str

    class EntailmentVerdict(BaseModel):
        items: list[EntailmentItemVerdict]

    # v2 실험: gpt-4.1은 탐지·일관성 100%, score diff 0.00, 약 5.8초·$0.025/call이었다.
    # v1 whole-report judge가 bias/coverage를 놓쳐 두 기준은 code-only로 유지한다.
    chosen = model or os.getenv("QUALITY_JUDGE_MODEL") or "gpt-4.1"
    model_args = {"model": chosen}
    if not (chosen.startswith("gpt-5") or chosen.startswith("o")):
        model_args["temperature"] = 0
    chat = ChatOpenAI(**model_args)
    neutrality_runnable = chat.with_structured_output(CriterionVerdict, include_raw=True)
    entailment_runnable = chat.with_structured_output(EntailmentVerdict, include_raw=True)

    def judge(markdown: str, statuses: dict[str, str], evidence_store: dict) -> dict:
        neutrality_result = neutrality_runnable.invoke([
            ("system", NEUTRALITY_SYSTEM_PROMPT),
            ("human", build_neutrality_prompt(markdown, statuses)),
        ])
        neutrality = _parsed(neutrality_result)
        neutrality["passed"] = int(neutrality["score"]) >= 3
        neutrality["problem_sentences"] = list(neutrality.get("problem_sentences") or [])[:3]
        usage = _usage(neutrality_result.get("raw"))

        items = extract_entailment_items(markdown, evidence_store)
        unsupported = []
        unsupported_items = []
        reasons = []
        if items:
            entailment_result = entailment_runnable.invoke([
                ("system", ENTAILMENT_SYSTEM_PROMPT),
                ("human", build_entailment_prompt(items)),
            ])
            returned = {int(item["index"]): item for item in _parsed(entailment_result).get("items", [])}
            usage = _add_usage(usage, _usage(entailment_result.get("raw")))
            for item in items:
                verdict = returned.get(item["index"])
                supported = bool(verdict and verdict.get("supported"))
                if not supported:
                    unsupported.append(item["sentence"])
                    unsupported_items.append({
                        "sentence": item["sentence"],
                        "evidence_ids": item["evidence_ids"],
                    })
                    reason = (verdict or {}).get("reason") or "판정 결과 누락"
                    reasons.append(f"{item['index']}: {reason}")
        score = max(1, 5 - 2 * len(unsupported))
        groundedness = {
            "score": score,
            "passed": score >= 3,
            "reasons": "; ".join(reasons) if reasons else "모든 측정값 문장이 인용 excerpt로 지지됨",
            "problem_sentences": unsupported,
            "unsupported_count": len(unsupported),
            "unsupported_items": unsupported_items,
            "checked_items": len(items),
        }
        return {
            "model": chosen,
            "criteria": {"neutrality": neutrality, "groundedness": groundedness},
            "usage": usage,
        }

    judge.model = chosen
    return judge

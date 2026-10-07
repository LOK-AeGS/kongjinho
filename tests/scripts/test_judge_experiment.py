"""Judge 실험 도구의 네트워크 없는 변형·메트릭 검사."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

from agents.quality.checks import PROHIBITED, neutrality
from agents.quality.judge import extract_entailment_items, make_judge
from scripts.judge_experiment import (
    CRITERIA,
    build_variants,
    compute_metrics,
    evaluate_judges,
)


BASE_REPORT = """# SUMMARY

- 요약 사실이다. 〔근거: e1〕

# 2. 기술 선정

- 고정 기술이다.

## 4.1 기술 성숙도(TRL)

- 기술 사실이다. 〔근거: e1〕

## 4.2 시장성

- 시장 사실이다. 〔근거: e1〕

## 4.3 이해관계자

- 이해관계자 사실이다. 〔근거: e1〕

## 4.4 도메인 적용

- 도메인 사실이다. 〔근거: e1〕

## 5.1 비교 매트릭스

| 항목 | SW | HW |
|---|---|---|
| 성숙도 | 확인 | 확인 〔근거: e1〕 |

## 5.5 남은 확인 과제

- 없음

# 6. 한계점

- 공개 정보 한계

# REFERENCE

- 자료
"""


def _state():
    return {
        "evidence_store": {"e1": {"stance": "support", "quote": "측정 결과 처리량은 63% 증가하지 않았다."}},
        "technical_findings": {
            "records": [{"technology": "sw", "evidence_ids": ["e1"]}],
            "claims": [],
        },
    }


def test_variant_builders_make_the_documented_single_defects():
    variants = build_variants(BASE_REPORT, _state())
    assert set(variants) == {
        "clean", "neutrality_implicit", "groundedness_fabricated",
        "bias_onesided", "coverage_missing",
    }
    implicit = variants["neutrality_implicit"]
    assert "합리적이다" in implicit and "타당하다" in implicit
    assert not any(word in implicit for word in PROHIBITED)
    assert neutrality(implicit)["passed"]
    fabricated = variants["groundedness_fabricated"]
    assert "87.4%" in fabricated and "63%" in fabricated
    assert "87.4% 높였다.\n" in fabricated
    assert variants["bias_onesided"].count("SW 접근은") == 5
    market = re.search(
        r"^## 4\.2 시장성\s*$(.*?)(?=^#{1,2}\s+)",
        variants["coverage_missing"],
        re.MULTILINE | re.DOTALL,
    )
    assert market and not market.group(1).strip()


def test_fake_judge_metrics_are_deterministic_and_network_free():
    variants = build_variants(BASE_REPORT, _state())

    def factory(_model):
        def judge(markdown, _statuses, _evidence_store):
            target = None
            if "합리적이다" in markdown:
                target = "neutrality"
            elif "87.4%" in markdown:
                target = "groundedness"
            elif markdown.count("SW 접근은") == 5:
                target = "bias_control"
            elif re.search(r"^## 4\.2 시장성\s*\n\s*\n## 4\.3", markdown, re.MULTILINE):
                target = "coverage"
            criteria = {
                "neutrality": {
                    "score": 1 if target == "neutrality" else 5,
                    "passed": target != "neutrality",
                    "reasons": "fake",
                    "problem_sentences": [],
                },
                "groundedness": {
                    "score": 3 if target == "groundedness" else 5,
                    "passed": True,
                    "reasons": "fake",
                    "problem_sentences": ["- 동일 구성은 랙 전력을 63% 절감했다."] if target == "groundedness" else [],
                    "unsupported_count": 1 if target == "groundedness" else 0,
                },
            }
            return {
                "criteria": criteria,
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            }
        return judge

    records = evaluate_judges(variants, {}, _state()["evidence_store"], ["fake-model"], 2, factory)
    metrics = compute_metrics(records, ["fake-model"], 2)["fake-model"]
    assert len(records) == 10
    assert metrics["detection_rate"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["consistency_rate"] == 1.0
    assert metrics["mean_absolute_score_diff"] == 0.0
    assert metrics["total_tokens"] == 300
    assert metrics["errors"] == 0


def test_entailment_items_pair_measurement_sentence_with_excerpt():
    markdown = """# SUMMARY

## 4.4 도메인 적용

- 처리량은 63% 증가했다. 〔근거: e1, e2, e3〕
- 조건 설명이다. 〔근거: e1〕
"""
    store = {
        "e1": {"quote": "A" * 450},
        "e2": {"excerpt": "두 번째 excerpt"},
        "e3": {"quote": "세 번째 excerpt"},
    }
    items = extract_entailment_items(markdown, store)
    assert len(items) == 1
    assert items[0]["sentence"].startswith("- 처리량은 63%")
    assert [item["evidence_id"] for item in items[0]["evidence"]] == ["e1", "e2"]
    assert len(items[0]["evidence"][0]["excerpt"]) == 400
    assert items[0]["evidence"][0]["excerpt"] == "A" * 400


def test_entailment_excerpt_centers_window_on_normalized_measurement():
    markdown = "# SUMMARY\n\n## 4.4 도메인 적용\n\n- 처리량은 5.76배 증가했다. 〔근거: e1〕"
    quote = "앞" * 500 + " 측정 결과는 5.76×였다. " + "뒤" * 500
    items = extract_entailment_items(markdown, {"e1": {"quote": quote}})
    excerpt = items[0]["evidence"][0]["excerpt"]
    assert "5.76×" in excerpt
    assert excerpt.startswith("…") and excerpt.endswith("…")
    assert len(excerpt) <= 402


def test_make_judge_omits_temperature_for_reasoning_models(monkeypatch):
    calls = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def with_structured_output(self, schema, include_raw=False):
            assert schema is not None and include_raw is True
            return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))
    monkeypatch.delenv("QUALITY_JUDGE_MODEL", raising=False)
    make_judge("gpt-4.1-mini")
    make_judge("gpt-5-mini")
    make_judge("o3-mini")
    make_judge()
    assert calls[0] == {"model": "gpt-4.1-mini", "temperature": 0}
    assert calls[1] == {"model": "gpt-5-mini"}
    assert calls[2] == {"model": "o3-mini"}
    assert calls[3] == {"model": "gpt-4.1", "temperature": 0}

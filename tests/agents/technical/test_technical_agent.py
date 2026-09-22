from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.technical.config import DEFAULT_REQUEST, DEFAULT_SELECTED_TECH, OUTPUT_CONTRACT, fixed_input, validate_fixed_input, validate_output_contract
from agents.technical.evidence import make_evidence_id, validate_reference
from agents.technical.models import TechnicalExtraction
from agents.technical.node import make_node
from agents.technical.subgraph import TechnicalAgentDeps
from agents.technical.trl import determine_trl, validate_metric_records
from agents.technical.web import TavilyProvider


class FakeHit:
    def __init__(self, candidate):
        self.candidate = candidate

    def to_candidate(self):
        return deepcopy(self.candidate)


class FakeRetriever:
    run_info = {"retrieval": "fake", "embedding_model": None, "embedding_device": "cpu"}

    def search(self, technology, criterion, query, top_k=4):
        quote = f"{technology} {criterion} validated statement"
        candidate = {
            "chunk_id": f"technical:chunk:{technology}:{criterion}", "doc_id": f"doc:{technology}",
            "title": f"{technology} primary paper", "author_or_org": "authors",
            "url": f"https://arxiv.org/{technology}", "published_at": "2026-01-01",
            "role": "primary", "technology": technology, "approach": "test",
            "page": 1, "locator": f"p.1:{criterion}", "text": quote,
            "content_hash": f"hash-{technology}-{criterion}", "origin": "pdf",
            "rrf_score": 1.0, "queries": [query], "matched_criteria": [criterion],
        }
        return [FakeHit(candidate)]


class FakeWeb:
    def __init__(self, found=True):
        self.found = found
        self.calls = []

    def search_and_extract(self, technology, query, as_of):
        self.calls.append((technology, query, as_of))
        if not self.found:
            return [], {"provider": "tavily", "technology": technology, "query": query, "status": "not_found"}
        text = f"{technology} data center LLM inference serving pilot evidence"
        candidate = {
            "chunk_id": f"technical:web:{technology}", "doc_id": None,
            "title": f"{technology} operator report", "author_or_org": "operator.example",
            "url": f"https://operator.example/{technology}", "published_at": "2026-02-01",
            "role": "web_operational", "technology": technology, "approach": "operational_evidence",
            "page": None, "locator": f"web:{technology}", "text": text,
            "content_hash": f"web-hash-{technology}", "origin": "tavily",
            "source_type": "official_web", "primary_or_secondary": "primary", "direct_or_proxy": "direct",
            "rrf_score": 1.0, "queries": [query], "matched_criteria": ["operational_evidence"],
        }
        return [candidate], {"provider": "tavily", "technology": technology, "query": query, "status": "ok", "accepted_urls": [candidate["url"]]}


def _ref(technology, criterion):
    return {"candidate_id": f"technical:chunk:{technology}:{criterion}", "quote": f"{technology} {criterion} validated statement"}


class FakeAnalyzer:
    run_info = {"model": "fake", "prompt_version": "test"}

    def extract(self, payload, feedback=None):
        records = []
        for technology in ("sw", "hw"):
            for criterion in ("mechanism", "application_scope", "performance", "limitations", "validation_environment"):
                records.append({
                    "technology": technology, "criterion": criterion, "assessment": "supported",
                    "findings": f"{technology} {criterion} 평가", "evidence_refs": [_ref(technology, criterion)],
                    "conditions": [], "limitations": [], "source_scope": "direct",
                })
        observations = []
        for technology in ("sw", "hw"):
            observations.append({
                "technology": technology, "summary": f"{technology} 관련 환경에서 프로토타입을 평가함",
                "artifact_level": "component" if technology == "sw" else "system_prototype",
                "environment": "relevant", "activity": "experiment", "evidence_level": "pilot",
                "source_scope": "direct", "real_llm": True, "real_accelerator": True,
                "representative_workload": True, "representative_scale": technology == "hw",
                "operational_requirements_validated": False,
                "evidence_refs": [_ref(technology, "validation_environment")],
                "blocking_facts": [], "missing_facts": ["지속 production 운용 공개 근거 미확인"],
            })
        return TechnicalExtraction.model_validate({
            "records": records,
            "claims": [{"technology": "sw", "text": "MLA는 KV 표현을 압축한다.", "evidence_refs": [_ref("sw", "mechanism")], "conditions": [], "limitations": []}],
            "metrics": [], "readiness_observations": observations, "gaps": [],
        })


class BrokenQuoteAnalyzer(FakeAnalyzer):
    def extract(self, payload, feedback=None):
        value = super().extract(payload, feedback).model_dump()
        value["records"][0]["evidence_refs"][0]["quote"] = "invented quotation"
        return TechnicalExtraction.model_validate(value)


def _deps(web_found=True):
    web = FakeWeb(found=web_found)
    return TechnicalAgentDeps(FakeRetriever(), web, FakeAnalyzer()), web


def test_fixed_input_is_exact_and_rejects_changed_technology():
    request, selected = fixed_input()
    assert request == DEFAULT_REQUEST
    assert selected == DEFAULT_SELECTED_TECH
    changed = deepcopy(selected)
    changed["sw"]["short_name"] = "other"
    with pytest.raises(ValueError):
        validate_fixed_input(request, changed)
    input_file = Path(__file__).resolve().parents[3] / "data" / "technical" / "default_input.json"
    saved = json.loads(input_file.read_text(encoding="utf-8"))
    assert saved == {"request": request, "selected_tech": selected}
    contract_file = input_file.with_name("output_contract.json")
    saved_contract = json.loads(contract_file.read_text(encoding="utf-8"))
    assert all(saved_contract[key] == sorted(value) for key, value in OUTPUT_CONTRACT.items())


def test_node_output_contract_and_determinism():
    deps, web = _deps()
    node = make_node(deps, corpus_manifest=[])
    state = {"request": deepcopy(DEFAULT_REQUEST), "selected_tech": deepcopy(DEFAULT_SELECTED_TECH)}
    first = node(state)
    second = node(state)
    assert set(first) == {"technical_findings", "evidence_store"}
    validate_output_contract(first)
    assert first == second
    findings = first["technical_findings"]
    assert findings["perspective"] == "technical"
    assert findings["status"] == "complete"
    assert len([item for item in findings["records"] if item["criterion"] == "trl"]) == 2
    assert findings["input_evidence_ids"] == sorted(first["evidence_store"])
    claim_ids = {item["claim_id"] for item in findings["claims"]}
    assert all(item["claim_id"] in claim_ids for item in first["evidence_store"].values())
    assert findings["meta"]["search_rounds_used"] == 1
    assert len(web.calls) == 4


def test_search_stops_after_two_rounds_and_marks_public_evidence_unknown():
    deps, _ = _deps(web_found=False)
    result = make_node(deps, corpus_manifest=[])(
        {"request": deepcopy(DEFAULT_REQUEST), "selected_tech": deepcopy(DEFAULT_SELECTED_TECH)}
    )
    findings = result["technical_findings"]
    assert findings["meta"]["search_rounds_used"] == 2
    reasons = [item["reason"] for item in findings["gaps"]]
    assert any("operational_evidence" in reason and "부재를 의미하지 않음" in reason for reason in reasons)
    assert not any("상용되지" in reason for reason in reasons)


def test_failed_reference_is_revised_once_then_returned_as_partial():
    deps = TechnicalAgentDeps(FakeRetriever(), FakeWeb(), BrokenQuoteAnalyzer())
    result = make_node(deps, corpus_manifest=[])(
        {"request": deepcopy(DEFAULT_REQUEST), "selected_tech": deepcopy(DEFAULT_SELECTED_TECH)}
    )
    findings = result["technical_findings"]
    assert findings["status"] == "partial"
    assert findings["meta"]["revision_rounds_used"] == 1
    assert findings["meta"]["quality_report"]["status"] == "failed"
    assert any("원문에 없는" in item["reason"] for item in findings["gaps"])


def test_quote_and_evidence_id_guards():
    candidate = {"chunk_id": "c1", "text": "exact source quotation"}
    valid, error = validate_reference({"candidate_id": "c1", "quote": "source quotation"}, {"c1": candidate})
    assert valid is candidate and error is None
    invalid, error = validate_reference({"candidate_id": "c1", "quote": "invented"}, {"c1": candidate})
    assert invalid is None and "원문에 없는" in error
    assert make_evidence_id("url", "p.1", "same quote") == make_evidence_id("url", "p.1", "same   quote")


def _observation(**updates):
    value = {
        "technology": "sw", "summary": "direct observation", "artifact_level": "final_system",
        "environment": "operational", "activity": "sustained_operation", "evidence_level": "production",
        "source_scope": "direct", "real_llm": True, "real_accelerator": True,
        "representative_workload": True, "representative_scale": True,
        "operational_requirements_validated": True, "evidence_ids": ["ev:1"],
        "blocking_facts": [], "missing_facts": [],
    }
    value.update(updates)
    return value


def test_trl_is_contiguous_and_api_announcement_cannot_be_operational_stage():
    assert determine_trl("sw", [_observation()]).level == 9
    announcement = _observation(environment="lab", activity="analysis", evidence_level="announcement", representative_scale=False)
    result = determine_trl("sw", [announcement])
    assert (result.level or 0) < 7
    assert all(not gate["satisfied"] for gate in result.gate_trace[6:])
    near_pilot_announcement = _observation(activity="deployment", evidence_level="announcement")
    result = determine_trl("sw", [near_pilot_announcement])
    assert result.level == 6
    assert result.level_range == [6]


def test_metric_guards_keep_baselines_separate():
    metrics, warnings = validate_metric_records([
        {"technology": "hw", "metric": "throughput", "value": "1.80x", "baseline": None, "hardware": None, "model": None, "context": None, "is_maximum": False, "is_upper_bound": False, "attributable_to_selected_technology": True},
        {"technology": "sw", "metric": "generation throughput", "value": "5.76x", "baseline": "DeepSeek 67B", "hardware": "8xH800", "model": "DeepSeek-V2", "context": "maximum", "is_maximum": False, "is_upper_bound": False, "attributable_to_selected_technology": True},
    ])
    assert any("NVMe-oF" in value for value in warnings)
    assert any("maximum" in value for value in warnings)
    assert all(item["validation_issues"] for item in metrics)


class FakeTavilyClient:
    def __init__(self):
        self.search_kwargs = None
        self.extract_kwargs = None

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return {"request_id": "s1", "results": [{"url": "https://deepseek.com/case", "title": "case", "published_date": "2026-01-01", "score": 0.9}]}

    def extract(self, **kwargs):
        self.extract_kwargs = kwargs
        return {"request_id": "e1", "results": [{"url": "https://deepseek.com/case", "raw_content": "DeepSeek-V2 MLA data center LLM inference production pilot customer deployment"}], "failed_results": []}


def test_tavily_uses_search_then_extract_and_never_answer(tmp_path):
    client = FakeTavilyClient()
    provider = TavilyProvider(client=client, cache_dir=tmp_path)
    candidates, log = provider.search_and_extract("sw", "query", "2026-09-22")
    assert candidates and log["status"] == "ok"
    assert client.search_kwargs["include_answer"] is False
    assert client.search_kwargs["include_raw_content"] is False
    assert client.search_kwargs["end_date"] == "2026-09-22"
    assert client.extract_kwargs["extract_depth"] == "advanced"
    assert "answer" not in candidates[0]

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.graph import END, START, StateGraph

from agents.stakeholder.backend import OpenAIBackend, unpack_search
from agents.stakeholder.evidence import merge_evidence
from agents.stakeholder.models import Extraction, Observation
from agents.stakeholder.node import build_request, make_node
from agents.stakeholder.subgraph import GROUPS, default_request, run_stakeholder, validate_observations
from agents.stakeholder.web import PageFetcher, digest, parse_html
from graph.state import AppState, create_initial_state, merge_evidence_store
from graph.workers import as_worker

URL = "https://example.com/source"
TEXT = "The deployment has operational benefits but migration requires careful planning."


def observation(**updates):
    value = dict(
        technology_id="sw", target_name="DeepSeek-V2 MLA",
        target_scope="selected_technology", group="adopter",
        speaker="Fixture engineer", affiliation=None, stance="positive",
        evidence_stance="support", domain_relevance="datacenter",
        statement="운영상 이점이 있다는 테스트 발언", source_url=URL,
        source_title="Fixture source", source_type="official_web",
        published_date="2026-01-01", primary_or_secondary="primary",
        direct_or_proxy="direct", evidence_level="unknown",
        page_or_locator="block:0001", quote=TEXT, conditions=[],
        uncertainty="합성 테스트 자료", bias_notes=[],
    )
    value.update(updates)
    return Observation(**value)


def batch():
    blocks = [{"locator": "block:0001", "text": TEXT}]
    page = dict(
        url=URL, final_url=URL, status="ok", blocks=blocks,
        metadata={"article:published_time": "2026-01-01"},
        content_hash=digest(json.dumps(blocks, ensure_ascii=False, sort_keys=True)),
        accessed_at="2026-09-21T00:00:00+00:00", snapshot_path=None,
    )
    logs = [
        dict(id=f"{side}:{group}:r1", technology_id=side, group=group,
             query="fixture support counter neutral", status="ok", urls=[URL],
             provider="offline_fixture", as_of_date="2026-09-21")
        for side in ("sw", "hw") for group in GROUPS
    ]
    return {"pages": {URL: page}, "search_logs": logs, "errors": []}


class FakeBackend:
    def __init__(self, items=None, data=None):
        self.items = [observation()] if items is None else items
        self.data = data or batch()
        self.calls = []

    def search(self, request, queries, technical):
        self.calls.append(deepcopy((request, queries, technical)))
        result = deepcopy(self.data)
        ids = {query["id"] for query in queries}
        result["search_logs"] = [query for query in result["search_logs"] if query["id"] in ids]
        return result

    def extract(self, *args):
        return Extraction(observations=self.items, gaps=[])


def app_state(**extra):
    request = default_request()
    state = create_initial_state(
        request={"as_of": "2026-09-21", "language": "ko", "scope": "t", "max_search_rounds": 2},
        selected_tech={
            side: {
                "name": request[side]["name"], "short_name": side,
                "technology": side, "approach": "", "source_ids": [],
                "selection_reason": "t",
            } for side in ("sw", "hw")
        },
        corpus_manifest=[],
    )
    state.update(extra)
    return state


def test_counter_not_found_is_allowed_with_logs():
    result = run_stakeholder(backend=FakeBackend())["result"]
    assert result["completion"]["status"] == "complete"
    assert all(
        item["status"] == "not_found" and item["query_ids"]
        for item in result["search_outcomes"] if item["stance"] == "counter"
    )
    assert len(result["evidence_store"]) == 1


def test_no_results_has_no_fabricated_evidence():
    data = batch()
    data["pages"] = {}
    for query in data["search_logs"]:
        query.update(status="no_results", urls=[])
    result = run_stakeholder(backend=FakeBackend([], data))["result"]
    assert not result["claims"]
    assert all(item["status"] == "not_found" for item in result["search_outcomes"])


def test_access_failure_is_blocked():
    data = batch()
    data["pages"][URL]["status"] = "paywall"
    for query in data["search_logs"]:
        query["status"] = "access_incomplete"
    result = run_stakeholder(backend=FakeBackend([], data))["result"]
    assert result["completion"]["status"] == "partial"
    assert all(item["status"] == "blocked" for item in result["search_outcomes"])


def test_quote_locator_hash_url_gate():
    for update in ({"quote": "invented"}, {"page_or_locator": "block:9999"}, {"source_url": "https://fake.example/"}):
        accepted, rejected = validate_observations(
            Extraction(observations=[observation(**update)], gaps=[]), [batch()], "2026-09-21"
        )
        assert not accepted and rejected
    data = batch()
    data["pages"][URL]["content_hash"] = "tampered"
    assert not validate_observations(
        Extraction(observations=[observation()], gaps=[]), [data], "2026-09-21"
    )[0]


@pytest.mark.parametrize("update", [
    {"published_date": "2027-01-01"}, {"statement": "성능 35.7% 향상"},
    {"statement": "5 GB/s"}, {"domain_relevance": "other"},
])
def test_date_numeric_domain_gates(update):
    assert not validate_observations(
        Extraction(observations=[observation(**update)], gaps=[]), [batch()], "2026-09-21"
    )[0]


def test_unverified_date_is_partial():
    result = run_stakeholder(
        backend=FakeBackend([observation(published_date="2026-02-01")])
    )["result"]
    assert result["completion"]["status"] == "partial"
    assert next(iter(result["evidence_store"].values()))["published_at"] is None


def test_shared_quote_has_multiple_claim_links():
    result = run_stakeholder(backend=FakeBackend([
        observation(), observation(group="investor", statement="다른 해석")
    ]))["result"]
    assert len(result["evidence_store"]) == 1
    assert len(next(iter(result["evidence_store"].values()))["claim_ids"]) == 2


def test_reducer_idempotence_associativity_and_collision():
    first = run_stakeholder(backend=FakeBackend())["result"]["evidence_store"]
    before = deepcopy(first)
    second = deepcopy(first)
    evidence = next(iter(second.values()))
    evidence.update(
        claim_id="market:x", claim_ids=["market:x"], perspective="market",
        perspectives=["market"],
        bindings=[{"claim_id": "market:x", "perspective": "market", "stance": "neutral"}],
    )
    third = deepcopy(second)
    next(iter(third.values()))["accessed_at"] = "2026-09-22"
    assert merge_evidence(first, first) == first
    assert merge_evidence(first, second) == merge_evidence(second, first)
    assert merge_evidence(merge_evidence(first, second), third) == merge_evidence(first, merge_evidence(second, third))
    assert first == before
    next(iter(second.values()))["quote"] = "collision"
    with pytest.raises(ValueError):
        merge_evidence(first, second)


def test_query_budget_and_unsearched_status():
    request = default_request()
    request["max_queries"] = 1
    backend = FakeBackend([])
    result = run_stakeholder(request, backend=backend)["result"]
    assert len(backend.calls) == 1
    assert len(backend.calls[0][1]) == 1
    assert any(item["status"] == "unsearched" for item in result["search_outcomes"])


def test_node_returns_appstate_keys_and_projects_input():
    state = app_state(
        technical_findings={"claims": [{"text": "기술 주장"}], "secret_field": "do not share"},
        market_findings={"secret": "do not share"},
    )
    backend = FakeBackend()
    update = make_node(backend)(state)
    assert set(update) == {
        "stakeholder_findings", "evidence_store", "search_log_by_perspective",
        "quality_by_perspective",
    }
    assert "secret" not in str(backend.calls)
    assert backend.calls[0][2] == {"claims": ["기술 주장"]}
    assert update["stakeholder_findings"]["claims"]
    graph = StateGraph(AppState)
    graph.add_node("a", lambda _: update)
    graph.add_node("b", lambda _: {"evidence_store": update["evidence_store"]})
    graph.add_node("join", lambda _: {})
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge(["a", "b"], "join")
    graph.add_edge("join", END)
    assert len(graph.compile().invoke(state)["evidence_store"]) == 1


def test_non_datacenter_domain_is_rejected():
    with pytest.raises(ValueError):
        make_node(FakeBackend())(app_state(domain="ondevice"))


def test_rework_hint_targets_pairs_and_focus():
    state = app_state(rework_hint={
        "extra_rounds": 1,
        "focus_queries": ["공식 발표"],
        "gaps": [{"technology": "hw", "criterion": "투자·산업 관계자"}],
    })
    state["request"]["max_search_rounds"] = 3
    request = build_request(state)
    assert request["only_pairs"] == [["hw", "investor"]]
    backend = FakeBackend([])
    make_node(backend)(state)
    assert [query["id"] for query in backend.calls[0][1]] == ["hw:investor:r1"]
    assert "공식 발표" in backend.calls[0][1][0]["query"]


def test_worker_builds_rework_hint_from_matching_current_gaps():
    captured = {}

    def node(view):
        captured.update(view["rework_hint"])
        return {}

    state = app_state(
        stakeholder_findings={"gaps": [
            {"technology": "sw", "criterion": "경쟁 기술 진영"},
            {"technology": "hw", "criterion": "투자·산업 관계자"},
        ]},
        rework={"stakeholder": {
            "reason": "근거 부족",
            "focus": ["hw/투자·산업 관계자"],
            "round": 1,
            "max_search_rounds": 3,
            "feedback": [],
            "created_step": 2,
        }},
        node_status={"stakeholder": {"attempts": 1}},
        step_count=2,
    )
    as_worker("stakeholder", node)(state)
    assert captured["focus_queries"] == ["hw/투자·산업 관계자"]
    assert captured["gaps"] == [
        {"technology": "hw", "criterion": "투자·산업 관계자"}
    ]


def test_hallucinated_opinions_never_reach_evidence_store():
    fake = [
        observation(quote="LLM이 지어낸 인용문"),
        observation(page_or_locator="block:9999", statement="다른 지어낸 발언"),
        observation(source_url="https://fake.example/x", statement="가짜 출처"),
        observation(published_date="2030-01-01", statement="미래 날짜"),
    ]
    update = make_node(FakeBackend(fake))(app_state())
    assert not update["evidence_store"]
    assert not update["stakeholder_findings"]["claims"]
    assert update["stakeholder_findings"]["status"] == "partial"
    assert update["quality_by_perspective"]["stakeholder"]["status"] == "needs_review"


def test_not_found_is_gap_without_fake_evidence():
    data = batch()
    data["pages"] = {}
    for query in data["search_logs"]:
        query.update(status="no_results", urls=[])
    findings = make_node(FakeBackend([], data))(app_state())["stakeholder_findings"]
    assert not findings["claims"]
    assert all(gap["reason"].startswith("not_found") for gap in findings["gaps"])


def test_family_is_not_direct():
    findings = make_node(FakeBackend([
        observation(target_scope="technology_family")
    ]))(app_state())["stakeholder_findings"]
    assert findings["records"][0]["scope"] == "class"
    assert findings["records"][0]["basis"] == "inferred"


def test_merge_with_parent_reducer_is_idempotent():
    store = make_node(FakeBackend())(app_state())["evidence_store"]
    assert merge_evidence_store(store, store) == store


def test_errors_are_redacted():
    class Bad(FakeBackend):
        def search(self, *args):
            raise RuntimeError("SECRET")

    final = run_stakeholder(backend=Bad())
    assert "SECRET" not in str(final)
    assert final["result"]["completion"]["status"] == "failed"


def test_parser_and_fetch_cache():
    blocks, _ = parse_html(
        '<meta name="date" content="2026-01-01"><p>Hello</p><script>bad()</script><p>World</p>'
    )
    assert [block["text"] for block in blocks] == ["Hello", "World"]

    class Fetcher(PageFetcher):
        def _get(self, url):
            return 200, {"content-type": "text/html"}, ("<p>" + ("original " * 30) + "</p>").encode(), url

    with tempfile.TemporaryDirectory() as directory:
        fetcher = Fetcher(directory, min_interval=0)
        result = fetcher.fetch(URL)
        assert result["status"] == "ok"
        assert Path(result["snapshot_path"]).exists()
        assert fetcher.fetch(URL) == result


def test_extract_uses_original_not_notes():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status="completed", output_parsed=Extraction(observations=[], gaps=[])
        )

    data = batch()
    data["search_logs"][0]["discovery_notes"] = "DO_NOT_USE_SUMMARY"
    backend = OpenAIBackend(client=SimpleNamespace(responses=SimpleNamespace(parse=parse)))
    backend.extract(default_request(), [data], [])
    assert "DO_NOT_USE_SUMMARY" not in captured["input"]
    assert TEXT in captured["input"]


def test_search_requires_completed_search_action():
    response = SimpleNamespace(
        status="completed", output_text="", id="x",
        model_dump=lambda: {"output": [{
            "type": "web_search_call", "status": "completed",
            "action": {"type": "search", "queries": ["test"], "sources": []},
        }]},
    )
    assert unpack_search(response)["actions"][0]["queries"] == ["test"]
    response.status = "incomplete"
    with pytest.raises(ValueError):
        unpack_search(response)

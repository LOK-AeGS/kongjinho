"""Supervisor 라우팅·종료·품질 루프의 API-key-free 검사."""

from __future__ import annotations

import json

from agents.quality.checks import bias_control, coverage, groundedness, neutrality
from agents.quality.node import quality_agent
from graph.build import WORKERS, build_graph
from graph.decision_log import DecisionLogger
from graph.stubs import load_fixture
from graph.supervisor import SupervisorPolicy, assess_sufficiency, decide
from graph.workers import as_worker
from main import build_nodes, initial_state, run_graph


def test_assess_sufficiency_complete_partial_and_failed():
    policy = SupervisorPolicy(min_evidence={name: 2 for name in ("technical", "market", "stakeholder", "domain")})
    store = {"e1": {}, "e2": {}}
    complete = {"status": "complete", "records": [], "claims": [], "gaps": []}
    enough = {"status": "partial", "records": [
        {"basis": "direct", "evidence_ids": ["e1"]},
        {"basis": "inferred", "evidence_ids": ["e2"]},
    ], "claims": [], "gaps": []}
    lacking = {"status": "partial", "records": [{"basis": "unknown", "evidence_ids": ["e1"]}], "claims": [], "gaps": []}
    failed = {"status": "failed", "records": [], "claims": [], "gaps": []}
    assert assess_sufficiency("market", complete, store, policy)[0] == "sufficient"
    assert assess_sufficiency("market", enough, store, policy)[0] == "sufficient"
    assert assess_sufficiency("market", lacking, store, policy)[0] == "insufficient"
    assert assess_sufficiency("market", failed, store, policy)[0] == "insufficient"


def test_partial_fixture_reworks_then_accepts_insufficient():
    fixture = load_fixture()
    nodes, _ = build_nodes(set(), fixture)
    policy = SupervisorPolicy(min_evidence={"technical": 3, "market": 3, "stakeholder": 3, "domain": 6})
    final, _ = run_graph(nodes, initial_state(), policy=policy)
    for name in ("stakeholder", "domain"):
        assert final["node_status"][name]["attempts"] == 2
        assert final["node_status"][name]["sufficiency"] == "accepted_insufficient"


def test_every_worker_only_returns_to_supervisor():
    nodes, _ = build_nodes(set(), load_fixture())
    edges = build_graph(**nodes).get_graph().edges
    for worker in WORKERS:
        assert {(edge.target, edge.conditional) for edge in edges if edge.source == worker} == {("supervisor", False)}


def test_max_steps_forces_end():
    nodes, _ = build_nodes(set(), load_fixture())
    final, _ = run_graph(nodes, initial_state(max_steps=1))
    assert final["last_decision"]["next"] == []
    assert "max_steps" in final["last_decision"]["reason"]


def test_failed_quality_rewrites_report_until_version_limit():
    fixture = load_fixture()
    nodes, _ = build_nodes(set(), fixture)

    def fail_quality(state):
        return {"eval_result": {
            "passed": False,
            "mode": "code_only",
            "criteria": {},
            "failed_criteria": ["neutrality"],
            "feedback": ["neutrality: 금지 표현 수정"],
            "rework_targets": [],
            "judge_model": None,
            "evaluated_report_version": state["report_version"],
        }}

    nodes["quality_eval"] = as_worker("quality_eval", fail_quality)
    final, _ = run_graph(nodes, initial_state())
    assert final["report_version"] == 2
    assert final["node_status"]["report"]["attempts"] == 2
    assert "품질 루프 상한" in final["last_decision"]["reason"]


def test_quality_failure_at_report_cap_does_not_dispatch_perspective_rework():
    complete = {"status": "complete", "records": [], "claims": [], "gaps": []}
    status = {
        name: {"status": "done", "attempts": 1, "completed_step": 1}
        for name in ("technical", "market", "stakeholder", "domain")
    }
    status.update({
        "synthesis": {"status": "done", "attempts": 1, "completed_step": 2},
        "report": {"status": "done", "attempts": 2, "completed_step": 3},
        "quality_eval": {"status": "done", "attempts": 2, "completed_step": 4},
    })
    state = {
        "step_count": 4,
        "max_steps": 20,
        "node_status": status,
        "technical_findings": complete,
        "market_findings": complete,
        "stakeholder_findings": complete,
        "domain_findings": complete,
        "evidence_store": {},
        "synthesis": {"status": "complete"},
        "report_version": 2,
        "eval_result": {
            "passed": False,
            "evaluated_report_version": 2,
            "feedback": ["groundedness: domain 근거 연결 수정"],
            "rework_targets": ["domain"],
        },
        "rework": {},
    }
    update = decide(state)
    assert update["next"] == []
    assert update["last_decision"]["reason"] == "품질 루프 상한 도달, needs_review로 종료"
    assert "domain" not in update["rework"]
    assert "report" not in update["rework"]


def test_worker_exception_is_failed_and_graph_continues():
    nodes, _ = build_nodes(set(), load_fixture())

    def broken(_state):
        raise RuntimeError("boom")

    nodes["domain"] = as_worker("domain", broken)
    final, trace = run_graph(nodes, initial_state())
    assert not [item for item in trace if item["error"]]
    assert final["node_status"]["domain"]["attempts"] == 2
    assert final["node_status"]["domain"]["sufficiency"] == "accepted_insufficient"
    assert final["synthesis"] is not None


def test_decision_log_contains_trace_and_reason(tmp_path):
    nodes, _ = build_nodes(set(), load_fixture())
    state = initial_state(max_steps=1)
    path = tmp_path / "decisions.jsonl"
    run_graph(nodes, state, decision_logger=DecisionLogger(path))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["trace_id"] == state["trace_id"]
    assert rows[0]["reason"]
    assert rows[-1]["decision"] == []


def test_quality_code_checks_detect_required_failures():
    cited = "- 사실 A 〔근거: e1〕\n- 사실 B\n- 추천 기술이다. 〔근거: e1〕"
    store = {"e1": {"url": "https://one.example/a", "stance": "support"}}
    findings = {"technical": {"records": [{"technology": "sw", "evidence_ids": ["e1"]}], "claims": []}}
    assert not groundedness(cited, store)["passed"]
    assert not neutrality(cited)["passed"]
    assert not bias_control(cited, store, findings)["passed"]
    assert not coverage("## 4.1 기술 성숙도(TRL)\n- 값 〔근거: e1〕")["passed"]


def test_groundedness_meta_line_without_citation_does_not_lower_ratio():
    markdown = """# SUMMARY

- 평가 범위: datacenter_inference / 조사 기준일: 2026-09-22

## 4.2 시장성

- 확인된 시장 사실이다. 〔근거: e1〕
"""
    result = groundedness(markdown, {"e1": {}})
    assert result["score"] == 1.0
    assert "(1/1)" in result["details"][0]


def test_groundedness_real_uncited_market_fact_lowers_ratio():
    markdown = """# SUMMARY

## 4.2 시장성

- 확인된 시장 사실이다. 〔근거: e1〕
- 실제 시장 규모가 증가했다.
"""
    result = groundedness(markdown, {"e1": {}})
    assert result["score"] == 0.5
    assert not result["passed"]


def test_groundedness_uncited_measurement_is_hard_failure_above_ratio_threshold():
    cited = "\n".join(f"- 사실 {index}이다. 〔근거: e1〕" for index in range(5))
    markdown = f"# SUMMARY\n\n## 4.2 시장성\n\n{cited}\n- 처리량은 87.4% 증가했다."
    result = groundedness(markdown, {"e1": {}})
    assert result["score"] > 0.8
    assert not result["passed"]
    assert any("측정값 무인용" in detail and "87.4%" in detail for detail in result["details"])


def test_groundedness_skips_table_header_row():
    markdown = """# SUMMARY

## 4.2 시장성

| 항목 | 결과 |
|---|---|
| 시장 규모 | 증가 〔근거: e1〕 |
"""
    result = groundedness(markdown, {"e1": {}})
    assert result["score"] == 1.0
    assert "(1/1)" in result["details"][0]


def test_report_blocking_violation_fails_groundedness_and_becomes_feedback():
    violation = "35.7%는 최대값으로 표시해야 함"
    state = {
        "report_sections": {"final_markdown_with_ids": "# SUMMARY\n\n- 사실이다. 〔근거: e1〕"},
        "evidence_store": {"e1": {}},
        "quality_by_perspective": {"report": {
            "status": "needs_review",
            "violations": [violation],
            "warnings": [],
            "checked_claim_ids": [],
        }},
        "report_version": 1,
    }
    result = quality_agent(state)
    grounded = result["eval_result"]["criteria"]["groundedness"]
    assert not grounded["passed"]
    assert violation in grounded["code"]["details"]
    assert f"groundedness: {violation}" in result["eval_result"]["feedback"]


def test_bias_control_detects_per_section_single_source_concentration():
    store = {
        key: {"url": "https://same.example/report", "stance": "support"}
        for key in ("e1", "e2", "e3")
    }
    findings = {
        "market": {"records": [
            {"technology": "sw", "evidence_ids": ["e1"]},
            {"technology": "hw", "evidence_ids": ["e2"]},
            {"technology": "both", "evidence_ids": ["e3"]},
        ], "claims": []},
    }
    markdown = "# SUMMARY\n\n## 4.2 시장성\n\n" + "\n".join(
        f"- 시장 사실 {index}. 〔근거: e{index}〕" for index in range(1, 4)
    )
    result = bias_control(markdown, store, findings)
    assert not result["passed"]
    assert "4.2 시장성: 단일 출처 편중 1.00" in result["details"]


def test_bias_control_warns_but_passes_at_point_seven_section_share():
    store = {}
    records = []
    section_lines = []
    for index in range(1, 11):
        evidence_id = f"section-{index}"
        url = "https://dominant.example/report" if index <= 7 else f"https://section-{index}.example/report"
        store[evidence_id] = {"url": url, "stance": "support"}
        records.append({"technology": "both", "evidence_ids": [evidence_id]})
        section_lines.append(f"- 시장 사실 {index}. 〔근거: {evidence_id}〕")
    summary_lines = []
    for index in range(1, 11):
        evidence_id = f"summary-{index}"
        store[evidence_id] = {"url": f"https://summary-{index}.example/report", "stance": "support"}
        records.append({"technology": "both", "evidence_ids": [evidence_id]})
        summary_lines.append(f"- 요약 사실 {index}. 〔근거: {evidence_id}〕")
    markdown = "# SUMMARY\n\n" + "\n".join(summary_lines) + "\n\n## 4.2 시장성\n\n" + "\n".join(section_lines)
    result = bias_control(markdown, store, {"market": {"records": records, "claims": []}})
    assert result["passed"]
    assert "경고: 4.2 시장성: 단일 출처 편중 0.70" in result["details"]


def test_bias_control_detects_section_covering_only_one_technology():
    store = {
        f"e{index}": {"url": f"https://source{index}.example/report", "stance": "support"}
        for index in range(1, 4)
    }
    findings = {"market": {"records": [
        {"technology": "sw", "evidence_ids": [f"e{index}"]} for index in range(1, 4)
    ], "claims": []}}
    markdown = "# SUMMARY\n\n## 4.2 시장성\n\n" + "\n".join(
        f"- 시장 사실 {index}. 〔근거: e{index}〕" for index in range(1, 4)
    )
    result = bias_control(markdown, store, findings)
    assert not result["passed"]
    assert "4.2 시장성: 한 기술만 다룸" in result["details"]


def test_quality_node_unsupported_entailment_fails_groundedness_with_sentence():
    sentence = "- ITME 처리량은 63% 증가했다. 〔근거: domain:ev:bad〕"

    def fake_judge(_markdown, _statuses, _store):
        return {"model": "fake", "criteria": {
            "neutrality": {"score": 5, "passed": True, "reasons": "중립", "problem_sentences": []},
            "groundedness": {
                "score": 3,
                "passed": True,
                "reasons": "불일치",
                "problem_sentences": [sentence],
                "unsupported_count": 1,
                "unsupported_items": [{"sentence": sentence, "evidence_ids": ["domain:ev:bad"]}],
            },
        }}

    technical = {"status": "complete", "records": [{"technology": "sw", "evidence_ids": ["technical:ev:ok"]}], "claims": [], "gaps": []}
    market = {"status": "complete", "records": [{"technology": "hw", "evidence_ids": ["market:ev:ok"]}], "claims": [], "gaps": []}
    stakeholder = {"status": "complete", "records": [{"technology": "both", "evidence_ids": ["ev:stake"]}], "claims": [], "gaps": []}
    domain = {
        "status": "complete",
        "records": [{"technology": "hw", "evidence_ids": ["domain:ev:bad"]}],
        "claims": [],
        "gaps": [],
    }
    state = {
        "report_sections": {"final_markdown_with_ids": (
            "# SUMMARY\n\n"
            "## 4.1 기술 성숙도(TRL)\n\n- 기술 사실. 〔근거: technical:ev:ok〕\n\n"
            "## 4.2 시장성\n\n- 시장 사실. 〔근거: market:ev:ok〕\n\n"
            "## 4.3 이해관계자\n\n- 이해관계자 사실. 〔근거: ev:stake〕\n\n"
            f"## 4.4 도메인 적용\n\n{sentence}"
        )},
        "evidence_store": {
            "technical:ev:ok": {"quote": "기술 사실", "url": "https://technical.example.com"},
            "market:ev:ok": {"quote": "시장 사실", "url": "https://market.example.com"},
            "ev:stake": {"quote": "이해관계자 사실", "url": "https://stakeholder.example.com"},
            "domain:ev:bad": {"quote": "처리량은 10% 증가했다.", "url": "https://domain.example.com"},
        },
        "technical_findings": technical,
        "market_findings": market,
        "stakeholder_findings": stakeholder,
        "domain_findings": domain,
        "quality_by_perspective": {"report": {"violations": []}},
        "report_version": 1,
    }
    result = quality_agent(state, judge=fake_judge)["eval_result"]
    assert "groundedness" in result["failed_criteria"]
    assert f"groundedness: {sentence}" in result["feedback"]
    assert result["rework_targets"] == ["domain"]

    status = {
        name: {"status": "done", "attempts": 1, "completed_step": 1}
        for name in ("technical", "market", "stakeholder", "domain")
    }
    status.update({
        "synthesis": {"status": "done", "attempts": 1, "completed_step": 2},
        "report": {"status": "done", "attempts": 1, "completed_step": 3},
    })
    supervisor_state = {
        **state,
        "step_count": 4,
        "max_steps": 20,
        "node_status": status,
        "synthesis": {"status": "complete"},
        "eval_result": result,
        "rework": {},
    }
    update = decide(supervisor_state)
    assert update["next"] == ["domain"]
    assert "품질 평가 지적으로 관점 재작업" in update["last_decision"]["reason"]


def test_stale_report_dispatch_clears_old_rework_directive():
    complete = {"status": "complete", "records": [], "claims": [], "gaps": []}
    status = {
        name: {"status": "done", "attempts": 1, "completed_step": 1}
        for name in ("technical", "market", "stakeholder", "domain")
    }
    status.update({
        "synthesis": {"status": "done", "attempts": 2, "completed_step": 7},
        "report": {"status": "done", "attempts": 1, "completed_step": 5},
    })
    state = {
        "step_count": 8,
        "max_steps": 20,
        "node_status": status,
        "technical_findings": complete,
        "market_findings": complete,
        "stakeholder_findings": complete,
        "domain_findings": complete,
        "evidence_store": {},
        "synthesis": {"status": "complete"},
        "report_version": 1,
        "eval_result": {"passed": False, "evaluated_report_version": 1},
        "rework": {"report": {
            "reason": "이전 품질 평가 불합격",
            "focus": ["오래된 피드백"],
            "round": 1,
            "max_search_rounds": 2,
            "feedback": ["오래된 피드백"],
            "created_step": 5,
        }},
    }
    update = decide(state)
    assert update["next"] == ["report"]
    assert "stale" in update["last_decision"]["reason"]
    assert "report" not in update["rework"]


def test_perspective_rework_preserves_report_feedback_until_stale_report_runs():
    complete = {"status": "complete", "records": [], "claims": [], "gaps": []}
    status = {
        name: {"status": "done", "attempts": 1, "completed_step": 1}
        for name in ("market", "stakeholder", "domain")
    }
    status.update({
        "technical": {"status": "done", "attempts": 1, "completed_step": 2},
        "synthesis": {"status": "done", "attempts": 1, "completed_step": 4},
        "report": {"status": "done", "attempts": 1, "completed_step": 5},
        "quality_eval": {"status": "done", "attempts": 1, "completed_step": 6},
    })
    state = {
        "step_count": 6,
        "max_steps": 20,
        "node_status": status,
        "technical_findings": complete,
        "market_findings": complete,
        "stakeholder_findings": complete,
        "domain_findings": complete,
        "evidence_store": {},
        "synthesis": {"status": "complete"},
        "report_version": 1,
        "eval_result": {
            "passed": False,
            "evaluated_report_version": 1,
            "feedback": ["groundedness: 잘못 연결된 근거를 수정"],
            "rework_targets": ["technical"],
        },
        "rework": {},
    }

    technical_update = decide(state)
    assert technical_update["next"] == ["technical"]
    assert technical_update["rework"]["report"]["feedback"] == state["eval_result"]["feedback"]

    state.update(technical_update)
    state["node_status"] = {**status, **technical_update["node_status"]}
    state["node_status"]["technical"] = {
        **status["technical"], "attempts": 2, "completed_step": 7,
    }
    synthesis_update = decide(state)
    assert synthesis_update["next"] == ["synthesis"]

    state.update(synthesis_update)
    state["node_status"] = {**state["node_status"], **synthesis_update["node_status"]}
    state["node_status"]["synthesis"] = {
        **status["synthesis"], "attempts": 2, "completed_step": 8,
    }
    report_update = decide(state)
    assert report_update["next"] == ["report"]
    assert report_update["rework"]["report"]["feedback"] == state["eval_result"]["feedback"]
    assert report_update["rework"]["report"]["created_step"] == 7


def test_consumed_directive_is_removed_on_next_dispatch():
    complete = {"status": "complete", "records": [], "claims": [], "gaps": []}
    status = {
        name: {"status": "done", "attempts": 1, "completed_step": 1}
        for name in ("technical", "market", "stakeholder", "domain")
    }
    status.update({
        "synthesis": {"status": "done", "attempts": 2, "completed_step": 9},
        "report": {"status": "done", "attempts": 1, "completed_step": 7},
    })
    state = {
        "step_count": 10,
        "max_steps": 20,
        "node_status": status,
        "technical_findings": complete,
        "market_findings": complete,
        "stakeholder_findings": complete,
        "domain_findings": complete,
        "evidence_store": {},
        "synthesis": {"status": "complete"},
        "report_version": 1,
        "rework": {"report": {
            "reason": "이미 반영된 품질 피드백",
            "focus": ["수정"],
            "round": 1,
            "max_search_rounds": 2,
            "feedback": ["이미 반영됨"],
            "created_step": 7,
        }},
    }
    update = decide(state)
    assert update["next"] == ["report"]
    assert "report" not in update["rework"]

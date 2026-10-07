"""Supervisor 재작업 요청(rework_hint)을 워커가 실제로 사용하는지. 힌트가 없으면 동작이 달라지지 않아야 한다."""
from types import SimpleNamespace

from agents.domain.node import hint_gaps
from agents.market.node import project_input as market_input, rework_focus
from agents.market.subgraph import plan as market_plan
from agents.technical.node import _fixed_parent_input

HINT = {"extra_rounds": 1, "requested_by": "supervisor-sufficiency",
        "gaps": [{"technology": "hw", "criterion": "상용화·채택 현황", "reason": "직접 근거 부족"}],
        "focus_queries": ["ITME 공식 발표"]}


def parent(**extra):
    spec = lambda n: {"name": n, "selection_reason": "이유", "short_name": n, "technology": "sw", "approach": "", "source_ids": []}
    return {"request": {"as_of": "2026-09-22", "language": "ko", "scope": "datacenter_inference", "max_search_rounds": 1},
            "selected_tech": {"sw": spec("MLA"), "hw": spec("ITME")}, "domain": "datacenter_inference", **extra}


def test_domain_seeds_first_plan_with_hint():
    assert hint_gaps(parent()) == []
    gaps = hint_gaps(parent(rework_hint=HINT))
    assert any("상용화·채택 현황" in g for g in gaps) and any("ITME 공식 발표" in g for g in gaps)


def test_market_hint_reaches_query_planning_prompt():
    assert rework_focus(parent()) == "" and market_input(parent())["rework_focus"] == ""
    seen = {}

    class Llm:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            seen["human"] = messages[-1][1]
            raise RuntimeError("계획 실패 → 템플릿 대체")

    local = market_input(parent(rework_hint=HINT))
    market_plan(local, SimpleNamespace(llm=Llm()))
    assert "ITME 공식 발표" in seen["human"] and "상용화·채택 현황" in seen["human"]
    local = market_input(parent())
    market_plan(local, SimpleNamespace(llm=Llm()))
    assert "근거가 부족했던 부분" not in seen["human"]


def test_technical_adds_focus_queries_only_with_hint():
    from agents.technical.config import DEFAULT_SELECTED_TECH
    state = {"selected_tech": DEFAULT_SELECTED_TECH}
    request, _ = _fixed_parent_input(state)
    assert "focus_queries" not in request
    request, _ = _fixed_parent_input({**state, "rework_hint": HINT})
    assert request["focus_queries"][0] == "ITME 공식 발표" and len(request["focus_queries"]) <= 2
    assert request["max_search_rounds"] <= 2  # 고정 입력 검증은 그대로


def test_technical_runs_extra_web_queries_for_hint():
    from copy import deepcopy

    from agents.technical.config import DEFAULT_REQUEST, DEFAULT_SELECTED_TECH
    from agents.technical.node import make_node
    from tests.agents.technical.test_technical_agent import _deps

    base = {"request": deepcopy(DEFAULT_REQUEST), "selected_tech": deepcopy(DEFAULT_SELECTED_TECH)}
    deps, plain_web = _deps()
    make_node(deps, corpus_manifest=[])(base)
    deps, hinted_web = _deps()
    make_node(deps, corpus_manifest=[])({**base, "rework_hint": HINT})
    extra = [c for c in hinted_web.calls if "ITME 공식 발표" in c[1]]
    assert len(extra) == 2 and len(hinted_web.calls) == len(plain_web.calls) + 4  # 보완 질의 2개 × 기술 2개

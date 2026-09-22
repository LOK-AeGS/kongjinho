"""시장 평가 에이전트 노드: 부모 State(graph/state.py AppState) ↔ 시장 내부 State 변환.

부모 그래프가 아는 것은 make_node 하나뿐이다. 내부 검색·판정 반복은 subgraph.py 에 있다.
"""

from __future__ import annotations

from agents.market.state import to_perspective_findings
from agents.market.subgraph import MarketAgentDeps, _build_subgraph


def summarize_technical(findings: dict | None) -> str:
    """선행 기술 조사 결과(AppState.technical_findings, PerspectiveFindings)를 압축한다.

    없으면 자체 근거만으로 진행한다는 뜻으로 빈 문자열을 돌려준다.
    """
    if not findings:
        return ""
    lines = [f"- ({c.get('technology', '?')}) {c.get('text', '')}" for c in findings.get("claims", [])[:12]]
    return "\n".join(lines)


def project_input(state: dict) -> dict:
    """부모 State에서 이 관점이 볼 것만 추린다.

    다른 관점(이해관계자·도메인)의 중간 결론은 넘기지 않는다. 넘기면 시장 판단이
    그쪽 결론에 물든다(앵커링). 선행 단계인 기술 조사 결과와 요청 정보만 통과시킨다.
    """
    sw, hw = state["selected_tech"]["sw"], state["selected_tech"]["hw"]
    req = state["request"]
    return {
        "technologies": {"sw": sw["name"], "hw": hw["name"]},
        "tech_desc": {"sw": sw["selection_reason"], "hw": hw["selection_reason"]},
        "domain": state["domain"],
        "as_of_date": req["as_of"],
        "max_search_rounds": req["max_search_rounds"],
        "technical_summary": summarize_technical(state.get("technical_findings")),
    }


def make_node(deps: MarketAgentDeps):
    """graph.build.build_graph(market=...) 에 주입할 노드 함수를 만든다.

    AppState 하나만 대상으로 한다(ISSUE.md 1-1 해결 이후 legacy 분기는 없앴다).
    반환 키: market_findings(PerspectiveFindings), evidence_store(Evidence, id 키),
    search_log_by_perspective(관점별 검색 로그 — Pool B 출처 등급 필터·페이지 예산 로그 등).
    """
    subgraph = _build_subgraph(deps)

    def market_node(state: dict) -> dict:
        local = {**project_input(state), "query_bank": {}, "pending": [], "raw": [], "seen": [],
                 "quote_keys": [], "evidence": [], "claims": [], "verdicts": [],
                 "search_rounds_used": 0, "errors": [], "next_action": "search",
                 "pages_used": 0, "search_log": []}
        result = subgraph.invoke(local)
        out = to_perspective_findings(result, result["completion"])
        out["search_log_by_perspective"] = {"market": result.get("search_log", [])}
        return out

    return market_node

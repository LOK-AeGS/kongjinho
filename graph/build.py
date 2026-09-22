"""부모 그래프 연결. 각 에이전트의 make_node() 결과를 주입받아 Edge만 정의한다.

부모 State 는 graph/state.py 의 AppState(팀 공통 State) 하나다.
각 노드는 AppState 의 자기 소유 키만 반환해야 한다. 선언되지 않은 키는 LangGraph가 조용히 버린다.
흐름(설계서 §8.1): technical → market·stakeholder·domain 병렬 → synthesis → report.
실행 진입점은 루트 main.py, 아직 없는 노드의 임시 노드는 graph/stubs.py 에 있다.
"""
from langgraph.graph import END, START, StateGraph

from graph.state import AppState


def build_graph(*, technical, market, stakeholder, domain, synthesis, report, checkpointer=None):
    # 각 함수는 해당 에이전트의 최종 결과 키 하나만 반환한다.
    # 내부 검색/수정 반복은 함수 안의 서브그래프가 마친 후 부모에 반환한다.
    graph = StateGraph(AppState)
    for name, node in {
        "technical": technical,
        "market": market,
        "stakeholder": stakeholder,
        "domain": domain,
        "synthesis": synthesis,
        "report": report,
    }.items():
        graph.add_node(name, node)
    graph.add_edge(START, "technical")
    for name in ("market", "stakeholder", "domain"):
        graph.add_edge("technical", name)
    # 세 분기가 모두 끝나야 종합을 실행한다.
    graph.add_edge(["market", "stakeholder", "domain"], "synthesis")
    graph.add_edge("synthesis", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)

"""에이전트 구현 함수를 주입하는 그래프 연결 예시. langgraph 설치 필요."""
from langgraph.graph import END, START, StateGraph

from state import PipelineState


def build_graph(*, technical, market, stakeholder, domain, synthesis, report, checkpointer=None):
    # 각 함수는 해당 에이전트의 최종 결과 키 하나만 반환한다.
    # 내부 검색/수정 반복은 함수 안의 서브그래프가 마친 후 부모에 반환한다.
    graph = StateGraph(PipelineState)
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

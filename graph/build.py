"""부모 그래프 연결 (Supervisor 패턴). 각 에이전트의 make_node() 결과를 주입받아 Edge만 정의한다.

부모 State 는 graph/state.py 의 AppState(팀 공통 State) 하나다.
각 노드는 AppState 의 자기 소유 키만 반환해야 한다. 선언되지 않은 키는 LangGraph가 조용히 버린다.

흐름: START → supervisor → (조건부 edge) → 워커 → supervisor → … → END
  - 모든 워커·종합·보고서·품질 평가 노드의 유일한 출구는 supervisor 다 (에이전트 간 직접 통신 금지).
  - 어느 노드로 갈지는 supervisor 가 State 에서 계산한다 (graph/supervisor.py). 순서가 코드에 박혀 있지 않다.
  - 근거 충분성 평가 뒤에야 보고서를 쓰고, 보고서 뒤 품질 평가가 미달이면 supervisor 가 재작업을 요청해 Loop 를 돈다.
실행 진입점은 루트 main.py, 아직 없는 노드의 임시 노드는 graph/stubs.py 에 있다.
"""
from langgraph.graph import END, START, StateGraph

from graph.quality import make_quality_node
from graph.state import AppState
from graph.sufficiency import PERSPECTIVES
from graph.supervisor import make_supervisor, route
from graph.workers import as_worker


def build_graph(*, technical, market, stakeholder, domain, synthesis, report, quality=None, supervisor=None, checkpointer=None):
    graph = StateGraph(AppState)
    graph.add_node("supervisor", supervisor or make_supervisor())
    for name, node in {
        "technical": technical,
        "market": market,
        "stakeholder": stakeholder,
        "domain": domain,
        "synthesis": synthesis,
        "report": report,
    }.items():
        graph.add_node(name, as_worker(name, node))
        graph.add_edge(name, "supervisor")
    graph.add_node("quality", quality or make_quality_node())
    graph.add_edge("quality", "supervisor")
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route, [*PERSPECTIVES, "synthesis", "report", "quality", END])
    return graph.compile(checkpointer=checkpointer)

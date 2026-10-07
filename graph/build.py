"""Supervisor 중심 부모 그래프 토폴로지."""

from langgraph.graph import END, START, StateGraph

from graph.state import SupervisorState
from graph.supervisor import SupervisorPolicy, make_supervisor, route


WORKERS = ("technical", "market", "stakeholder", "domain", "synthesis", "report", "quality_eval")


def build_graph(
    *,
    technical,
    market,
    stakeholder,
    domain,
    synthesis,
    report,
    quality_eval,
    checkpointer=None,
    policy: SupervisorPolicy | None = None,
    proposer=None,
    decision_logger=None,
    on_decision=None,
):
    graph = StateGraph(SupervisorState)
    graph.add_node(
        "supervisor",
        make_supervisor(
            policy=policy,
            proposer=proposer,
            logger=decision_logger,
            on_decision=on_decision,
        ),
    )
    for name, node in {
        "technical": technical,
        "market": market,
        "stakeholder": stakeholder,
        "domain": domain,
        "synthesis": synthesis,
        "report": report,
        "quality_eval": quality_eval,
    }.items():
        graph.add_node(name, node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route,
        {**{name: name for name in WORKERS}, "__end__": END},
    )
    for name in WORKERS:
        graph.add_edge(name, "supervisor")
    return graph.compile(checkpointer=checkpointer)

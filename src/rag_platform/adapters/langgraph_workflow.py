"""The query workflow executed as a LangGraph state graph (specification sections 5 and 9).

The states and the edges between them are defined once, in `rag_platform.workflow`. This adapter
changes only who drives them: instead of the in-process loop, LangGraph owns the control flow,
with one graph node per workflow state and a conditional edge for every declared transition, so
clarification, repair and fallback are real branches of the graph rather than a wrapped chain.
Because the node functions are the same objects, an answer and its trace are identical under
either executor, which is what the equivalence tests assert.

LangGraph's recursion limit replaces the loop's step bound. Hitting it routes to the same
fallback an exhausted loop would, so a cycling repair path can never hang a query.
"""

from typing import Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from rag_platform.models import AccessContext, QueryResponse
from rag_platform.workflow import (
    ENTRY,
    MAX_STEPS,
    TERMINALS,
    TRANSITIONS,
    QueryObservation,
    QueryWorkflow,
    WorkflowState,
)


class GraphState(TypedDict):
    """What LangGraph carries between nodes: the workflow state and where it goes next."""

    workflow: WorkflowState
    observation: QueryObservation
    next: str


class LangGraphQueryWorkflow(QueryWorkflow):
    """`QueryWorkflow` with LangGraph as the executor."""

    _graph: Any = None

    def compiled(self) -> Any:
        """The compiled graph. Built once per workflow; it holds no per-query state."""
        if self._graph is None:
            self._graph = self._build().compile()
        return self._graph

    def mermaid(self) -> str:
        """The graph as a Mermaid diagram, for documentation and review."""
        return str(self.compiled().get_graph().draw_mermaid())

    def _build(self) -> Any:
        graph = StateGraph(GraphState)
        for name in TRANSITIONS:
            graph.add_node(name, self._node(name))
        graph.add_edge(START, ENTRY)
        for name, targets in TRANSITIONS.items():
            routes: dict[Any, Any] = {
                target: END if target in TERMINALS else target for target in targets
            }
            graph.add_conditional_edges(name, _route, routes)
        return graph

    def _node(self, name: str) -> Any:
        def run(graph_state: GraphState) -> dict[str, str]:
            following = self.step(name, graph_state["workflow"], graph_state["observation"])
            return {"next": following}

        return run

    def _execute(
        self, question: str, access: AccessContext, observation: QueryObservation
    ) -> QueryResponse:
        state = self.new_state(question, access)
        initial: GraphState = {"workflow": state, "observation": observation, "next": ""}
        try:
            final = self.compiled().invoke(initial, config={"recursion_limit": MAX_STEPS})
        except GraphRecursionError:
            return self.respond("fallback", state)
        return self.respond(str(final["next"]), state)


def _route(graph_state: GraphState) -> str:
    return graph_state["next"]

"""The LangGraph executor must be indistinguishable from the in-process loop.

Both drive the same state functions, so for every path through the state machine the answer and
the trace have to match. Anything that differs is a bug in the adapter.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from rag_platform.context import ContextBuilder, EvidenceItem
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig, QueryResponse
from rag_platform.programs import ProgramSuite, SynthesizedAnswer
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.workflow import (
    ENTRY,
    TERMINALS,
    TRANSITIONS,
    QueryWorkflow,
    WorkflowTransitionError,
)

pytest.importorskip("langgraph")

import rag_platform.adapters.langgraph_workflow as adapter
from rag_platform.adapters.langgraph_workflow import LangGraphQueryWorkflow

ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))
CORPUS = [
    DocumentChunk(
        chunk_id="leave",
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text="Annual leave requests use the HR portal. Managers approve within two working days.",
        source_uri="https://example.test/leave",
        source_title="Leave policy",
        index_version="index-v1",
        source_version_id="sv-leave",
    ),
    DocumentChunk(
        chunk_id="expenses",
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text="Expense claims are approved by the cost centre owner.",
        source_uri="https://example.test/expenses",
        source_title="Expenses policy",
        index_version="index-v1",
        source_version_id="sv-expenses",
    ),
]


@dataclass(slots=True)
class InventingSynthesizer:
    """Invents a claim for the first `inventions` calls, then quotes its evidence."""

    inventions: int = 1
    revision: str = "synthesizer-inventing"
    calls: int = 0

    def synthesize(
        self, question: str, evidence: Sequence[EvidenceItem], *, sentence_limit: int
    ) -> SynthesizedAnswer:
        self.calls += 1
        if self.calls <= self.inventions:
            return SynthesizedAnswer(text="Annual leave is unlimited.", citation_ids=("C1",))
        return SynthesizedAnswer(
            text=evidence[0].chunk.text.split(".")[0] + ".",
            citation_ids=(evidence[0].citation_id,),
        )


def build(
    workflow_class: type[QueryWorkflow],
    *,
    config: PipelineConfig | None = None,
    programs: ProgramSuite | None = None,
) -> QueryWorkflow:
    resolved = config or PipelineConfig()
    return workflow_class(
        HybridRetriever(InMemoryChunkRepository(CORPUS), resolved),
        resolved,
        ContextBuilder(resolved),
        programs,
    )


def comparable(response: QueryResponse) -> dict[str, Any]:
    """Everything except the wall-clock measurement."""
    body = response.model_dump()
    body["trace"].pop("latency_ms")
    return body


SCENARIOS: dict[str, tuple[str, int]] = {
    "answered": ("How do I request annual leave?", 0),
    "multi_part": ("How do I request annual leave and who approves expense claims?", 0),
    "clarification": ("What about it?", 0),
    "no_evidence": ("What is the parking permit fee?", 0),
    "repaired": ("How do I request annual leave?", 1),
    "repair_exhausted": ("How do I request annual leave?", 5),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_both_executors_produce_the_same_answer_and_trace(scenario: str) -> None:
    question, inventions = SCENARIOS[scenario]

    def programs() -> ProgramSuite | None:
        if not inventions:
            return None
        return ProgramSuite(synthesizer=InventingSynthesizer(inventions=inventions))

    loop = build(QueryWorkflow, programs=programs()).run(question, ACCESS)
    graph = build(LangGraphQueryWorkflow, programs=programs()).run(question, ACCESS)

    assert comparable(graph) == comparable(loop)


def test_the_scenarios_cover_every_terminal_and_the_repair_branch() -> None:
    outcomes = {}
    for name, (question, inventions) in SCENARIOS.items():
        programs = (
            ProgramSuite(synthesizer=InventingSynthesizer(inventions=inventions))
            if inventions
            else None
        )
        outcomes[name] = build(LangGraphQueryWorkflow, programs=programs).run(question, ACCESS)

    assert {response.status for response in outcomes.values()} == {
        "answered",
        "clarification_needed",
        "insufficient_evidence",
    }
    assert outcomes["repaired"].trace.repair_attempts == 1
    assert "repair" in outcomes["repaired"].trace.workflow_path
    assert outcomes["repair_exhausted"].status == "insufficient_evidence"


def test_the_graph_has_one_node_per_state_and_only_declared_edges() -> None:
    workflow = build(LangGraphQueryWorkflow)
    assert isinstance(workflow, LangGraphQueryWorkflow)
    drawn = workflow.compiled().get_graph()

    assert set(drawn.nodes) - {"__start__", "__end__"} == set(TRANSITIONS)
    edges = {(edge.source, edge.target) for edge in drawn.edges}
    expected = {("__start__", ENTRY)} | {
        (source, "__end__" if target in TERMINALS else target)
        for source, targets in TRANSITIONS.items()
        for target in targets
    }
    assert edges == expected


def test_clarification_repair_and_fallback_are_branches_of_the_graph() -> None:
    workflow = build(LangGraphQueryWorkflow)
    assert isinstance(workflow, LangGraphQueryWorkflow)
    diagram = workflow.mermaid()

    assert "classify -.-> clarify" in diagram
    assert "verify -.-> repair" in diagram
    assert "repair -.-> generate" in diagram
    assert "fallback" in diagram


def test_the_graph_is_compiled_once_and_reused_across_queries() -> None:
    workflow = build(LangGraphQueryWorkflow)
    assert isinstance(workflow, LangGraphQueryWorkflow)
    first = workflow.compiled()
    workflow.run("How do I request annual leave?", ACCESS)
    workflow.run("What about it?", ACCESS)
    assert workflow.compiled() is first


def test_queries_do_not_share_state() -> None:
    workflow = build(LangGraphQueryWorkflow)
    answered = workflow.run("How do I request annual leave?", ACCESS)
    clarification = workflow.run("What about it?", ACCESS)
    again = workflow.run("How do I request annual leave?", ACCESS)

    assert clarification.trace.workflow_path == ["classify", "clarify"]
    assert comparable(again) == comparable(answered)


def test_hitting_the_recursion_limit_falls_back_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "MAX_STEPS", 3)
    response = build(LangGraphQueryWorkflow).run("How do I request annual leave?", ACCESS)

    assert response.status == "insufficient_evidence"
    assert response.citations == []


def test_a_state_cannot_route_outside_the_declared_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for workflow_class in (QueryWorkflow, LangGraphQueryWorkflow):
        workflow = build(workflow_class)
        monkeypatch.setattr(workflow, "_rewrite", lambda state: "answer")
        with pytest.raises(WorkflowTransitionError, match="rewrite cannot route to answer"):
            workflow.run("How do I request annual leave?", ACCESS)


def test_the_in_process_loop_also_falls_back_when_it_runs_out_of_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_platform.workflow as core

    monkeypatch.setattr(core, "MAX_STEPS", 3)
    response = build(QueryWorkflow).run("How do I request annual leave?", ACCESS)
    assert response.status == "insufficient_evidence"


def test_every_transition_target_is_a_state_or_a_terminal() -> None:
    for targets in TRANSITIONS.values():
        for target in targets:
            assert target in TRANSITIONS or target in TERMINALS
    assert ENTRY in TRANSITIONS

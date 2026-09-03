from collections.abc import Sequence
from dataclasses import dataclass

from rag_platform.context import ContextBuilder, EvidenceItem
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.programs import ProgramSuite, SynthesizedAnswer
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.workflow import QueryWorkflow

ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))


def chunk(identifier: str, text: str, source_version_id: str = "sv-1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text=text,
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="index-v1",
        source_version_id=source_version_id,
    )


CORPUS = [
    chunk(
        "leave",
        "Annual leave requests use the HR portal. Managers approve within two working days.",
        "sv-leave",
    ),
    chunk("expenses", "Expense claims are approved by the cost centre owner.", "sv-expenses"),
]


@dataclass(slots=True)
class InventingSynthesizer:
    """Stands in for a generative synthesizer that drifts away from its evidence."""

    revision: str = "synthesizer-inventing"
    calls: int = 0

    def synthesize(
        self, question: str, evidence: Sequence[EvidenceItem], *, sentence_limit: int
    ) -> SynthesizedAnswer:
        self.calls += 1
        if self.calls == 1:
            return SynthesizedAnswer(
                text="Annual leave is unlimited.", citation_ids=("C1",)
            )
        return SynthesizedAnswer(
            text=evidence[0].chunk.text.split(".")[0] + ".",
            citation_ids=(evidence[0].citation_id,),
        )


def build_workflow(
    config: PipelineConfig | None = None,
    programs: ProgramSuite | None = None,
    corpus: list[DocumentChunk] | None = None,
) -> QueryWorkflow:
    resolved = config or PipelineConfig()
    repository = InMemoryChunkRepository(corpus if corpus is not None else CORPUS)
    return QueryWorkflow(
        HybridRetriever(repository, resolved),
        resolved,
        ContextBuilder(resolved),
        programs,
    )


def test_the_happy_path_visits_every_expected_state() -> None:
    response = build_workflow().run("How do I request annual leave?", ACCESS)

    assert response.status == "answered"
    assert response.trace.workflow_path == [
        "classify",
        "rewrite",
        "decompose",
        "retrieve",
        "build_context",
        "generate",
        "verify",
    ]
    assert response.trace.intent == "procedural"
    assert response.trace.repair_attempts == 0
    assert "programs-v1" in response.trace.program_revisions


def test_every_cited_source_appears_in_the_response() -> None:
    response = build_workflow().run("How do I request annual leave?", ACCESS)
    assert {citation.citation_id for citation in response.citations} == {"C1"}
    assert response.citations[0].chunk_id == "leave"
    assert response.answer in CORPUS[0].text


def test_an_ambiguous_question_asks_for_clarification_instead_of_guessing() -> None:
    response = build_workflow().run("leave", ACCESS)

    assert response.status == "clarification_needed"
    assert response.citations == []
    assert response.trace.workflow_path == ["classify", "clarify"]
    assert response.trace.intent == "ambiguous"
    assert "more detail" in response.answer


def test_clarification_can_be_turned_off() -> None:
    config = PipelineConfig(clarify_ambiguous_queries=False)
    response = build_workflow(config).run("leave", ACCESS)
    assert response.status == "answered"
    assert "clarify" not in response.trace.workflow_path


def test_a_synonym_is_rewritten_into_the_indexed_vocabulary() -> None:
    response = build_workflow().run("How do I book vacation?", ACCESS)
    assert response.status == "answered"
    assert any(item.startswith("rewrite:") for item in response.trace.query_transformations)


def test_a_multi_part_question_is_decomposed_and_answered_from_both_sources() -> None:
    config = PipelineConfig(answer_sentence_limit=2, max_chunks_per_source=1)
    response = build_workflow(config).run(
        "How do I request annual leave and who signs off expense claims?", ACCESS
    )

    assert any(
        item.startswith("decompose:") for item in response.trace.query_transformations
    )
    assert set(response.trace.context_chunk_ids) == {"leave", "expenses"}
    assert {citation.chunk_id for citation in response.citations} == {"leave", "expenses"}


def test_an_unsupported_answer_is_repaired_before_it_is_returned() -> None:
    programs = ProgramSuite(synthesizer=InventingSynthesizer())
    response = build_workflow(programs=programs).run("annual leave", ACCESS)

    assert response.status == "answered"
    assert response.trace.repair_attempts == 1
    assert response.trace.workflow_path.count("generate") == 2
    assert "repair:attempt_1" in response.trace.query_transformations


def test_an_answer_that_cannot_be_repaired_falls_back_to_abstention() -> None:
    @dataclass(slots=True)
    class AlwaysInventing:
        revision: str = "synthesizer-always-inventing"

        def synthesize(
            self, question: str, evidence: Sequence[EvidenceItem], *, sentence_limit: int
        ) -> SynthesizedAnswer:
            return SynthesizedAnswer(text="Leave is unlimited.", citation_ids=("C1",))

    programs = ProgramSuite(synthesizer=AlwaysInventing())
    response = build_workflow(programs=programs).run("annual leave", ACCESS)

    assert response.status == "insufficient_evidence"
    assert response.trace.unsupported_claims == ["Leave is unlimited."]
    assert response.trace.repair_attempts == 1


def test_repair_can_be_disabled() -> None:
    programs = ProgramSuite(synthesizer=InventingSynthesizer())
    config = PipelineConfig(max_repair_attempts=0)
    response = build_workflow(config, programs).run("annual leave", ACCESS)

    assert response.status == "insufficient_evidence"
    assert response.trace.repair_attempts == 0


def test_a_question_without_authorized_evidence_falls_back() -> None:
    response = build_workflow().run("When is the next lunar eclipse?", ACCESS)

    assert response.status == "insufficient_evidence"
    assert response.trace.workflow_path[-1] == "build_context"
    assert response.trace.context_chunk_ids == []


def test_an_empty_synthesis_never_becomes_an_uncited_answer() -> None:
    @dataclass(slots=True)
    class SilentSynthesizer:
        revision: str = "synthesizer-silent"

        def synthesize(
            self, question: str, evidence: Sequence[EvidenceItem], *, sentence_limit: int
        ) -> SynthesizedAnswer:
            return SynthesizedAnswer(text="", citation_ids=())

    programs = ProgramSuite(synthesizer=SilentSynthesizer())
    response = build_workflow(programs=programs).run("annual leave", ACCESS)

    assert response.status == "insufficient_evidence"
    assert response.citations == []
    # The evidence was found; only the generation step failed, and the trace says so.
    assert response.trace.context_chunk_ids == ["leave"]

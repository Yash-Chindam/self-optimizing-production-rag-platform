from dataclasses import dataclass, field
from typing import Literal

from rag_platform.context import ContextBuilder
from rag_platform.evaluation import Evaluator
from rag_platform.models import (
    AccessContext,
    AnswerTrace,
    Citation,
    DocumentChunk,
    EvaluationCase,
    PipelineConfig,
    QueryResponse,
)
from rag_platform.programs import ProgramSuite
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService

TENANT = "tenant-a"


def chunk(identifier: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id=TENANT,
        access_labels=frozenset({"public"}),
        text=text,
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="index-v1",
        source_version_id=identifier,
    )


CORPUS = [
    chunk("leave", "Annual leave requests use the HR portal. Managers approve within two days."),
    chunk("expenses", "Expense claims are approved by the cost centre owner."),
]


def build_service(config: PipelineConfig | None = None) -> QueryService:
    resolved = config or PipelineConfig()
    repository = InMemoryChunkRepository(CORPUS)
    return QueryService(
        HybridRetriever(repository, resolved), resolved, ContextBuilder(resolved)
    )


def case(
    *,
    question: str = "How do I request annual leave?",
    required_evidence_chunk_ids: frozenset[str] = frozenset(),
    forbidden_chunk_ids: frozenset[str] = frozenset(),
    acceptable_answer_terms: frozenset[str] = frozenset(),
    forbidden_claims: tuple[str, ...] = (),
    expected_status: Literal[
        "answered", "insufficient_evidence", "clarification_needed"
    ] = "answered",
) -> EvaluationCase:
    return EvaluationCase(
        case_id="case-1",
        question=question,
        tenant_id=TENANT,
        access_labels=frozenset({"public"}),
        required_evidence_chunk_ids=required_evidence_chunk_ids,
        forbidden_chunk_ids=forbidden_chunk_ids,
        acceptable_answer_terms=acceptable_answer_terms,
        forbidden_claims=forbidden_claims,
        expected_status=expected_status,
        reviewer="reviewer-1",
    )


def test_a_correctly_grounded_answer_passes() -> None:
    report = Evaluator(build_service(), dataset_revision="eval-v1").evaluate(
        [
            case(
                required_evidence_chunk_ids=frozenset({"leave"}),
                acceptable_answer_terms=frozenset({"portal"}),
            )
        ]
    )
    assert report.dataset_revision == "eval-v1"
    assert report.summary.case_count == 1
    assert report.summary.passed_count == 1
    assert report.summary.mean_recall_at_k == 1.0
    assert report.summary.mean_reciprocal_rank == 1.0
    assert report.summary.grounded_rate == 1.0
    assert report.summary.pass_rate == 1.0
    assert report.failures() == ()
    assert report.results[0].failure_category is None


def test_missing_required_evidence_is_tagged_retrieval() -> None:
    report = Evaluator(build_service(), dataset_revision="eval-v1").evaluate(
        [case(question="annual leave", required_evidence_chunk_ids=frozenset({"nonexistent"}))]
    )
    result = report.results[0]
    assert result.passed is False
    assert result.failure_category == "retrieval"
    assert result.recall_at_k == 0.0


def test_an_unexpected_abstention_with_missing_evidence_is_tagged_retrieval() -> None:
    report = Evaluator(build_service(), dataset_revision="eval-v1").evaluate(
        [
            case(
                question="When is the next lunar eclipse?",
                required_evidence_chunk_ids=frozenset({"leave"}),
                expected_status="answered",
            )
        ]
    )
    result = report.results[0]
    assert result.status == "insufficient_evidence"
    assert result.passed is False
    assert result.failure_category == "retrieval"


def test_an_unexpected_abstention_is_tagged_generation() -> None:
    report = Evaluator(build_service(), dataset_revision="eval-v1").evaluate(
        [case(question="When is the next lunar eclipse?", expected_status="answered")]
    )
    result = report.results[0]
    assert result.status == "insufficient_evidence"
    assert result.passed is False
    assert result.failure_category == "generation"


def test_a_correctly_predicted_abstention_passes() -> None:
    report = Evaluator(build_service(), dataset_revision="eval-v1").evaluate(
        [
            case(
                question="When is the next lunar eclipse?",
                expected_status="insufficient_evidence",
            )
        ]
    )
    assert report.results[0].passed is True


def test_a_missing_acceptable_term_is_tagged_generation() -> None:
    report = Evaluator(build_service(), dataset_revision="eval-v1").evaluate(
        [
            case(
                required_evidence_chunk_ids=frozenset({"leave"}),
                acceptable_answer_terms=frozenset({"quantum-teleportation"}),
            )
        ]
    )
    result = report.results[0]
    assert result.passed is False
    assert result.failure_category == "generation"
    assert "quantum-teleportation" in result.notes


def test_a_forbidden_claim_is_tagged_citation() -> None:
    report = Evaluator(build_service(), dataset_revision="eval-v1").evaluate(
        [
            case(
                required_evidence_chunk_ids=frozenset({"leave"}),
                forbidden_claims=("HR portal",),
            )
        ]
    )
    result = report.results[0]
    assert result.passed is False
    assert result.failure_category == "citation"
    assert "HR portal" in result.notes


def test_an_unauthorized_chunk_in_context_is_tagged_authorization() -> None:
    """A stand-in service simulates a regression a real retriever should never allow."""

    @dataclass(slots=True)
    class LeakyService:
        config: PipelineConfig = field(default_factory=PipelineConfig)
        programs: ProgramSuite = field(default_factory=ProgramSuite)

        def answer(self, question: str, access: AccessContext) -> QueryResponse:
            return QueryResponse(
                status="answered",
                answer="Confidential forecast details.",
                citations=[
                    Citation(
                        citation_id="C1",
                        source_title="Finance forecast",
                        source_uri="https://example.test/forecast",
                        chunk_id="forecast",
                    )
                ],
                trace=AnswerTrace(
                    config_version="pipeline-v1",
                    index_version="index-v1",
                    retrieval_strategy="hybrid_reciprocal_rank",
                    retrieved_chunk_ids=["forecast"],
                    policy="tenant_and_access_labels_required_before_scoring_and_before_context",
                    context_chunk_ids=["forecast"],
                    unsupported_claims=[],
                ),
            )

    report = Evaluator(LeakyService(), dataset_revision="eval-v1").evaluate(
        [case(forbidden_chunk_ids=frozenset({"forecast"}))]
    )
    result = report.results[0]
    assert result.passed is False
    assert result.failure_category == "authorization"
    assert result.authorization_violations == ("forecast",)


def test_reciprocal_rank_credits_earlier_ranks_more() -> None:
    config = PipelineConfig(dense_top_k=10, sparse_top_k=10, final_top_k=10)
    report = Evaluator(build_service(config), dataset_revision="eval-v1").evaluate(
        [
            case(
                question="How do I request annual leave and who approves expense claims?",
                required_evidence_chunk_ids=frozenset({"expenses"}),
            )
        ]
    )
    result = report.results[0]
    assert 0.0 < result.reciprocal_rank <= 1.0


def test_an_empty_case_set_produces_a_zeroed_summary() -> None:
    report = Evaluator(build_service(), dataset_revision="eval-v1").evaluate([])
    assert report.summary.case_count == 0
    assert report.summary.pass_rate == 0.0
    assert report.results == ()

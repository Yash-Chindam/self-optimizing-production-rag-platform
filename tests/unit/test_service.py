from rag_platform.context import ContextBuilder
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService


def build_service(
    repository: InMemoryChunkRepository, config: PipelineConfig
) -> QueryService:
    return QueryService(HybridRetriever(repository, config), config, ContextBuilder(config))


def test_answer_includes_citation_and_reproducibility_trace(
    repository: InMemoryChunkRepository,
    employee_access: AccessContext,
    config: PipelineConfig,
) -> None:
    response = build_service(repository, config).answer(
        "How do I request vacation?", employee_access
    )
    assert response.status == "answered"
    assert response.citations[0].chunk_id == "public-leave"
    assert response.citations[0].citation_id == "C1"
    assert response.trace.config_version == config.version
    assert response.trace.index_version == "index-v1"
    assert response.trace.retrieved_chunk_ids == ["public-leave"]
    assert response.trace.context_chunk_ids == ["public-leave"]
    assert response.trace.context_characters == len(response.answer)
    assert response.trace.retrieval_strategy == "hybrid_reciprocal_rank"


def test_answer_abstains_when_authorized_evidence_is_missing(
    repository: InMemoryChunkRepository,
    employee_access: AccessContext,
    config: PipelineConfig,
) -> None:
    response = build_service(repository, config).answer(
        "What is the revenue growth target?", employee_access
    )
    assert response.status == "insufficient_evidence"
    assert response.citations == []
    assert response.trace.index_version is None
    assert response.trace.context_chunk_ids == []


def test_context_budget_can_force_abstention(employee_access: AccessContext) -> None:
    long_chunk = DocumentChunk(
        chunk_id="long-leave",
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text="Annual leave requests use the HR portal. " * 20,
        source_uri="https://example.test/leave",
        source_title="Leave policy",
        index_version="index-v1",
    )
    config = PipelineConfig(context_character_budget=200)
    service = build_service(InMemoryChunkRepository([long_chunk]), config)

    response = service.answer("annual leave", employee_access)
    assert response.status == "insufficient_evidence"
    assert response.trace.retrieved_chunk_ids == ["long-leave"]
    assert response.trace.dropped_chunk_ids == ["long-leave"]
    assert "context_budget_removed:long-leave" in response.trace.policy_notes

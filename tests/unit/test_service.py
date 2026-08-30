from rag_platform.models import AccessContext, PipelineConfig
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService


def test_answer_includes_citation_and_reproducibility_trace(
    repository: InMemoryChunkRepository,
    employee_access: AccessContext,
    config: PipelineConfig,
) -> None:
    service = QueryService(HybridRetriever(repository, config), config)
    response = service.answer("How do I request vacation?", employee_access)
    assert response.status == "answered"
    assert response.citations[0].chunk_id == "public-leave"
    assert response.trace.config_version == config.version
    assert response.trace.index_version == "index-v1"
    assert response.trace.retrieved_chunk_ids == ["public-leave"]


def test_answer_abstains_when_authorized_evidence_is_missing(
    repository: InMemoryChunkRepository,
    employee_access: AccessContext,
    config: PipelineConfig,
) -> None:
    service = QueryService(HybridRetriever(repository, config), config)
    response = service.answer("What is the revenue growth target?", employee_access)
    assert response.status == "insufficient_evidence"
    assert response.citations == []
    assert response.trace.index_version is None


def test_context_budget_can_force_abstention(
    repository: InMemoryChunkRepository, employee_access: AccessContext
) -> None:
    config = PipelineConfig(context_character_budget=200)
    service = QueryService(HybridRetriever(repository, config), config)
    service.config = config.model_copy(update={"context_character_budget": 1})
    response = service.answer("annual leave", employee_access)
    assert response.status == "insufficient_evidence"


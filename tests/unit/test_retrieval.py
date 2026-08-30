from rag_platform.models import AccessContext, PipelineConfig
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever


def test_hybrid_retrieval_fuses_dense_and_sparse_results(
    repository: InMemoryChunkRepository,
    employee_access: AccessContext,
    config: PipelineConfig,
) -> None:
    results = HybridRetriever(repository, config).retrieve("annual leave", employee_access)
    assert [result.chunk.chunk_id for result in results] == ["public-leave"]
    assert results[0].fused_score == 2 / 61


def test_hybrid_retrieval_returns_no_cross_tenant_results(
    repository: InMemoryChunkRepository, config: PipelineConfig
) -> None:
    unknown_tenant = AccessContext(tenant_id="tenant-c")
    assert HybridRetriever(repository, config).retrieve("annual leave", unknown_tenant) == []


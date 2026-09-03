from collections.abc import Iterable

from rag_platform.graph import GraphRetriever, KnowledgeGraph
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.reliability import CircuitBreaker
from rag_platform.repository import InMemoryChunkRepository, ScoredChunk
from rag_platform.rerank import LexicalCrossEncoder
from rag_platform.retrieval import HybridRetriever


def test_hybrid_retrieval_fuses_dense_and_sparse_results(
    repository: InMemoryChunkRepository,
    employee_access: AccessContext,
    config: PipelineConfig,
) -> None:
    result = HybridRetriever(repository, config).retrieve("annual leave", employee_access)
    assert [item.chunk.chunk_id for item in result.chunks] == ["public-leave"]
    assert result.chunks[0].fused_score == 2 / 61
    assert result.strategy == "hybrid_reciprocal_rank"
    assert result.graph_expanded_chunk_ids == ()


def test_hybrid_retrieval_returns_no_cross_tenant_results(
    repository: InMemoryChunkRepository, config: PipelineConfig
) -> None:
    unknown_tenant = AccessContext(tenant_id="tenant-c")
    result = HybridRetriever(repository, config).retrieve("annual leave", unknown_tenant)
    assert result.chunks == ()


def test_weighted_fusion_normalizes_each_strategy(
    repository: InMemoryChunkRepository, employee_access: AccessContext
) -> None:
    config = PipelineConfig(fusion_method="weighted", dense_weight=0.7, sparse_weight=0.3)
    result = HybridRetriever(repository, config).retrieve("annual leave", employee_access)

    assert result.strategy == "hybrid_weighted"
    # The single authorized match tops both strategies, so it earns the full combined weight.
    assert result.chunks[0].fused_score == 1.0


def test_weighted_fusion_ignores_strategies_without_positive_scores(
    employee_access: AccessContext,
) -> None:
    empty = InMemoryChunkRepository([])
    config = PipelineConfig(fusion_method="weighted")
    assert HybridRetriever(empty, config).retrieve("annual leave", employee_access).chunks == ()


def graph_corpus() -> list[DocumentChunk]:
    def chunk(identifier: str, text: str, labels: frozenset[str]) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=identifier,
            tenant_id="tenant-a",
            access_labels=labels,
            text=text,
            source_uri=f"https://example.test/{identifier}",
            source_title=identifier,
            index_version="index-v1",
            source_version_id=f"sv-{identifier}",
        )

    public = frozenset({"public"})
    return [
        chunk("payroll", "The Payroll Service depends on Identity Platform.", public),
        chunk(
            "identity",
            "Identity Platform enforces hardware keys for every administrator sign-in.",
            public,
        ),
        chunk("catering", "Catering guidance for the Dublin Office.", public),
        chunk("secret", "Identity Platform incident notes.", frozenset({"security"})),
    ]


def build_graph_retriever(chunks: list[DocumentChunk]) -> GraphRetriever:
    graph = KnowledgeGraph()
    graph.index(chunks)
    return GraphRetriever(graph, chunks)


def test_graph_expansion_reaches_related_evidence_without_shared_vocabulary() -> None:
    chunks = graph_corpus()
    config = PipelineConfig(graph_expansion_depth=1, final_top_k=5)
    retriever = HybridRetriever(
        InMemoryChunkRepository(chunks), config, graph_retriever=build_graph_retriever(chunks)
    )
    result = retriever.retrieve("payroll service", AccessContext(tenant_id="tenant-a"))

    assert result.graph_expanded_chunk_ids == ("identity",)
    assert "identity" in {item.chunk.chunk_id for item in result.chunks}
    assert "catering" not in {item.chunk.chunk_id for item in result.chunks}
    assert result.strategy == "hybrid_reciprocal_rank+graph_depth_1"


def test_graph_expansion_never_crosses_an_access_boundary() -> None:
    chunks = graph_corpus()
    config = PipelineConfig(graph_expansion_depth=1, final_top_k=5)
    retriever = HybridRetriever(
        InMemoryChunkRepository(chunks), config, graph_retriever=build_graph_retriever(chunks)
    )
    result = retriever.retrieve("payroll service", AccessContext(tenant_id="tenant-a"))
    assert "secret" not in result.graph_expanded_chunk_ids


def test_graph_expansion_is_disabled_at_depth_zero() -> None:
    chunks = graph_corpus()
    config = PipelineConfig(graph_expansion_depth=0, final_top_k=5)
    retriever = HybridRetriever(
        InMemoryChunkRepository(chunks), config, graph_retriever=build_graph_retriever(chunks)
    )
    result = retriever.retrieve("payroll service", AccessContext(tenant_id="tenant-a"))
    assert result.graph_expanded_chunk_ids == ()
    assert result.strategy == "hybrid_reciprocal_rank"


def test_reranking_reorders_a_bounded_candidate_set(
    repository: InMemoryChunkRepository, employee_access: AccessContext
) -> None:
    config = PipelineConfig(rerank_candidate_count=5)
    retriever = HybridRetriever(repository, config, reranker=LexicalCrossEncoder())
    result = retriever.retrieve("annual leave", employee_access)

    assert result.strategy.endswith("reranker-lexical-v1")
    assert result.chunks[0].fused_score > 1.0


class FailingGraphRetriever:
    def expand(
        self, seeds: Iterable[DocumentChunk], access: AccessContext, *, depth: int, limit: int
    ) -> list[DocumentChunk]:
        raise RuntimeError("graph service unavailable")


class FailingReranker:
    revision = "reranker-broken"

    def rerank(
        self, question: str, candidates: Iterable[DocumentChunk], *, limit: int
    ) -> list[ScoredChunk]:
        raise RuntimeError("reranker service unavailable")


def test_no_configured_dependency_is_ever_reported_as_degraded(
    repository: InMemoryChunkRepository, employee_access: AccessContext, config: PipelineConfig
) -> None:
    result = HybridRetriever(repository, config).retrieve("annual leave", employee_access)
    assert result.degraded_dependencies == ()


def test_a_failing_graph_dependency_degrades_instead_of_crashing() -> None:
    chunks = graph_corpus()
    config = PipelineConfig(graph_expansion_depth=1, final_top_k=5)
    breaker = CircuitBreaker(name="graph_expansion", failure_threshold=5)
    retriever = HybridRetriever(
        InMemoryChunkRepository(chunks),
        config,
        graph_retriever=FailingGraphRetriever(),
        graph_breaker=breaker,
    )
    result = retriever.retrieve("payroll service", AccessContext(tenant_id="tenant-a"))

    assert result.graph_expanded_chunk_ids == ()
    assert [item.chunk.chunk_id for item in result.chunks] == ["payroll"]
    assert any(note.startswith("graph_expansion_failed") for note in result.degraded_dependencies)


def test_a_failing_reranker_degrades_to_the_unranked_candidates(
    repository: InMemoryChunkRepository, employee_access: AccessContext
) -> None:
    config = PipelineConfig(rerank_candidate_count=5)
    breaker = CircuitBreaker(name="reranker", failure_threshold=5)
    retriever = HybridRetriever(
        repository, config, reranker=FailingReranker(), reranker_breaker=breaker
    )
    result = retriever.retrieve("annual leave", employee_access)

    assert [item.chunk.chunk_id for item in result.chunks] == ["public-leave"]
    assert any(note.startswith("reranker_failed") for note in result.degraded_dependencies)


def test_repeated_graph_failures_open_the_circuit_and_fail_fast() -> None:
    chunks = graph_corpus()
    config = PipelineConfig(graph_expansion_depth=1, final_top_k=5)
    breaker = CircuitBreaker(name="graph_expansion", failure_threshold=1)
    retriever = HybridRetriever(
        InMemoryChunkRepository(chunks),
        config,
        graph_retriever=FailingGraphRetriever(),
        graph_breaker=breaker,
    )
    retriever.retrieve("payroll service", AccessContext(tenant_id="tenant-a"))
    result = retriever.retrieve("payroll service", AccessContext(tenant_id="tenant-a"))

    assert "graph_expansion_circuit_open" in result.degraded_dependencies


def test_retrieval_result_exposes_scored_chunks(
    repository: InMemoryChunkRepository,
    employee_access: AccessContext,
    config: PipelineConfig,
) -> None:
    result = HybridRetriever(repository, config).retrieve("annual leave", employee_access)
    scored = result.scored()
    assert [item.chunk.chunk_id for item in scored] == ["public-leave"]
    assert scored[0].score == result.chunks[0].fused_score

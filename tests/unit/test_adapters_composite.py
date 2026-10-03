"""The composite repository is what lets the real stores run the existing retrieval path."""

from dataclasses import dataclass, field

from rag_platform.adapters.composite import CompositeChunkRepository
from rag_platform.context import ContextBuilder
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.repository import ScoredChunk
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService

ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))


def chunk(identifier: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text=text,
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="iv-tenant-a-abc123",
        source_version_id=f"sv-{identifier}",
    )


LEAVE = chunk("leave", "Annual leave requests use the HR portal.")
EXPENSES = chunk("expenses", "Expense claims are approved by the cost centre owner.")


@dataclass
class StubDense:
    results: list[ScoredChunk] = field(default_factory=list)
    calls: list[int] = field(default_factory=list)

    def semantic_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        self.calls.append(limit)
        return list(self.results)


@dataclass
class StubSparse:
    results: list[ScoredChunk] = field(default_factory=list)
    calls: list[int] = field(default_factory=list)

    def lexical_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        self.calls.append(limit)
        return list(self.results)


def test_each_search_is_routed_to_its_own_store() -> None:
    dense = StubDense([ScoredChunk(chunk=LEAVE, score=0.9)])
    sparse = StubSparse([ScoredChunk(chunk=EXPENSES, score=3.1)])
    repository = CompositeChunkRepository(dense=dense, sparse=sparse)

    assert repository.semantic_search("q", ACCESS, limit=5)[0].chunk.chunk_id == "leave"
    assert repository.lexical_search("q", ACCESS, limit=3)[0].chunk.chunk_id == "expenses"
    assert dense.calls == [5]
    assert sparse.calls == [3]


def test_a_missing_dense_store_degrades_to_sparse_only() -> None:
    repository = CompositeChunkRepository(sparse=StubSparse([ScoredChunk(chunk=LEAVE, score=1.0)]))
    assert repository.semantic_search("q", ACCESS, limit=5) == []
    assert repository.lexical_search("q", ACCESS, limit=5)[0].chunk.chunk_id == "leave"


def test_a_missing_sparse_store_degrades_to_dense_only() -> None:
    repository = CompositeChunkRepository(dense=StubDense([ScoredChunk(chunk=LEAVE, score=1.0)]))
    assert repository.lexical_search("q", ACCESS, limit=5) == []
    assert repository.semantic_search("q", ACCESS, limit=5)[0].chunk.chunk_id == "leave"


def test_a_composite_with_no_store_returns_nothing_rather_than_raising() -> None:
    repository = CompositeChunkRepository()
    assert repository.semantic_search("q", ACCESS, limit=5) == []
    assert repository.lexical_search("q", ACCESS, limit=5) == []


def test_the_whole_query_path_runs_unchanged_on_a_composite_repository() -> None:
    """The point of the adapter layer: no retrieval or workflow change to use real stores."""
    config = PipelineConfig()
    repository = CompositeChunkRepository(
        dense=StubDense([ScoredChunk(chunk=LEAVE, score=0.9)]),
        sparse=StubSparse([ScoredChunk(chunk=LEAVE, score=3.1)]),
    )
    service = QueryService(
        HybridRetriever(repository, config), config, ContextBuilder(config)
    )

    response = service.answer("How do I request annual leave?", ACCESS)

    assert response.status == "answered"
    assert response.citations[0].chunk_id == "leave"
    assert response.trace.context_chunk_ids == ["leave"]

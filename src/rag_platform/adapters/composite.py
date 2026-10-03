"""One `ChunkRepository` assembled from a dense store and a sparse store.

`HybridRetriever` asks a single repository for both searches, while Qdrant does dense and
OpenSearch does sparse. Composing them here means hybrid retrieval, fusion, graph expansion,
reranking and context construction need no change at all to run on the real stores
(specification section 8).

Either half can be absent. A store that is not configured returns no candidates rather than
raising, which is the same declared degradation `CircuitBreaker` provides for the graph and the
reranker (specification section 17): hybrid retrieval on one leg still answers.
"""

from dataclasses import dataclass
from typing import Protocol

from rag_platform.models import AccessContext
from rag_platform.repository import ScoredChunk


class DenseIndex(Protocol):
    def semantic_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]: ...


class SparseIndex(Protocol):
    def lexical_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]: ...


@dataclass(slots=True)
class CompositeChunkRepository:
    dense: DenseIndex | None = None
    sparse: SparseIndex | None = None

    def semantic_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        if self.dense is None:
            return []
        return self.dense.semantic_search(question, access, limit=limit)

    def lexical_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        if self.sparse is None:
            return []
        return self.sparse.lexical_search(question, access, limit=limit)

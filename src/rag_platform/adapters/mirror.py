"""Writing an activated index version into the real stores (specification section 7).

The ingestion flow's "build dense, lexical and graph indexes" step is this: once a staged index
version passes validation, its chunks are written into stores named by the `physical_*_index`
identifiers the `IndexVersion` already carries. Because that name is derived from the index
version's fingerprint, every version gets its own collection or index, so activation and
rollback switch between whole indexes instead of mutating the one serving traffic.

Mirroring runs after validation, never before: a rejected index version is retired without
anything reaching a store. Only retrievable chunks are written — a parent chunk exists for
context expansion and must never be a search candidate.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from rag_platform.adapters.embeddings import HashingEmbedder, TextEmbedder
from rag_platform.adapters.opensearch import OpenSearchSparseIndex
from rag_platform.adapters.qdrant import QdrantDenseIndex
from rag_platform.models import DocumentChunk, IndexVersion

if TYPE_CHECKING:  # pragma: no cover - imports used for typing only
    from opensearchpy import OpenSearch
    from qdrant_client import QdrantClient


@dataclass(frozen=True, slots=True)
class MirrorReport:
    physical_index: str
    chunks_written: int


class IndexMirror(Protocol):
    def mirror(
        self, index_version: IndexVersion, chunks: Sequence[DocumentChunk]
    ) -> MirrorReport: ...


@dataclass(slots=True)
class QdrantMirror:
    """Writes each index version into its own Qdrant collection."""

    client: "QdrantClient"
    embedder: TextEmbedder = field(default_factory=HashingEmbedder)

    def mirror(
        self, index_version: IndexVersion, chunks: Sequence[DocumentChunk]
    ) -> MirrorReport:
        index = QdrantDenseIndex(
            client=self.client,
            collection=index_version.physical_dense_index,
            embedder=self.embedder,
        )
        index.ensure_collection()
        written = index.upsert(chunk for chunk in chunks if chunk.retrievable)
        return MirrorReport(
            physical_index=index_version.physical_dense_index, chunks_written=written
        )


@dataclass(slots=True)
class OpenSearchMirror:
    """Writes each index version into its own OpenSearch index."""

    client: "OpenSearch"

    def mirror(
        self, index_version: IndexVersion, chunks: Sequence[DocumentChunk]
    ) -> MirrorReport:
        index = OpenSearchSparseIndex(
            client=self.client, index=index_version.physical_sparse_index
        )
        index.ensure_index()
        written = index.upsert(chunk for chunk in chunks if chunk.retrievable)
        return MirrorReport(
            physical_index=index_version.physical_sparse_index, chunks_written=written
        )


def readers_for(
    index_version: IndexVersion,
    *,
    qdrant_client: "QdrantClient | None" = None,
    opensearch_client: "OpenSearch | None" = None,
    embedder: TextEmbedder | None = None,
) -> tuple[QdrantDenseIndex | None, OpenSearchSparseIndex | None]:
    """The read-side pair for one index version, named by the same physical identifiers."""
    dense = (
        QdrantDenseIndex(
            client=qdrant_client,
            collection=index_version.physical_dense_index,
            embedder=embedder or HashingEmbedder(),
        )
        if qdrant_client is not None
        else None
    )
    sparse = (
        OpenSearchSparseIndex(
            client=opensearch_client, index=index_version.physical_sparse_index
        )
        if opensearch_client is not None
        else None
    )
    return dense, sparse


EmbedderFactory = Callable[[], TextEmbedder]

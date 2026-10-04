"""Reading from whichever index version is active, on the real stores.

Each index version has its own Qdrant collection, OpenSearch index and Neo4j subgraph, named on
the `IndexVersion`. Serving therefore has to resolve the tenant's active version on every query
and read from the stores that version names. That is what makes activation and rollback atomic
for readers: the catalog flips one pointer, and the next query reads a different, complete index.

A leg that has no store configured is served from the catalog's own chunks, so a deployment can
adopt the stores one at a time.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_platform.adapters.embeddings import HashingEmbedder, TextEmbedder
from rag_platform.adapters.mirror import readers_for
from rag_platform.catalog import IndexCatalog
from rag_platform.graph import CatalogGraphRetriever, GraphExpander
from rag_platform.models import AccessContext, DocumentChunk
from rag_platform.repository import CatalogChunkRepository, ScoredChunk

if TYPE_CHECKING:  # pragma: no cover - imports used for typing only
    from neo4j import Driver
    from opensearchpy import OpenSearch
    from qdrant_client import QdrantClient


@dataclass(slots=True)
class ActiveIndexRepository:
    """A `ChunkRepository` over the active index version's dense and sparse stores."""

    catalog: IndexCatalog
    qdrant_client: "QdrantClient | None" = None
    opensearch_client: "OpenSearch | None" = None
    embedder: TextEmbedder = field(default_factory=HashingEmbedder)

    def semantic_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        active = self.catalog.active_index_version(access.tenant_id)
        if active is None:
            return []
        dense, _sparse = readers_for(
            active, qdrant_client=self.qdrant_client, embedder=self.embedder
        )
        if dense is None:
            return CatalogChunkRepository(self.catalog).semantic_search(
                question, access, limit=limit
            )
        return dense.semantic_search(question, access, limit=limit)

    def lexical_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        active = self.catalog.active_index_version(access.tenant_id)
        if active is None:
            return []
        _dense, sparse = readers_for(active, opensearch_client=self.opensearch_client)
        if sparse is None:
            return CatalogChunkRepository(self.catalog).lexical_search(
                question, access, limit=limit
            )
        return sparse.lexical_search(question, access, limit=limit)


@dataclass(slots=True)
class ActiveGraphRetriever:
    """A `GraphExpander` over the active index version's Neo4j subgraph."""

    catalog: IndexCatalog
    driver: "Driver"
    database: str | None = None

    def expand(
        self,
        seeds: Iterable[DocumentChunk],
        access: AccessContext,
        *,
        depth: int,
        limit: int,
    ) -> list[DocumentChunk]:
        from rag_platform.adapters.neo4j_graph import Neo4jGraphRetriever

        active = self.catalog.active_index_version(access.tenant_id)
        if active is None:
            return []
        retriever = Neo4jGraphRetriever(
            driver=self.driver, graph_index=active.physical_graph_index, database=self.database
        )
        return retriever.expand(seeds, access, depth=depth, limit=limit)


def graph_retriever_for(
    catalog: IndexCatalog, driver: "Driver | None", revision: str
) -> GraphExpander:
    if driver is None:
        return CatalogGraphRetriever(catalog, revision)
    return ActiveGraphRetriever(catalog=catalog, driver=driver)

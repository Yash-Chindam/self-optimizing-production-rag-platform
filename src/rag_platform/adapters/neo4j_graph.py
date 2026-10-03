"""Graph retrieval and provenance over Neo4j (specification sections 6 and 8).

The graph holds what the in-process `KnowledgeGraph` holds: entities, typed relationships
between them, the chunk that evidences each relationship, and which chunks mention which
entity. Extraction is the same code (`extract_entities`, `extract_relationships`), so the graph
extractor revision means the same thing whichever store serves it.

Every node carries the `graph_index` of its `IndexVersion` (the `physical_graph_index` the
catalog records). A new index version writes a new, separate subgraph, so activation and
rollback switch between whole graphs the same way they do for Qdrant and OpenSearch.

Traversal is tenant-scoped in Cypher, and the authoritative `is_authorized` check still runs in
Python on every chunk that comes back (specification section 4).
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_platform.adapters.mirror import MirrorReport
from rag_platform.adapters.payload import from_payload, to_payload
from rag_platform.graph import extract_entities, extract_relationships
from rag_platform.models import AccessContext, DocumentChunk, IndexVersion
from rag_platform.repository import is_authorized

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from neo4j import Driver

MAX_DEPTH = 4
OVERFETCH = 4

SCHEMA_STATEMENTS = (
    "CREATE INDEX rag_entity IF NOT EXISTS FOR (e:Entity) ON (e.graph_index, e.name)",
    "CREATE INDEX rag_chunk IF NOT EXISTS FOR (c:Chunk) ON (c.graph_index, c.chunk_id)",
)

WRITE_CHUNKS = """
UNWIND $chunks AS row
MERGE (c:Chunk {graph_index: $graph_index, chunk_id: row.chunk_id})
SET c.tenant_id = row.tenant_id, c.retrievable = row.retrievable, c.payload = row.payload
"""

WRITE_MENTIONS = """
UNWIND $mentions AS row
MATCH (c:Chunk {graph_index: $graph_index, chunk_id: row.chunk_id})
MERGE (e:Entity {graph_index: $graph_index, name: row.entity})
MERGE (c)-[:MENTIONS]->(e)
"""

WRITE_RELATIONSHIPS = """
UNWIND $relationships AS row
MERGE (a:Entity {graph_index: $graph_index, name: row.subject})
MERGE (b:Entity {graph_index: $graph_index, name: row.object})
MERGE (a)-[:RELATED {relation: row.relation, evidence_chunk_id: row.evidence_chunk_id}]->(b)
"""

DROP_GRAPH = "MATCH (n {graph_index: $graph_index}) DETACH DELETE n"


def expansion_query(depth: int) -> str:
    """Cypher cannot parameterize a path length, so the bounded integer is written in.

    A Cypher path never reuses a relationship, so it cannot walk back to the entity it started
    from. The in-process graph can — two hops out and back — which makes a seed entity that has
    any relationship reachable from depth two. Adding the matched seeds back at that depth keeps
    the two stores returning the same expansion.
    """
    hops = max(1, min(int(depth), MAX_DEPTH))
    reached = "reached + seeds" if hops >= 2 else "reached"
    return f"""
MATCH (seed:Entity {{graph_index: $graph_index}})
WHERE seed.name IN $entities
MATCH (seed)-[:RELATED*1..{hops}]-(other:Entity {{graph_index: $graph_index}})
WITH collect(DISTINCT other) AS reached, collect(DISTINCT seed) AS seeds
UNWIND {reached} AS related
MATCH (c:Chunk {{graph_index: $graph_index, tenant_id: $tenant_id}})-[:MENTIONS]->(related)
WHERE c.retrievable AND NOT c.chunk_id IN $seed_ids
RETURN DISTINCT c.chunk_id AS chunk_id, c.payload AS payload
ORDER BY chunk_id
LIMIT $limit
"""


@dataclass(slots=True)
class Neo4jGraphRetriever:
    """`GraphExpander` over one index version's subgraph."""

    driver: "Driver"
    graph_index: str
    database: str | None = None

    def expand(
        self,
        seeds: Iterable[DocumentChunk],
        access: AccessContext,
        *,
        depth: int,
        limit: int,
    ) -> list[DocumentChunk]:
        seed_chunks = list(seeds)
        if depth <= 0 or limit <= 0 or not seed_chunks:
            return []
        entities = sorted(
            {entity for chunk in seed_chunks for entity in extract_entities(chunk.text)}
        )
        if not entities:
            return []
        records, _summary, _keys = self.driver.execute_query(
            expansion_query(depth),
            parameters_={
                "graph_index": self.graph_index,
                "tenant_id": access.tenant_id,
                "entities": entities,
                "seed_ids": sorted(chunk.chunk_id for chunk in seed_chunks),
                "limit": limit * OVERFETCH,
            },
            database_=self.database,
        )
        expanded: list[DocumentChunk] = []
        for record in records:
            chunk = from_payload(json.loads(record["payload"]))
            if not chunk.retrievable or not is_authorized(chunk, access):
                continue
            expanded.append(chunk)
            if len(expanded) == limit:
                break
        return expanded


@dataclass(slots=True)
class Neo4jGraphMirror:
    """Writes each index version's entities, relationships and provenance as its own subgraph."""

    driver: "Driver"
    database: str | None = None

    def ensure_schema(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            self.driver.execute_query(statement, database_=self.database)

    def mirror(
        self, index_version: IndexVersion, chunks: Sequence[DocumentChunk]
    ) -> MirrorReport:
        graph_index = index_version.physical_graph_index
        retrievable = [chunk for chunk in chunks if chunk.retrievable]
        self.ensure_schema()
        self._run(WRITE_CHUNKS, graph_index, chunks=[_chunk_row(chunk) for chunk in retrievable])
        self._run(
            WRITE_MENTIONS,
            graph_index,
            mentions=[
                {"chunk_id": chunk.chunk_id, "entity": entity}
                for chunk in retrievable
                for entity in sorted(extract_entities(chunk.text))
            ],
        )
        self._run(
            WRITE_RELATIONSHIPS,
            graph_index,
            relationships=[
                {
                    "subject": relationship.subject,
                    "relation": relationship.relation,
                    "object": relationship.object,
                    "evidence_chunk_id": relationship.evidence_chunk_id,
                }
                for chunk in retrievable
                for relationship in extract_relationships(chunk)
            ],
        )
        return MirrorReport(physical_index=graph_index, chunks_written=len(retrievable))

    def drop(self, index_version: IndexVersion) -> None:
        self._run(DROP_GRAPH, index_version.physical_graph_index)

    def _run(self, query: str, graph_index: str, **parameters: Any) -> None:
        self.driver.execute_query(
            query,
            parameters_={"graph_index": graph_index, **parameters},
            database_=self.database,
        )


def _chunk_row(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "tenant_id": chunk.tenant_id,
        "retrievable": chunk.retrievable,
        "payload": json.dumps(to_payload(chunk)),
    }

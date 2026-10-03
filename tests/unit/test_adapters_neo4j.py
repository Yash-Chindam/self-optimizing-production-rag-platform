"""Neo4j adapter behavior, verified against a fake driver that captures what was sent."""

import json
from dataclasses import dataclass, field
from typing import Any

from rag_platform.adapters.neo4j_graph import (
    MAX_DEPTH,
    Neo4jGraphMirror,
    Neo4jGraphRetriever,
    expansion_query,
)
from rag_platform.adapters.payload import to_payload
from rag_platform.models import AccessContext, ChunkingConfig, DocumentChunk, IndexVersion

ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))
GRAPH = "neo4j:tenant-a:abc123"


def chunk(
    identifier: str,
    text: str,
    *,
    tenant: str = "tenant-a",
    labels: frozenset[str] = frozenset({"public"}),
    retrievable: bool = True,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id=tenant,
        access_labels=labels,
        text=text,
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="iv-tenant-a-abc123",
        source_version_id=f"sv-{identifier}",
        retrievable=retrievable,
    )


PAYROLL = chunk("payroll", "The Payroll Service depends on Identity Platform.")
IDENTITY = chunk("identity", "Identity Platform enforces hardware keys for every sign-in.")

INDEX_VERSION = IndexVersion(
    index_version_id="iv-tenant-a-abc123",
    tenant_id="tenant-a",
    source_version_ids=("sv-payroll",),
    chunking=ChunkingConfig(),
    embedding_revision="embed-v1",
    analyzer_revision="analyzer-v1",
    graph_extractor_revision="graph-v1",
    physical_dense_index="qdrant:tenant-a:abc123",
    physical_sparse_index="opensearch:tenant-a:abc123",
    physical_graph_index=GRAPH,
    chunk_count=2,
)


@dataclass
class FakeDriver:
    records: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def execute_query(
        self,
        query: str,
        parameters_: dict[str, Any] | None = None,
        database_: str | None = None,
    ) -> tuple[list[dict[str, Any]], None, list[str]]:
        self.calls.append((query, dict(parameters_ or {})))
        return list(self.records), None, ["chunk_id", "payload"]


def record_for(chunk_value: DocumentChunk) -> dict[str, Any]:
    return {"chunk_id": chunk_value.chunk_id, "payload": json.dumps(to_payload(chunk_value))}


def retriever(
    records: list[dict[str, Any]] | None = None,
) -> tuple[Neo4jGraphRetriever, FakeDriver]:
    driver = FakeDriver(records=records or [])
    return Neo4jGraphRetriever(driver=driver, graph_index=GRAPH), driver  # type: ignore[arg-type]


# Traversal ---------------------------------------------------------------------


def test_expansion_is_scoped_to_the_tenant_and_the_index_versions_graph() -> None:
    graph, driver = retriever()
    graph.expand([PAYROLL], ACCESS, depth=1, limit=4)

    _query, parameters = driver.calls[0]
    assert parameters["tenant_id"] == "tenant-a"
    assert parameters["graph_index"] == GRAPH


def test_the_seed_entities_and_seed_ids_are_sent() -> None:
    graph, driver = retriever()
    graph.expand([PAYROLL], ACCESS, depth=1, limit=4)

    _query, parameters = driver.calls[0]
    assert "payroll service" in parameters["entities"]
    assert "identity platform" in parameters["entities"]
    assert parameters["seed_ids"] == ["payroll"]


def test_the_depth_is_written_into_the_path_pattern() -> None:
    assert "RELATED*1..2]" in expansion_query(2)


def test_matched_seeds_are_reachable_again_from_depth_two() -> None:
    """Out and back is two hops in the in-process graph; a Cypher path cannot retrace an edge."""
    assert "UNWIND reached AS related" in expansion_query(1)
    assert "UNWIND reached + seeds AS related" in expansion_query(2)


def test_the_depth_is_clamped_to_the_supported_bound() -> None:
    assert f"RELATED*1..{MAX_DEPTH}]" in expansion_query(99)


def test_an_authorized_related_chunk_is_returned() -> None:
    graph, _ = retriever([record_for(IDENTITY)])
    expanded = graph.expand([PAYROLL], ACCESS, depth=1, limit=4)
    assert [item.chunk_id for item in expanded] == ["identity"]


def test_a_chunk_requiring_an_unheld_label_is_dropped_even_if_the_graph_returns_it() -> None:
    secret = chunk("secret", "Identity Platform incident notes.", labels=frozenset({"security"}))
    graph, _ = retriever([record_for(secret)])
    assert graph.expand([PAYROLL], ACCESS, depth=1, limit=4) == []


def test_a_cross_tenant_chunk_is_dropped_even_if_the_graph_returns_it() -> None:
    foreign = chunk("foreign", "Identity Platform at another tenant.", tenant="tenant-b")
    graph, _ = retriever([record_for(foreign)])
    assert graph.expand([PAYROLL], ACCESS, depth=1, limit=4) == []


def test_results_are_capped_at_the_limit() -> None:
    records = [record_for(chunk(f"c{number}", "Identity Platform note.")) for number in range(5)]
    graph, _ = retriever(records)
    assert len(graph.expand([PAYROLL], ACCESS, depth=1, limit=2)) == 2


def test_depth_zero_never_reaches_the_graph() -> None:
    graph, driver = retriever()
    assert graph.expand([PAYROLL], ACCESS, depth=0, limit=4) == []
    assert driver.calls == []


def test_seeds_without_entities_never_reach_the_graph() -> None:
    graph, driver = retriever()
    plain = chunk("plain", "requests are submitted monthly.")
    assert graph.expand([plain], ACCESS, depth=1, limit=4) == []
    assert driver.calls == []


def test_no_seeds_never_reach_the_graph() -> None:
    graph, driver = retriever()
    assert graph.expand([], ACCESS, depth=1, limit=4) == []
    assert driver.calls == []


# Write path --------------------------------------------------------------------


def mirror() -> tuple[Neo4jGraphMirror, FakeDriver]:
    driver = FakeDriver()
    return Neo4jGraphMirror(driver=driver), driver  # type: ignore[arg-type]


def parameters_with(driver: FakeDriver, key: str) -> dict[str, Any]:
    return next(parameters for _query, parameters in driver.calls if key in parameters)


def test_the_mirror_writes_into_the_graph_the_index_version_names() -> None:
    graph_mirror, driver = mirror()
    report = graph_mirror.mirror(INDEX_VERSION, [PAYROLL, IDENTITY])

    assert report.physical_index == GRAPH
    assert report.chunks_written == 2
    assert parameters_with(driver, "chunks")["graph_index"] == GRAPH


def test_relationships_carry_the_chunk_that_evidences_them() -> None:
    graph_mirror, driver = mirror()
    graph_mirror.mirror(INDEX_VERSION, [PAYROLL, IDENTITY])

    relationships = parameters_with(driver, "relationships")["relationships"]
    assert relationships == [
        {
            "subject": "payroll service",
            "relation": "depends_on",
            "object": "identity platform",
            "evidence_chunk_id": "payroll",
        }
    ]


def test_every_entity_mention_is_written() -> None:
    graph_mirror, driver = mirror()
    graph_mirror.mirror(INDEX_VERSION, [PAYROLL, IDENTITY])

    mentions = parameters_with(driver, "mentions")["mentions"]
    assert {"chunk_id": "identity", "entity": "identity platform"} in mentions
    assert {"chunk_id": "payroll", "entity": "payroll service"} in mentions


def test_parent_chunks_are_never_written_to_the_graph() -> None:
    graph_mirror, driver = mirror()
    parent = chunk("parent", "The Payroll Service depends on Identity Platform.", retrievable=False)
    report = graph_mirror.mirror(INDEX_VERSION, [PAYROLL, parent])

    assert report.chunks_written == 1
    assert [row["chunk_id"] for row in parameters_with(driver, "chunks")["chunks"]] == ["payroll"]


def test_the_schema_is_ensured_before_writing() -> None:
    graph_mirror, driver = mirror()
    graph_mirror.mirror(INDEX_VERSION, [PAYROLL])
    assert driver.calls[0][0].startswith("CREATE INDEX")


def test_drop_removes_only_the_index_versions_subgraph() -> None:
    graph_mirror, driver = mirror()
    graph_mirror.drop(INDEX_VERSION)

    query, parameters = driver.calls[0]
    assert "DETACH DELETE" in query
    assert parameters == {"graph_index": GRAPH}

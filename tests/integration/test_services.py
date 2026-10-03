"""Adapter tests against real Neo4j, PostgreSQL, MinIO and Redis.

Skipped unless the matching environment variable is set, so the default `pytest` run stays
offline. `docker compose up -d --wait` brings every service up with the credentials CI uses.
"""

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from rag_platform.catalog import IndexCatalog
from rag_platform.graph import GraphRetriever, KnowledgeGraph
from rag_platform.ingestion import IngestionPipeline
from rag_platform.models import (
    AccessContext,
    ChunkingConfig,
    DocumentChunk,
    IndexVersion,
    SourceRegistration,
)

NEO4J_URL = os.environ.get("RAG_NEO4J_URL")
POSTGRES_DSN = os.environ.get("RAG_POSTGRES_DSN")
MINIO_ENDPOINT = os.environ.get("RAG_MINIO_ENDPOINT")
REDIS_URL = os.environ.get("RAG_REDIS_URL")

ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))


def chunk(
    identifier: str,
    text: str,
    *,
    tenant: str = "tenant-a",
    labels: frozenset[str] = frozenset({"public"}),
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id=tenant,
        access_labels=labels,
        text=text,
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="iv-test",
        source_version_id=f"sv-{identifier}",
    )


GRAPH_CORPUS = [
    chunk("payroll", "The Payroll Service depends on Identity Platform."),
    chunk("identity", "Identity Platform enforces hardware keys for every administrator sign-in."),
    chunk("keys", "Identity Platform is owned by the Security Office."),
    chunk("office", "The Security Office publishes the quarterly audit calendar."),
    chunk("risk-link", "The Security Office reports to the Risk Committee."),
    chunk("committee", "The Risk Committee meets monthly."),
    chunk("catering", "Catering guidance for the Dublin Office."),
    chunk("secret", "Identity Platform incident notes.", labels=frozenset({"security"})),
    chunk("foreign", "Identity Platform at another company.", tenant="tenant-b"),
]


def index_version(suffix: str) -> IndexVersion:
    return IndexVersion(
        index_version_id=f"iv-{suffix}",
        tenant_id="tenant-a",
        source_version_ids=("sv-test",),
        chunking=ChunkingConfig(),
        embedding_revision="embed-v1",
        analyzer_revision="analyzer-v1",
        graph_extractor_revision="graph-v1",
        physical_dense_index=f"qdrant:{suffix}",
        physical_sparse_index=f"opensearch:{suffix}",
        physical_graph_index=f"neo4j:{suffix}",
        chunk_count=len(GRAPH_CORPUS),
    )


# Neo4j -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def neo4j_graph() -> Iterator[Any]:
    if not NEO4J_URL:
        pytest.skip("RAG_NEO4J_URL is not set")
    from neo4j import GraphDatabase

    from rag_platform.adapters.neo4j_graph import Neo4jGraphMirror, Neo4jGraphRetriever

    driver = GraphDatabase.driver(
        NEO4J_URL,
        auth=(
            os.environ.get("RAG_NEO4J_USER", "neo4j"),
            os.environ.get("RAG_NEO4J_PASSWORD", ""),
        ),
    )
    version = index_version(uuid.uuid4().hex[:8])
    mirror = Neo4jGraphMirror(driver=driver)
    mirror.mirror(version, GRAPH_CORPUS)
    try:
        yield Neo4jGraphRetriever(driver=driver, graph_index=version.physical_graph_index)
    finally:
        mirror.drop(version)
        driver.close()


BY_ID = {item.chunk_id: item for item in GRAPH_CORPUS}


def in_process_expansion(seed: str, depth: int) -> list[str]:
    graph = KnowledgeGraph()
    graph.index(GRAPH_CORPUS)
    expanded = GraphRetriever(graph, GRAPH_CORPUS).expand(
        [BY_ID[seed]], ACCESS, depth=depth, limit=20
    )
    return sorted(item.chunk_id for item in expanded)


def neo4j_expansion(graph: Any, seed: str, depth: int, access: AccessContext = ACCESS) -> set[str]:
    return {item.chunk_id for item in graph.expand([BY_ID[seed]], access, depth=depth, limit=20)}


@pytest.mark.integration
@pytest.mark.parametrize("seed", ["payroll", "office", "committee"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_neo4j_expansion_matches_the_in_process_graph(
    neo4j_graph: Any, seed: str, depth: int
) -> None:
    """The same extractor revision must mean the same expansion whichever store serves it."""
    assert sorted(neo4j_expansion(neo4j_graph, seed, depth)) == in_process_expansion(seed, depth)


@pytest.mark.integration
def test_neo4j_reaches_evidence_two_hops_away(neo4j_graph: Any) -> None:
    assert "committee" not in neo4j_expansion(neo4j_graph, "payroll", 1)
    assert "committee" in neo4j_expansion(neo4j_graph, "payroll", 2)


@pytest.mark.integration
def test_neo4j_never_crosses_a_tenant_or_label_boundary(neo4j_graph: Any) -> None:
    expanded = neo4j_expansion(neo4j_graph, "payroll", 3)
    assert "secret" not in expanded
    assert "foreign" not in expanded
    assert "catering" not in expanded


@pytest.mark.integration
def test_neo4j_releases_a_labelled_chunk_once_the_label_is_held(neo4j_graph: Any) -> None:
    security = AccessContext(tenant_id="tenant-a", labels=frozenset({"public", "security"}))
    assert "secret" in neo4j_expansion(neo4j_graph, "payroll", 1, security)


# PostgreSQL --------------------------------------------------------------------

HANDBOOK = """# ACME Employee Handbook

## Annual leave

Annual leave requests must be submitted in the HR portal at least five working days ahead.
"""


@pytest.fixture
def postgres_connection() -> Iterator[Any]:
    if not POSTGRES_DSN:
        pytest.skip("RAG_POSTGRES_DSN is not set")
    import psycopg

    with psycopg.connect(POSTGRES_DSN) as connection:
        connection.execute(
            "DROP TABLE IF EXISTS rag_chunks, rag_activations, rag_index_versions, "
            "rag_source_versions"
        )
        connection.commit()
        yield connection


def registration(tenant: str) -> SourceRegistration:
    return SourceRegistration(
        source_id="handbook",
        tenant_id=tenant,
        owner="people-operations",
        source_uri="https://example.test/handbook",
        source_title="Handbook",
    )


@pytest.mark.integration
def test_the_catalog_survives_a_restart(postgres_connection: Any) -> None:
    from rag_platform.adapters.postgres import PostgresIndexCatalog

    first = PostgresIndexCatalog(postgres_connection)
    first.ensure_schema()
    result = IngestionPipeline(catalog=first).ingest(registration("tenant-a"), HANDBOOK)

    restarted = PostgresIndexCatalog(postgres_connection)
    restarted.load()

    assert restarted.active_index_version("tenant-a") == result.index_version
    assert restarted.active_chunks("tenant-a") == first.active_chunks("tenant-a")
    assert restarted.source_version(result.source_version.source_version_id) == (
        result.source_version
    )


@pytest.mark.integration
def test_rollback_after_a_restart_needs_no_reingestion(postgres_connection: Any) -> None:
    from rag_platform.adapters.postgres import PostgresIndexCatalog

    first = PostgresIndexCatalog(postgres_connection)
    first.ensure_schema()
    pipeline = IngestionPipeline(catalog=first)
    original = pipeline.ingest(registration("tenant-a"), HANDBOOK)
    pipeline.ingest(registration("tenant-a"), HANDBOOK + "\n## Expenses\n\nClaims are monthly.\n")

    restarted = PostgresIndexCatalog(postgres_connection)
    restarted.load()
    restored = restarted.rollback("tenant-a")

    assert restored.index_version_id == original.index_version.index_version_id
    assert restored.status == "active"

    again = PostgresIndexCatalog(postgres_connection)
    again.load()
    assert again.active_index_version("tenant-a") == restored


@pytest.mark.integration
def test_the_durable_catalog_behaves_like_the_in_process_one(postgres_connection: Any) -> None:
    from rag_platform.adapters.postgres import PostgresIndexCatalog

    durable = PostgresIndexCatalog(postgres_connection)
    durable.ensure_schema()
    reference = IndexCatalog()
    for catalog in (durable, reference):
        IngestionPipeline(catalog=catalog).ingest(registration("tenant-a"), HANDBOOK)

    assert durable.active_index_version("tenant-a") == reference.active_index_version("tenant-a")
    assert durable.active_chunks("tenant-a") == reference.active_chunks("tenant-a")


# MinIO -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minio_store() -> Any:
    if not MINIO_ENDPOINT:
        pytest.skip("RAG_MINIO_ENDPOINT is not set")
    from minio import Minio

    from rag_platform.adapters.originals import MinioOriginalStore

    client = Minio(
        MINIO_ENDPOINT,
        access_key=os.environ.get("RAG_MINIO_ACCESS_KEY", ""),
        secret_key=os.environ.get("RAG_MINIO_SECRET_KEY", ""),
        secure=False,
    )
    store = MinioOriginalStore(client=client, bucket=f"rag-test-{uuid.uuid4().hex[:8]}")
    store.ensure_bucket()
    return store


@pytest.mark.integration
def test_minio_round_trips_an_original(minio_store: Any) -> None:
    key = minio_store.put("tenant-a", "handbook", "hash-1", "Übersicht — annual leave")
    assert minio_store.get(key) == "Übersicht — annual leave"


@pytest.mark.integration
def test_minio_keeps_the_first_write_for_a_content_hash(minio_store: Any) -> None:
    key = minio_store.put("tenant-a", "handbook", "hash-2", "first")
    minio_store.put("tenant-a", "handbook", "hash-2", "second")
    assert minio_store.get(key) == "first"


@pytest.mark.integration
def test_ingestion_stores_and_can_delete_the_original_in_minio(minio_store: Any) -> None:
    from minio.error import S3Error

    from rag_platform.models import RetentionPolicy

    kept = IngestionPipeline(catalog=IndexCatalog(), originals=minio_store).ingest(
        registration("tenant-keep"), HANDBOOK
    )
    stored = next(
        check for check in kept.report.validation_checks if check.startswith("original_stored:")
    )
    assert minio_store.get(stored.removeprefix("original_stored:")) == HANDBOOK

    deleting = registration("tenant-delete").model_copy(
        update={"retention": RetentionPolicy(delete_original_after_indexing=True)}
    )
    IngestionPipeline(catalog=IndexCatalog(), originals=minio_store).ingest(deleting, HANDBOOK)
    with pytest.raises(S3Error):
        minio_store.get(f"tenant-delete/handbook/{kept.source_version.content_hash}")


# Redis -------------------------------------------------------------------------


@pytest.mark.integration
def test_redis_serves_a_repeated_question_from_the_cache() -> None:
    if not REDIS_URL:
        pytest.skip("RAG_REDIS_URL is not set")
    import redis

    from rag_platform.adapters.cache import CachedQueryService, RedisAnswerCache
    from rag_platform.bootstrap import build_platform

    client = redis.Redis.from_url(REDIS_URL)
    platform = build_platform()
    cached = CachedQueryService(
        service=platform.query_service,
        cache=RedisAnswerCache(client=client, prefix=f"rag:test:{uuid.uuid4().hex[:8]}:"),
        active_index_version=lambda tenant: (
            active.index_version_id
            if (active := platform.catalog.active_index_version(tenant)) is not None
            else None
        ),
        ttl_seconds=30,
    )
    access = AccessContext(tenant_id="tenant-acme", labels=frozenset({"public", "employees"}))

    first = cached.answer("How do I request annual leave?", access)
    second = cached.answer("How do I request annual leave?", access)

    assert first.status == "answered"
    assert first.trace.served_from_cache is False
    assert second.trace.served_from_cache is True
    assert second.answer == first.answer
    assert second.trace.degraded_dependencies == []

    finance = AccessContext(tenant_id="tenant-acme", labels=frozenset({"public", "finance"}))
    other = cached.answer("How do I request annual leave?", finance)
    assert other.trace.served_from_cache is False

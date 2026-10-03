"""Adapter tests against real Qdrant and OpenSearch instances.

These are skipped unless the services are reachable, so the default `pytest` run stays offline.
Bring them up with `docker compose up -d qdrant opensearch` and set `RAG_QDRANT_URL` and
`RAG_OPENSEARCH_URL` (the compose defaults are what CI uses).

What a fake client cannot prove is exactly what matters here: that the filters this platform
builds mean the same thing to the real stores that they mean to `is_authorized`. Each test seeds
a tenant-mixed, label-mixed corpus and asserts the store never hands back evidence the caller is
not entitled to.
"""

import os
import uuid
from collections.abc import Iterator

import pytest

from rag_platform.adapters.embeddings import HashingEmbedder
from rag_platform.adapters.opensearch import OpenSearchSparseIndex
from rag_platform.adapters.qdrant import QdrantDenseIndex
from rag_platform.models import AccessContext, DocumentChunk

QDRANT_URL = os.environ.get("RAG_QDRANT_URL")
OPENSEARCH_URL = os.environ.get("RAG_OPENSEARCH_URL")

ACME_EMPLOYEE = AccessContext(tenant_id="tenant-acme", labels=frozenset({"public", "employees"}))
ACME_FINANCE = AccessContext(
    tenant_id="tenant-acme", labels=frozenset({"public", "employees", "finance"})
)
GLOBEX_EMPLOYEE = AccessContext(tenant_id="tenant-globex", labels=frozenset({"public"}))


def chunk(
    identifier: str,
    tenant: str,
    labels: frozenset[str],
    text: str,
    *,
    retrievable: bool = True,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id=tenant,
        access_labels=labels,
        text=text,
        source_uri=f"https://knowledge.example/{identifier}",
        source_title=identifier,
        index_version="iv-test",
        source_version_id=f"sv-{identifier}",
        retrievable=retrievable,
    )


CORPUS = [
    chunk(
        "acme-leave",
        "tenant-acme",
        frozenset({"public"}),
        "Annual leave requests must be submitted in the HR portal five working days ahead.",
    ),
    chunk(
        "acme-forecast",
        "tenant-acme",
        frozenset({"finance"}),
        "The confidential finance forecast projects twelve percent revenue growth.",
    ),
    chunk(
        "acme-board",
        "tenant-acme",
        frozenset({"finance", "management"}),
        "Board compensation review notes for the annual leave of the chief executive.",
    ),
    chunk(
        "globex-leave",
        "tenant-globex",
        frozenset({"public"}),
        "Globex employees submit annual leave requests by email to their line manager.",
    ),
    chunk(
        "acme-leave-parent",
        "tenant-acme",
        frozenset({"public"}),
        "Handbook section covering annual leave in full.",
        retrievable=False,
    ),
]


@pytest.fixture(scope="module")
def qdrant_index() -> Iterator[QdrantDenseIndex]:
    if not QDRANT_URL:
        pytest.skip("RAG_QDRANT_URL is not set")
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL)
    index = QdrantDenseIndex(
        client=client,
        collection=f"rag-test-{uuid.uuid4().hex[:8]}",
        embedder=HashingEmbedder(dimensions=128),
    )
    index.ensure_collection()
    index.upsert(CORPUS)
    try:
        yield index
    finally:
        index.drop()


@pytest.fixture(scope="module")
def opensearch_index() -> Iterator[OpenSearchSparseIndex]:
    if not OPENSEARCH_URL:
        pytest.skip("RAG_OPENSEARCH_URL is not set")
    from opensearchpy import OpenSearch

    client = OpenSearch(hosts=[OPENSEARCH_URL], verify_certs=False, ssl_show_warn=False)
    index = OpenSearchSparseIndex(client=client, index=f"rag-test-{uuid.uuid4().hex[:8]}")
    index.ensure_index()
    index.upsert(CORPUS)
    try:
        yield index
    finally:
        index.drop()


# Qdrant ------------------------------------------------------------------------


@pytest.mark.integration
def test_qdrant_returns_authorized_evidence(qdrant_index: QdrantDenseIndex) -> None:
    results = qdrant_index.semantic_search("annual leave request", ACME_EMPLOYEE, limit=5)
    assert "acme-leave" in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_qdrant_never_crosses_a_tenant_boundary(qdrant_index: QdrantDenseIndex) -> None:
    results = qdrant_index.semantic_search("annual leave request", ACME_EMPLOYEE, limit=10)
    assert all(item.chunk.tenant_id == "tenant-acme" for item in results)
    assert "globex-leave" not in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_qdrant_withholds_a_label_the_caller_lacks(qdrant_index: QdrantDenseIndex) -> None:
    results = qdrant_index.semantic_search("revenue growth forecast", ACME_EMPLOYEE, limit=10)
    assert "acme-forecast" not in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_qdrant_releases_the_same_chunk_once_the_label_is_held(
    qdrant_index: QdrantDenseIndex,
) -> None:
    results = qdrant_index.semantic_search("revenue growth forecast", ACME_FINANCE, limit=10)
    assert "acme-forecast" in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_qdrant_requires_every_label_of_a_multi_label_chunk(
    qdrant_index: QdrantDenseIndex,
) -> None:
    """`finance` alone is not enough for a chunk labelled finance *and* management."""
    results = qdrant_index.semantic_search("board compensation review", ACME_FINANCE, limit=10)
    assert "acme-board" not in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_qdrant_never_returns_a_parent_chunk(qdrant_index: QdrantDenseIndex) -> None:
    results = qdrant_index.semantic_search("annual leave", ACME_EMPLOYEE, limit=10)
    assert "acme-leave-parent" not in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_qdrant_respects_the_limit(qdrant_index: QdrantDenseIndex) -> None:
    assert len(qdrant_index.semantic_search("annual leave", ACME_EMPLOYEE, limit=1)) <= 1


@pytest.mark.integration
def test_qdrant_returns_nothing_for_an_unknown_tenant(qdrant_index: QdrantDenseIndex) -> None:
    stranger = AccessContext(tenant_id="tenant-nobody", labels=frozenset({"public"}))
    assert qdrant_index.semantic_search("annual leave", stranger, limit=5) == []


# OpenSearch --------------------------------------------------------------------


@pytest.mark.integration
def test_opensearch_returns_authorized_evidence(
    opensearch_index: OpenSearchSparseIndex,
) -> None:
    results = opensearch_index.lexical_search("annual leave portal", ACME_EMPLOYEE, limit=5)
    assert "acme-leave" in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_opensearch_never_crosses_a_tenant_boundary(
    opensearch_index: OpenSearchSparseIndex,
) -> None:
    results = opensearch_index.lexical_search("annual leave", ACME_EMPLOYEE, limit=10)
    assert all(item.chunk.tenant_id == "tenant-acme" for item in results)
    assert "globex-leave" not in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_opensearch_withholds_a_label_the_caller_lacks(
    opensearch_index: OpenSearchSparseIndex,
) -> None:
    results = opensearch_index.lexical_search("confidential forecast", ACME_EMPLOYEE, limit=10)
    assert "acme-forecast" not in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_opensearch_releases_the_same_chunk_once_the_label_is_held(
    opensearch_index: OpenSearchSparseIndex,
) -> None:
    results = opensearch_index.lexical_search("confidential forecast", ACME_FINANCE, limit=10)
    assert "acme-forecast" in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_the_opensearch_subset_filter_is_exact(
    opensearch_index: OpenSearchSparseIndex,
) -> None:
    """terms_set must reject a chunk needing finance *and* management from a finance-only caller."""
    results = opensearch_index.lexical_search("board compensation", ACME_FINANCE, limit=10)
    assert "acme-board" not in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_opensearch_never_returns_a_parent_chunk(
    opensearch_index: OpenSearchSparseIndex,
) -> None:
    results = opensearch_index.lexical_search("handbook annual leave", ACME_EMPLOYEE, limit=10)
    assert "acme-leave-parent" not in {item.chunk.chunk_id for item in results}


@pytest.mark.integration
def test_opensearch_serves_the_other_tenant_its_own_evidence(
    opensearch_index: OpenSearchSparseIndex,
) -> None:
    results = opensearch_index.lexical_search("annual leave", GLOBEX_EMPLOYEE, limit=10)
    assert {item.chunk.chunk_id for item in results} == {"globex-leave"}


# Hybrid over both stores -------------------------------------------------------


@pytest.mark.integration
def test_the_whole_query_path_runs_on_the_real_stores(
    qdrant_index: QdrantDenseIndex, opensearch_index: OpenSearchSparseIndex
) -> None:
    from rag_platform.adapters.composite import CompositeChunkRepository
    from rag_platform.context import ContextBuilder
    from rag_platform.models import PipelineConfig
    from rag_platform.retrieval import HybridRetriever
    from rag_platform.service import QueryService

    config = PipelineConfig()
    repository = CompositeChunkRepository(dense=qdrant_index, sparse=opensearch_index)
    service = QueryService(HybridRetriever(repository, config), config, ContextBuilder(config))

    response = service.answer("How do I request annual leave?", ACME_EMPLOYEE)

    assert response.status == "answered"
    assert response.citations
    assert "acme-forecast" not in response.trace.context_chunk_ids
    assert "globex-leave" not in response.trace.context_chunk_ids


@pytest.mark.integration
def test_the_real_stores_still_abstain_without_evidence(
    qdrant_index: QdrantDenseIndex, opensearch_index: OpenSearchSparseIndex
) -> None:
    from rag_platform.adapters.composite import CompositeChunkRepository
    from rag_platform.context import ContextBuilder
    from rag_platform.models import PipelineConfig
    from rag_platform.retrieval import HybridRetriever
    from rag_platform.service import QueryService

    config = PipelineConfig()
    repository = CompositeChunkRepository(dense=qdrant_index, sparse=opensearch_index)
    service = QueryService(HybridRetriever(repository, config), config, ContextBuilder(config))

    response = service.answer("When is the next lunar eclipse?", ACME_EMPLOYEE)

    assert response.status == "insufficient_evidence"
    assert response.citations == []

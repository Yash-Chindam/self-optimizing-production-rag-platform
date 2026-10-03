"""Qdrant adapter behavior, verified against a fake client that captures what was sent.

These tests are about the adapter's half of the contract: that a query always carries the
mandatory tenant filter, and that the Python authorization recheck is what actually decides
whether a chunk is returned. The live-service tests in tests/integration/test_stores.py cover
the wire protocol against a real Qdrant.
"""

from dataclasses import dataclass, field
from typing import Any

from rag_platform.adapters.embeddings import HashingEmbedder
from rag_platform.adapters.payload import to_payload
from rag_platform.adapters.qdrant import QdrantDenseIndex
from rag_platform.models import AccessContext, DocumentChunk

EMPLOYEE = AccessContext(tenant_id="tenant-a", labels=frozenset({"public", "employees"}))


def chunk(
    identifier: str,
    *,
    tenant: str = "tenant-a",
    labels: frozenset[str] = frozenset({"public"}),
    retrievable: bool = True,
    text: str = "Annual leave requests use the HR portal.",
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


@dataclass
class FakeHit:
    payload: dict[str, Any]
    score: float


@dataclass
class FakeResponse:
    points: list[FakeHit]


@dataclass
class FakeQdrantClient:
    """Captures calls instead of talking to a server."""

    hits: list[FakeHit] = field(default_factory=list)
    existing: set[str] = field(default_factory=set)
    queries: list[dict[str, Any]] = field(default_factory=list)
    upserted: list[Any] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    payload_indexes: list[tuple[str, Any]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.existing

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.created.append(collection_name)
        self.existing.add(collection_name)

    def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: Any
    ) -> None:
        self.payload_indexes.append((field_name, field_schema))

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)
        self.existing.discard(collection_name)

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        self.upserted.extend(points)

    def query_points(self, **kwargs: Any) -> FakeResponse:
        self.queries.append(kwargs)
        return FakeResponse(points=list(self.hits))


def build_index(hits: list[FakeHit] | None = None) -> tuple[QdrantDenseIndex, FakeQdrantClient]:
    client = FakeQdrantClient(hits=hits or [])
    index = QdrantDenseIndex(
        client=client,  # type: ignore[arg-type]
        collection="qdrant:tenant-a:abc123",
        embedder=HashingEmbedder(dimensions=64),
    )
    return index, client


def hit_for(chunk_value: DocumentChunk, score: float = 0.9) -> FakeHit:
    return FakeHit(payload=to_payload(chunk_value), score=score)


# Query construction ------------------------------------------------------------


def test_every_query_carries_a_mandatory_tenant_filter() -> None:
    index, client = build_index()
    index.semantic_search("annual leave", EMPLOYEE, limit=4)

    query_filter = client.queries[0]["query_filter"]
    tenant_conditions = [
        condition
        for condition in query_filter.must
        if getattr(condition, "key", None) == "tenant_id"
    ]
    assert len(tenant_conditions) == 1
    assert tenant_conditions[0].match.value == "tenant-a"


def test_every_query_excludes_non_retrievable_parent_chunks() -> None:
    index, client = build_index()
    index.semantic_search("annual leave", EMPLOYEE, limit=4)

    query_filter = client.queries[0]["query_filter"]
    retrievable = [
        condition
        for condition in query_filter.must
        if getattr(condition, "key", None) == "retrievable"
    ]
    assert retrievable[0].match.value is True


def test_the_callers_labels_are_offered_to_the_store_as_a_narrowing_filter() -> None:
    index, client = build_index()
    index.semantic_search("annual leave", EMPLOYEE, limit=4)

    should = client.queries[0]["query_filter"].should
    assert should is not None
    assert should[0].key == "required_labels"
    assert sorted(should[0].match.any) == ["employees", "public"]


def test_the_query_overfetches_so_the_recheck_can_drop_hits() -> None:
    index, client = build_index()
    index.semantic_search("annual leave", EMPLOYEE, limit=4)
    assert client.queries[0]["limit"] > 4


def test_a_zero_limit_never_reaches_the_store() -> None:
    index, client = build_index()
    assert index.semantic_search("annual leave", EMPLOYEE, limit=0) == []
    assert client.queries == []


# Authorization recheck ---------------------------------------------------------


def test_an_authorized_hit_is_returned_with_its_score() -> None:
    index, _ = build_index([hit_for(chunk("leave"), score=0.75)])
    results = index.semantic_search("annual leave", EMPLOYEE, limit=4)

    assert [item.chunk.chunk_id for item in results] == ["leave"]
    assert results[0].score == 0.75


def test_a_cross_tenant_hit_is_dropped_even_if_the_store_returns_it() -> None:
    leaked = chunk("other-tenant", tenant="tenant-b")
    index, _ = build_index([hit_for(leaked)])
    assert index.semantic_search("annual leave", EMPLOYEE, limit=4) == []


def test_a_hit_requiring_an_unheld_label_is_dropped() -> None:
    confidential = chunk("forecast", labels=frozenset({"finance"}))
    index, _ = build_index([hit_for(confidential)])
    assert index.semantic_search("annual leave", EMPLOYEE, limit=4) == []


def test_a_non_retrievable_hit_is_dropped() -> None:
    parent = chunk("leave-parent", retrievable=False)
    index, _ = build_index([hit_for(parent)])
    assert index.semantic_search("annual leave", EMPLOYEE, limit=4) == []


def test_results_are_capped_at_the_requested_limit() -> None:
    hits = [hit_for(chunk(f"leave-{number}"), score=1.0 - number / 10) for number in range(6)]
    index, _ = build_index(hits)
    assert len(index.semantic_search("annual leave", EMPLOYEE, limit=2)) == 2


def test_a_hit_without_a_payload_is_skipped() -> None:
    index, _ = build_index([FakeHit(payload={}, score=0.5)])
    assert index.semantic_search("annual leave", EMPLOYEE, limit=4) == []


# Write path --------------------------------------------------------------------


def test_the_collection_is_created_with_a_payload_index_per_filtered_field() -> None:
    index, client = build_index()
    index.ensure_collection()

    assert client.created == ["qdrant:tenant-a:abc123"]
    assert [name for name, _schema in client.payload_indexes] == [
        "tenant_id",
        "required_labels",
        "retrievable",
    ]


def test_an_existing_collection_is_left_alone() -> None:
    index, client = build_index()
    client.existing.add("qdrant:tenant-a:abc123")
    index.ensure_collection()
    assert client.created == []


def test_upsert_embeds_every_chunk_once() -> None:
    index, client = build_index()
    assert index.upsert([chunk("a"), chunk("b")]) == 2
    assert len(client.upserted) == 2
    assert all(len(point.vector) == 64 for point in client.upserted)


def test_upserting_nothing_never_calls_the_store() -> None:
    index, client = build_index()
    assert index.upsert([]) == 0
    assert client.upserted == []


def test_drop_removes_the_collection_when_it_exists() -> None:
    index, client = build_index()
    client.existing.add("qdrant:tenant-a:abc123")
    index.drop()
    assert client.deleted == ["qdrant:tenant-a:abc123"]


def test_dropping_a_missing_collection_is_a_no_op() -> None:
    index, client = build_index()
    index.drop()
    assert client.deleted == []

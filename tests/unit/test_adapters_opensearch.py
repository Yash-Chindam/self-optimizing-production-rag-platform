"""OpenSearch adapter behavior, verified against a fake client that captures what was sent."""

from dataclasses import dataclass, field
from typing import Any

from rag_platform.adapters.opensearch import OpenSearchSparseIndex, document_for
from rag_platform.models import AccessContext, DocumentChunk

EMPLOYEE = AccessContext(tenant_id="tenant-a", labels=frozenset({"public", "employees"}))


def chunk(
    identifier: str,
    *,
    tenant: str = "tenant-a",
    labels: frozenset[str] = frozenset({"public"}),
    retrievable: bool = True,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id=tenant,
        access_labels=labels,
        text="Annual leave requests use the HR portal.",
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="iv-tenant-a-abc123",
        source_version_id=f"sv-{identifier}",
        retrievable=retrievable,
    )


@dataclass
class FakeIndices:
    present: set[str] = field(default_factory=set)
    created: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def exists(self, index: str) -> bool:
        return index in self.present

    def create(self, index: str, body: dict[str, Any]) -> None:
        self.created.append(index)
        self.present.add(index)

    def delete(self, index: str) -> None:
        self.deleted.append(index)
        self.present.discard(index)


@dataclass
class FakeOpenSearchClient:
    hits: list[dict[str, Any]] = field(default_factory=list)
    indices: FakeIndices = field(default_factory=FakeIndices)
    searches: list[dict[str, Any]] = field(default_factory=list)
    bulk_bodies: list[list[dict[str, Any]]] = field(default_factory=list)

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.searches.append(body)
        return {"hits": {"hits": list(self.hits)}}

    def bulk(self, body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        self.bulk_bodies.append(body)
        return {"errors": False}


def build_index(
    hits: list[dict[str, Any]] | None = None,
) -> tuple[OpenSearchSparseIndex, FakeOpenSearchClient]:
    client = FakeOpenSearchClient(hits=hits or [])
    return (
        OpenSearchSparseIndex(client=client, index="opensearch:tenant-a:abc123"),  # type: ignore[arg-type]
        client,
    )


def hit_for(chunk_value: DocumentChunk, score: float = 3.5) -> dict[str, Any]:
    return {"_source": document_for(chunk_value), "_score": score}


def filter_clause(body: dict[str, Any], key: str) -> dict[str, Any]:
    for clause in body["query"]["bool"]["filter"]:
        if key in clause:
            return dict(clause[key])
    raise AssertionError(f"no {key} clause in {body}")


# Query construction ------------------------------------------------------------


def test_every_query_carries_a_mandatory_tenant_term() -> None:
    index, client = build_index()
    index.lexical_search("annual leave", EMPLOYEE, limit=4)
    assert filter_clause(client.searches[0], "term")["tenant_id"] == "tenant-a"


def test_the_label_subset_test_is_exact() -> None:
    """terms_set + minimum_should_match_field is the real subset semantics, not an approximation."""
    index, client = build_index()
    index.lexical_search("annual leave", EMPLOYEE, limit=4)

    terms_set = filter_clause(client.searches[0], "terms_set")["required_labels"]
    assert sorted(terms_set["terms"]) == ["employees", "public"]
    assert terms_set["minimum_should_match_field"] == "required_label_count"


def test_the_question_is_the_scored_clause() -> None:
    index, client = build_index()
    index.lexical_search("annual leave", EMPLOYEE, limit=4)
    must = client.searches[0]["query"]["bool"]["must"]
    assert must[0]["match"]["text"]["query"] == "annual leave"


def test_the_requested_limit_is_the_query_size() -> None:
    index, client = build_index()
    index.lexical_search("annual leave", EMPLOYEE, limit=7)
    assert client.searches[0]["size"] == 7


def test_a_zero_limit_never_reaches_the_store() -> None:
    index, client = build_index()
    assert index.lexical_search("annual leave", EMPLOYEE, limit=0) == []
    assert client.searches == []


# Authorization recheck ---------------------------------------------------------


def test_an_authorized_hit_is_returned_with_its_bm25_score() -> None:
    index, _ = build_index([hit_for(chunk("leave"), score=4.25)])
    results = index.lexical_search("annual leave", EMPLOYEE, limit=4)
    assert [item.chunk.chunk_id for item in results] == ["leave"]
    assert results[0].score == 4.25


def test_a_cross_tenant_hit_is_dropped_even_if_the_store_returns_it() -> None:
    index, _ = build_index([hit_for(chunk("other", tenant="tenant-b"))])
    assert index.lexical_search("annual leave", EMPLOYEE, limit=4) == []


def test_a_hit_requiring_an_unheld_label_is_dropped() -> None:
    index, _ = build_index([hit_for(chunk("forecast", labels=frozenset({"finance"})))])
    assert index.lexical_search("annual leave", EMPLOYEE, limit=4) == []


def test_a_non_retrievable_hit_is_dropped() -> None:
    index, _ = build_index([hit_for(chunk("parent", retrievable=False))])
    assert index.lexical_search("annual leave", EMPLOYEE, limit=4) == []


def test_a_hit_without_a_source_is_skipped() -> None:
    index, _ = build_index([{"_score": 1.0}])
    assert index.lexical_search("annual leave", EMPLOYEE, limit=4) == []


def test_a_hit_without_a_score_defaults_to_zero() -> None:
    index, _ = build_index([{"_source": document_for(chunk("leave"))}])
    assert index.lexical_search("annual leave", EMPLOYEE, limit=4)[0].score == 0.0


def test_results_stop_at_the_limit_even_when_the_store_returns_more() -> None:
    hits = [hit_for(chunk(f"leave-{number}")) for number in range(5)]
    index, _ = build_index(hits)
    assert len(index.lexical_search("annual leave", EMPLOYEE, limit=2)) == 2


# Write path --------------------------------------------------------------------


def test_the_index_is_created_with_an_explicit_mapping() -> None:
    index, client = build_index()
    index.ensure_index()
    assert client.indices.created == ["opensearch:tenant-a:abc123"]


def test_an_existing_index_is_left_alone() -> None:
    index, client = build_index()
    client.indices.present.add("opensearch:tenant-a:abc123")
    index.ensure_index()
    assert client.indices.created == []


def test_upsert_sends_one_action_and_one_document_per_chunk() -> None:
    index, client = build_index()
    assert index.upsert([chunk("a"), chunk("b")]) == 2
    body = client.bulk_bodies[0]
    assert len(body) == 4
    assert body[0]["index"]["_id"] == "a"


def test_the_indexed_document_carries_the_required_label_count() -> None:
    """The subset filter reads this field, so writing it is part of the write contract."""
    document = document_for(chunk("forecast", labels=frozenset({"finance", "management"})))
    assert document["required_label_count"] == 2
    assert document["required_labels"] == ["finance", "management"]


def test_a_public_chunk_has_a_required_label_count_of_zero() -> None:
    assert document_for(chunk("leave"))["required_label_count"] == 0


def test_upserting_nothing_never_calls_the_store() -> None:
    index, client = build_index()
    assert index.upsert([]) == 0
    assert client.bulk_bodies == []


def test_drop_removes_the_index_when_it_exists() -> None:
    index, client = build_index()
    client.indices.present.add("opensearch:tenant-a:abc123")
    index.drop()
    assert client.indices.deleted == ["opensearch:tenant-a:abc123"]


def test_dropping_a_missing_index_is_a_no_op() -> None:
    index, client = build_index()
    index.drop()
    assert client.indices.deleted == []

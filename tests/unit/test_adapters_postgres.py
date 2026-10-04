"""PostgreSQL catalog behavior against a fake connection.

The lifecycle rules are inherited from `IndexCatalog` and already covered there; these tests are
about the write-through and the hydration. tests/integration/test_services.py proves the round
trip against a real PostgreSQL.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_platform.adapters.payload import to_payload
from rag_platform.adapters.postgres import (
    SCHEMA,
    PostgresFeedbackLog,
    PostgresIndexCatalog,
)
from rag_platform.catalog import IndexActivationError
from rag_platform.feedback import FeedbackRequest, FeedbackReview, UnknownFeedbackError
from rag_platform.models import (
    ChunkingConfig,
    DocumentChunk,
    IndexVersion,
    RetentionPolicy,
    SourceVersion,
)


@dataclass
class FakeCursor:
    rows: list[tuple[Any, ...]]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


@dataclass
class FakeConnection:
    results: dict[str, list[tuple[Any, ...]]] = field(default_factory=dict)
    statements: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    commits: int = 0

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> FakeCursor:
        normalized = " ".join(query.split())
        self.statements.append((normalized, params))
        for fragment, rows in self.results.items():
            if fragment in normalized:
                return FakeCursor(rows)
        return FakeCursor([])

    def commit(self) -> None:
        self.commits += 1

    def touching(self, table: str) -> list[tuple[str, tuple[Any, ...]]]:
        return [entry for entry in self.statements if table in entry[0]]


SOURCE_VERSION = SourceVersion(
    source_version_id="sv-handbook-0001",
    source_id="handbook",
    tenant_id="tenant-a",
    owner="people-operations",
    access_labels=frozenset({"public"}),
    source_uri="https://example.test/handbook",
    source_title="Handbook",
    content_hash="0001",
    parser_revision="parser-v1",
    transformation_revision="transform-v1",
    retention=RetentionPolicy(),
    document_type="markdown",
    language="en",
)


def index_version(identifier: str) -> IndexVersion:
    return IndexVersion(
        index_version_id=identifier,
        tenant_id="tenant-a",
        source_version_ids=("sv-handbook-0001",),
        chunking=ChunkingConfig(),
        embedding_revision="embed-v1",
        analyzer_revision="analyzer-v1",
        graph_extractor_revision="graph-v1",
        physical_dense_index=f"qdrant:{identifier}",
        physical_sparse_index=f"opensearch:{identifier}",
        physical_graph_index=f"neo4j:{identifier}",
        chunk_count=1,
    )


def chunk(identifier: str, version: str = "iv-1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text="Annual leave requests use the HR portal.",
        source_uri="https://example.test/handbook",
        source_title="Handbook",
        index_version=version,
        source_version_id="sv-handbook-0001",
    )


def catalog() -> tuple[PostgresIndexCatalog, FakeConnection]:
    connection = FakeConnection()
    return PostgresIndexCatalog(connection), connection  # type: ignore[arg-type]


def activated(target: PostgresIndexCatalog, identifier: str) -> None:
    target.stage_index_version(index_version(identifier), [chunk("c1", identifier)])
    target.mark_validated(identifier)
    target.activate(identifier)


def test_the_schema_is_created_idempotently() -> None:
    target, connection = catalog()
    target.ensure_schema()
    assert len(connection.statements) == len(SCHEMA)
    assert all("IF NOT EXISTS" in statement for statement, _ in connection.statements)
    assert connection.commits == 1


def test_a_source_version_is_written_through() -> None:
    target, connection = catalog()
    target.register_source_version(SOURCE_VERSION)

    [(statement, params)] = connection.touching("rag_source_versions")
    assert "ON CONFLICT (source_version_id) DO NOTHING" in statement
    assert params[:4] == ("sv-handbook-0001", "handbook", "tenant-a", "0001")
    assert connection.commits == 1


def test_staging_writes_the_index_version_and_its_chunks_in_order() -> None:
    target, connection = catalog()
    target.stage_index_version(index_version("iv-1"), [chunk("c1"), chunk("c2")])

    inserts = [
        params for statement, params in connection.touching("rag_chunks") if "INSERT" in statement
    ]
    assert [(params[1], params[2]) for params in inserts] == [(0, "c1"), (1, "c2")]
    assert connection.touching("rag_index_versions")[0][1][2] == "building"


def test_every_lifecycle_transition_persists_the_new_status() -> None:
    target, connection = catalog()
    activated(target, "iv-1")

    statuses = [params[2] for _, params in connection.touching("rag_index_versions")]
    assert statuses == ["building", "validated", "active"]


def test_activation_persists_the_active_pointer_and_history() -> None:
    target, connection = catalog()
    activated(target, "iv-1")
    activated(target, "iv-2")

    _, params = connection.touching("rag_activations")[-1]
    assert params == ("tenant-a", "iv-2", '["iv-1"]')


def test_rollback_persists_the_restored_pointer() -> None:
    target, connection = catalog()
    activated(target, "iv-1")
    activated(target, "iv-2")
    target.rollback("tenant-a")

    _, params = connection.touching("rag_activations")[-1]
    assert params == ("tenant-a", "iv-1", "[]")
    assert target.active_index_version("tenant-a").index_version_id == "iv-1"  # type: ignore[union-attr]


def test_the_inherited_activation_gate_still_applies() -> None:
    target, _ = catalog()
    target.stage_index_version(index_version("iv-1"), [chunk("c1")])
    with pytest.raises(IndexActivationError):
        target.activate("iv-1")


def test_load_rebuilds_the_catalog_from_stored_rows() -> None:
    active = index_version("iv-2").with_status("active")
    retired = index_version("iv-1").with_status("retired")
    connection = FakeConnection(
        results={
            "FROM rag_source_versions": [(SOURCE_VERSION.model_dump(mode="json"),)],
            "FROM rag_index_versions": [
                (retired.model_dump(mode="json"),),
                (active.model_dump_json(),),
            ],
            "FROM rag_chunks": [("iv-2", to_payload(chunk("c1", "iv-2")))],
            "FROM rag_activations": [("tenant-a", "iv-2", ["iv-1"])],
        }
    )
    target = PostgresIndexCatalog(connection)  # type: ignore[arg-type]
    target.load()

    assert target.source_version("sv-handbook-0001") == SOURCE_VERSION
    assert target.active_index_version("tenant-a") == active
    assert [item.chunk_id for item in target.active_chunks("tenant-a")] == ["c1"]
    assert target.chunks_for("iv-1") == ()
    # History survived the restart, so rollback needs no reingestion.
    assert target.rollback("tenant-a").index_version_id == "iv-1"


def test_load_tolerates_a_tenant_with_no_active_version() -> None:
    connection = FakeConnection(results={"FROM rag_activations": [("tenant-a", None, [])]})
    target = PostgresIndexCatalog(connection)  # type: ignore[arg-type]
    target.load()
    assert target.active_index_version("tenant-a") is None


# Replicas ----------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def stored_rows(active_id: str) -> dict[str, list[tuple[Any, ...]]]:
    active = index_version(active_id).with_status("active")
    return {
        "SELECT tenant_id, active_index_version_id FROM rag_activations": [("tenant-a", active_id)],
        "FROM rag_source_versions": [(SOURCE_VERSION.model_dump(mode="json"),)],
        "FROM rag_index_versions": [(active.model_dump(mode="json"),)],
        "FROM rag_chunks": [(active_id, to_payload(chunk("c1", active_id)))],
        "SELECT tenant_id, active_index_version_id, history": [("tenant-a", active_id, [])],
    }


def test_a_replica_picks_up_an_activation_made_elsewhere_after_the_sync_interval() -> None:
    clock = Clock()
    connection = FakeConnection()
    target = PostgresIndexCatalog(connection, sync_interval_seconds=5, clock=clock)  # type: ignore[arg-type]
    assert target.active_index_version("tenant-a") is None

    # Another replica activates iv-2; this one only sees the database.
    connection.results.update(stored_rows("iv-2"))
    clock.now = 4.0
    assert target.active_index_version("tenant-a") is None
    clock.now = 5.0
    active = target.active_index_version("tenant-a")

    assert active is not None and active.index_version_id == "iv-2"
    assert [item.chunk_id for item in target.active_chunks("tenant-a")] == ["c1"]


def test_an_unchanged_pointer_does_not_reload_the_catalog() -> None:
    clock = Clock()
    connection = FakeConnection()
    target = PostgresIndexCatalog(connection, sync_interval_seconds=5, clock=clock)  # type: ignore[arg-type]

    clock.now = 10.0
    assert target.sync() is False
    assert not connection.touching("rag_index_versions")


def test_a_rollback_made_elsewhere_is_followed() -> None:
    connection = FakeConnection(results=stored_rows("iv-2"))
    target = PostgresIndexCatalog(connection)  # type: ignore[arg-type]
    target.load()

    connection.results = stored_rows("iv-1")
    assert target.sync() is True
    assert target.active_index_version("tenant-a").index_version_id == "iv-1"  # type: ignore[union-attr]


def test_without_an_interval_the_catalog_never_syncs_on_read() -> None:
    connection = FakeConnection()
    target = PostgresIndexCatalog(connection)  # type: ignore[arg-type]
    connection.results.update(stored_rows("iv-2"))
    assert target.active_index_version("tenant-a") is None


# Feedback ----------------------------------------------------------------------

FEEDBACK = FeedbackRequest(question="How do I request annual leave?", rating="unhelpful")


def feedback_log(connection: FakeConnection) -> PostgresFeedbackLog:
    return PostgresFeedbackLog(connection=connection, id_factory=lambda: "fb-0001")  # type: ignore[arg-type]


def test_feedback_is_written_through_and_committed() -> None:
    connection = FakeConnection()
    log = feedback_log(connection)
    log.ensure_schema()
    submitted = log.submit("tenant-a", frozenset({"public"}), FEEDBACK)

    insert = next(entry for entry in connection.touching("rag_feedback") if "INSERT" in entry[0])
    assert insert[1][:3] == ("fb-0001", "tenant-a", "submitted")
    assert submitted.feedback_id == "fb-0001"
    assert connection.commits == 2


def test_feedback_is_read_back_from_the_database_not_from_memory() -> None:
    writer = FakeConnection()
    submitted = feedback_log(writer).submit("tenant-a", frozenset({"public"}), FEEDBACK)

    # A different replica: a new log on a connection that returns the stored row.
    body = submitted.model_dump(mode="json")
    reader = FakeConnection(results={"FROM rag_feedback": [(body,)]})
    log = feedback_log(reader)

    assert log.get("tenant-a", "fb-0001") == submitted
    assert log.items("tenant-a") == [submitted]
    reviewed = log.review(
        "tenant-a", "fb-0001", FeedbackReview(reviewer="r.osei", decision="accepted")
    )
    assert reviewed.status == "accepted"
    update = [entry for entry in reader.touching("rag_feedback") if "INSERT" in entry[0]][-1]
    assert update[1][2] == "accepted"


def test_stored_feedback_stays_tenant_scoped() -> None:
    body = feedback_log(FakeConnection()).submit("tenant-a", frozenset({"public"}), FEEDBACK)
    reader = FakeConnection(results={"WHERE feedback_id": [(body.model_dump_json(),)]})
    with pytest.raises(UnknownFeedbackError):
        feedback_log(reader).get("tenant-b", "fb-0001")


def test_missing_feedback_is_unknown_and_a_log_needs_a_connection() -> None:
    with pytest.raises(UnknownFeedbackError):
        feedback_log(FakeConnection()).get("tenant-a", "fb-missing")
    with pytest.raises(RuntimeError, match="needs a connection"):
        PostgresFeedbackLog().ensure_schema()

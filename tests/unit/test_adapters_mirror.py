"""Mirroring an index version into real stores, and what happens when a store refuses."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from rag_platform.adapters.embeddings import HashingEmbedder
from rag_platform.adapters.mirror import (
    MirrorReport,
    OpenSearchMirror,
    QdrantMirror,
    readers_for,
)
from rag_platform.catalog import IndexCatalog
from rag_platform.ingestion import IngestionPipeline, IngestionValidationError
from rag_platform.models import (
    ChunkingConfig,
    DocumentChunk,
    IndexVersion,
    SourceRegistration,
)

HANDBOOK = """# ACME Employee Handbook

## Annual leave

Annual leave requests must be submitted in the HR portal at least five working days before the
first day of leave. Managers approve or decline a request within two working days.
"""

REGISTRATION = SourceRegistration(
    source_id="acme-handbook",
    tenant_id="tenant-acme",
    owner="people-operations",
    access_labels=frozenset({"public"}),
    source_uri="https://knowledge.example/acme/handbook",
    source_title="ACME Employee Handbook",
)


@dataclass
class RecordingMirror:
    name: str = "qdrant:recorded"
    calls: list[tuple[str, int]] = field(default_factory=list)
    seen_chunk_ids: list[str] = field(default_factory=list)

    def mirror(
        self, index_version: IndexVersion, chunks: Sequence[DocumentChunk]
    ) -> MirrorReport:
        retrievable = [chunk for chunk in chunks if chunk.retrievable]
        self.calls.append((index_version.index_version_id, len(retrievable)))
        self.seen_chunk_ids.extend(chunk.chunk_id for chunk in chunks)
        return MirrorReport(physical_index=self.name, chunks_written=len(retrievable))


@dataclass
class FailingMirror:
    def mirror(
        self, index_version: IndexVersion, chunks: Sequence[DocumentChunk]
    ) -> MirrorReport:
        raise RuntimeError("store unreachable")


def test_a_validated_index_version_is_mirrored_before_activation() -> None:
    catalog = IndexCatalog()
    mirror = RecordingMirror()
    pipeline = IngestionPipeline(catalog=catalog, mirrors=(mirror,))

    result = pipeline.ingest(REGISTRATION, HANDBOOK)

    assert mirror.calls
    assert mirror.calls[0][0] == result.index_version.index_version_id
    assert result.index_version.status == "active"


def test_the_mirror_report_is_recorded_in_the_validation_checks() -> None:
    catalog = IndexCatalog()
    pipeline = IngestionPipeline(catalog=catalog, mirrors=(RecordingMirror(),))

    result = pipeline.ingest(REGISTRATION, HANDBOOK)

    mirrored = [
        check for check in result.report.validation_checks if check.startswith("mirrored:")
    ]
    assert mirrored == ["mirrored:qdrant:recorded:1"]


def test_every_configured_store_is_mirrored() -> None:
    catalog = IndexCatalog()
    first, second = RecordingMirror("qdrant:one"), RecordingMirror("opensearch:two")
    pipeline = IngestionPipeline(catalog=catalog, mirrors=(first, second))

    pipeline.ingest(REGISTRATION, HANDBOOK)

    assert first.calls and second.calls


def test_a_store_that_refuses_the_write_blocks_activation() -> None:
    """A half-built index must never serve traffic."""
    catalog = IndexCatalog()
    pipeline = IngestionPipeline(catalog=catalog, mirrors=(FailingMirror(),))

    with pytest.raises(IngestionValidationError, match="mirroring"):
        pipeline.ingest(REGISTRATION, HANDBOOK)

    assert catalog.active_index_version("tenant-acme") is None
    assert all(version.status == "retired" for version in catalog.index_versions())


def test_a_failing_store_leaves_the_previous_index_version_serving() -> None:
    catalog = IndexCatalog()
    healthy = IngestionPipeline(catalog=catalog, mirrors=(RecordingMirror(),))
    first = healthy.ingest(REGISTRATION, HANDBOOK)

    broken = IngestionPipeline(catalog=catalog, mirrors=(FailingMirror(),))
    with pytest.raises(IngestionValidationError):
        broken.ingest(REGISTRATION, HANDBOOK + "\n## Expenses\n\nClaims are monthly.\n")

    active = catalog.active_index_version("tenant-acme")
    assert active is not None
    assert active.index_version_id == first.index_version.index_version_id


def test_ingestion_without_mirrors_is_unchanged() -> None:
    catalog = IndexCatalog()
    result = IngestionPipeline(catalog=catalog).ingest(REGISTRATION, HANDBOOK)

    assert result.index_version.status == "active"
    assert not any(
        check.startswith("mirrored:") for check in result.report.validation_checks
    )


def chunk(identifier: str, *, retrievable: bool = True) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id="tenant-acme",
        access_labels=frozenset({"public"}),
        text="Annual leave requests use the HR portal.",
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="iv-tenant-acme-abc123",
        source_version_id=f"sv-{identifier}",
        retrievable=retrievable,
    )


INDEX_VERSION = IndexVersion(
    index_version_id="iv-tenant-acme-abc123",
    tenant_id="tenant-acme",
    source_version_ids=("sv-handbook",),
    chunking=ChunkingConfig(),
    embedding_revision="embed-v1",
    analyzer_revision="analyzer-v1",
    graph_extractor_revision="graph-v1",
    physical_dense_index="qdrant:tenant-acme:abc123",
    physical_sparse_index="opensearch:tenant-acme:abc123",
    physical_graph_index="neo4j:tenant-acme:abc123",
    chunk_count=2,
)


@dataclass
class FakeQdrant:
    collections: set[str] = field(default_factory=set)
    points: list[object] = field(default_factory=list)

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.collections.add(collection_name)

    def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: object
    ) -> None:
        return None

    def upsert(self, collection_name: str, points: list[object]) -> None:
        self.points.extend(points)


@dataclass
class FakeOpenSearchIndices:
    present: set[str] = field(default_factory=set)

    def exists(self, index: str) -> bool:
        return index in self.present

    def create(self, index: str, body: dict[str, object]) -> None:
        self.present.add(index)


@dataclass
class FakeOpenSearch:
    indices: FakeOpenSearchIndices = field(default_factory=FakeOpenSearchIndices)
    bulk_bodies: list[list[dict[str, object]]] = field(default_factory=list)

    def bulk(self, body: list[dict[str, object]], refresh: bool = False) -> dict[str, object]:
        self.bulk_bodies.append(body)
        return {"errors": False}


def test_the_qdrant_mirror_writes_into_the_collection_the_index_version_names() -> None:
    client = FakeQdrant()
    mirror = QdrantMirror(client=client, embedder=HashingEmbedder(dimensions=32))  # type: ignore[arg-type]

    report = mirror.mirror(INDEX_VERSION, [chunk("a"), chunk("parent", retrievable=False)])

    assert report.physical_index == "qdrant:tenant-acme:abc123"
    assert report.chunks_written == 1
    assert client.collections == {"qdrant:tenant-acme:abc123"}
    assert len(client.points) == 1


def test_the_opensearch_mirror_writes_into_the_index_the_index_version_names() -> None:
    client = FakeOpenSearch()
    mirror = OpenSearchMirror(client=client)  # type: ignore[arg-type]

    report = mirror.mirror(INDEX_VERSION, [chunk("a"), chunk("b")])

    assert report.physical_index == "opensearch:tenant-acme:abc123"
    assert report.chunks_written == 2
    assert client.indices.present == {"opensearch:tenant-acme:abc123"}


def test_the_readers_point_at_the_same_physical_indexes_the_mirrors_wrote() -> None:
    dense, sparse = readers_for(
        INDEX_VERSION,
        qdrant_client=FakeQdrant(),  # type: ignore[arg-type]
        opensearch_client=FakeOpenSearch(),  # type: ignore[arg-type]
    )
    assert dense is not None and dense.collection == INDEX_VERSION.physical_dense_index
    assert sparse is not None and sparse.index == INDEX_VERSION.physical_sparse_index


def test_an_unconfigured_store_has_no_reader() -> None:
    dense, sparse = readers_for(INDEX_VERSION)
    assert dense is None
    assert sparse is None


def test_the_mirror_receives_the_parent_chunks_but_reports_only_retrievable_ones() -> None:
    """Parents must reach the store's caller for context expansion decisions, not the index."""
    catalog = IndexCatalog()
    mirror = RecordingMirror()
    long_section = "## Annual leave\n\n" + ("Leave is requested in the portal. " * 60)
    pipeline = IngestionPipeline(catalog=catalog, mirrors=(mirror,))

    pipeline.ingest(REGISTRATION, f"# Handbook\n\n{long_section}")

    chunks = catalog.chunks_for(catalog.index_versions()[0].index_version_id)
    assert any(not chunk.retrievable for chunk in chunks)
    assert mirror.calls[0][1] == sum(1 for chunk in chunks if chunk.retrievable)

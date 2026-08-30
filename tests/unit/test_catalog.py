import pytest

from rag_platform.catalog import (
    IndexActivationError,
    IndexCatalog,
    UnknownIndexVersionError,
)
from rag_platform.models import (
    ChunkingConfig,
    DocumentChunk,
    IndexVersion,
    SourceVersion,
)


def index_version(identifier: str, *, tenant: str = "tenant-a") -> IndexVersion:
    return IndexVersion(
        index_version_id=identifier,
        tenant_id=tenant,
        source_version_ids=("sv-1",),
        chunking=ChunkingConfig(),
        embedding_revision="embed-v1",
        analyzer_revision="analyzer-v1",
        graph_extractor_revision="graph-v1",
        physical_dense_index="qdrant:a",
        physical_sparse_index="opensearch:a",
        physical_graph_index="neo4j:a",
        chunk_count=1,
    )


def chunk(identifier: str, *, index_version_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text="Annual leave uses the HR portal.",
        source_uri="https://example.test/leave",
        source_title="Leave policy",
        index_version=index_version_id,
        source_version_id="sv-1",
    )


@pytest.fixture
def catalog() -> IndexCatalog:
    return IndexCatalog()


def test_staged_index_versions_are_invisible_until_activated(catalog: IndexCatalog) -> None:
    catalog.stage_index_version(index_version("iv-1"), [chunk("c1", index_version_id="iv-1")])
    assert catalog.active_index_version("tenant-a") is None
    assert catalog.active_chunks("tenant-a") == ()

    catalog.mark_validated("iv-1")
    activated = catalog.activate("iv-1")
    assert activated.status == "active"
    assert [item.chunk_id for item in catalog.active_chunks("tenant-a")] == ["c1"]


def test_activation_requires_validation(catalog: IndexCatalog) -> None:
    catalog.stage_index_version(index_version("iv-1"), [chunk("c1", index_version_id="iv-1")])
    with pytest.raises(IndexActivationError, match="validated"):
        catalog.activate("iv-1")


def test_rollback_restores_the_previous_active_index_without_reingestion(
    catalog: IndexCatalog,
) -> None:
    for identifier in ("iv-1", "iv-2"):
        catalog.stage_index_version(
            index_version(identifier), [chunk(f"c-{identifier}", index_version_id=identifier)]
        )
        catalog.mark_validated(identifier)
        catalog.activate(identifier)

    restored = catalog.rollback("tenant-a")
    assert restored.index_version_id == "iv-1"
    assert restored.status == "active"
    assert catalog.index_version("iv-2").status == "retired"
    assert [item.chunk_id for item in catalog.active_chunks("tenant-a")] == ["c-iv-1"]


def test_rollback_without_history_is_rejected(catalog: IndexCatalog) -> None:
    with pytest.raises(IndexActivationError, match="no previous index version"):
        catalog.rollback("tenant-a")


def test_activating_the_same_version_twice_does_not_grow_history(catalog: IndexCatalog) -> None:
    catalog.stage_index_version(index_version("iv-1"), [chunk("c1", index_version_id="iv-1")])
    catalog.mark_validated("iv-1")
    catalog.activate("iv-1")
    catalog.mark_validated("iv-1")
    catalog.activate("iv-1")
    with pytest.raises(IndexActivationError):
        catalog.rollback("tenant-a")


def test_tenants_activate_independently(catalog: IndexCatalog) -> None:
    catalog.stage_index_version(index_version("iv-a"), [chunk("c1", index_version_id="iv-a")])
    catalog.mark_validated("iv-a")
    catalog.activate("iv-a")
    other = index_version("iv-b", tenant="tenant-b")
    catalog.stage_index_version(other, [])
    catalog.mark_validated("iv-b")
    catalog.activate("iv-b")

    assert catalog.active_index_version("tenant-a").index_version_id == "iv-a"
    assert catalog.active_chunks("tenant-b") == ()


def test_chunk_lookup_is_scoped_to_the_active_index(catalog: IndexCatalog) -> None:
    catalog.stage_index_version(index_version("iv-1"), [chunk("c1", index_version_id="iv-1")])
    catalog.mark_validated("iv-1")
    catalog.activate("iv-1")
    assert catalog.chunk("tenant-a", "c1") is not None
    assert catalog.chunk("tenant-a", "missing") is None


def test_unknown_index_versions_are_reported(catalog: IndexCatalog) -> None:
    with pytest.raises(UnknownIndexVersionError):
        catalog.chunks_for("iv-missing")


def test_registering_the_same_source_version_twice_is_idempotent(
    catalog: IndexCatalog, source_version: SourceVersion
) -> None:
    first = catalog.register_source_version(source_version)
    second = catalog.register_source_version(source_version.model_copy(update={"owner": "new"}))
    assert first is second
    assert catalog.find_source_version("acme-handbook", first.content_hash) is first
    assert catalog.find_source_version("acme-handbook", "other-hash") is None

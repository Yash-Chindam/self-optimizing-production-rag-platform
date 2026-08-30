import pytest

from rag_platform.catalog import IndexCatalog
from rag_platform.ingestion import (
    IngestionPipeline,
    IngestionValidationError,
    detect_document_type,
    detect_language,
)
from rag_platform.models import (
    AccessContext,
    ChunkingConfig,
    PiiPolicy,
    SourceRegistration,
)
from rag_platform.repository import CatalogChunkRepository

HANDBOOK = """# ACME Handbook

## Annual leave

Annual leave requests must be submitted in the HR portal at least five working days before the
first day of leave.
"""

UPDATED_HANDBOOK = """# ACME Handbook

## Annual leave

Annual leave requests must be submitted in the HR portal at least ten working days before the
first day of leave.
"""

FORECAST = """# ACME Forecast

## Revenue

The confidential forecast projects twelve percent revenue growth.
"""


def registration(source_id: str = "acme-handbook", **overrides: object) -> SourceRegistration:
    values: dict[str, object] = {
        "source_id": source_id,
        "tenant_id": "tenant-a",
        "owner": "people-operations",
        "access_labels": frozenset({"public"}),
        "source_uri": f"https://example.test/{source_id}",
        "source_title": source_id.replace("-", " ").title(),
    }
    values.update(overrides)
    return SourceRegistration(**values)  # type: ignore[arg-type]


@pytest.fixture
def catalog() -> IndexCatalog:
    return IndexCatalog()


@pytest.fixture
def pipeline(catalog: IndexCatalog) -> IngestionPipeline:
    return IngestionPipeline(catalog=catalog)


def test_ingestion_activates_a_validated_index_version(pipeline: IngestionPipeline) -> None:
    result = pipeline.ingest(registration(), HANDBOOK)

    assert result.index_version.status == "active"
    assert result.index_version.chunk_count == result.report.chunk_count
    assert result.source_version.source_version_id in result.index_version.source_version_ids
    assert "retrieval_smoke_passed" in result.report.validation_checks
    assert result.index_version.physical_dense_index.startswith("qdrant:")


def test_source_version_records_lineage_and_detected_properties(
    pipeline: IngestionPipeline,
) -> None:
    source_version = pipeline.ingest(registration(), HANDBOOK).source_version

    assert source_version.content_hash
    assert source_version.parser_revision == "parser-v1"
    assert source_version.transformation_revision == "transform-v1"
    assert source_version.document_type == "markdown"
    assert source_version.language == "en"


def test_reingesting_identical_content_is_a_no_operation(
    pipeline: IngestionPipeline, catalog: IndexCatalog
) -> None:
    first = pipeline.ingest(registration(), HANDBOOK)
    second = pipeline.ingest(registration(), HANDBOOK)

    assert second.report.reused_existing_source_version
    assert second.index_version.index_version_id == first.index_version.index_version_id
    assert len(catalog.index_versions()) == 1


def test_updated_content_creates_a_new_index_version_and_replaces_old_chunks(
    pipeline: IngestionPipeline, catalog: IndexCatalog
) -> None:
    first = pipeline.ingest(registration(), HANDBOOK)
    second = pipeline.ingest(registration(), UPDATED_HANDBOOK)

    assert second.index_version.index_version_id != first.index_version.index_version_id
    assert catalog.index_version(first.index_version.index_version_id).status == "retired"
    texts = " ".join(chunk.text for chunk in catalog.active_chunks("tenant-a"))
    assert "ten working days" in texts
    assert "five working days" not in texts


def test_a_second_source_is_carried_into_the_new_index_version(
    pipeline: IngestionPipeline, catalog: IndexCatalog
) -> None:
    pipeline.ingest(registration(), HANDBOOK)
    result = pipeline.ingest(registration("acme-forecast"), FORECAST)

    source_ids = {
        catalog.source_version(chunk.source_version_id).source_id
        for chunk in catalog.active_chunks("tenant-a")
    }
    assert source_ids == {"acme-handbook", "acme-forecast"}
    assert len(result.index_version.source_version_ids) == 2


def test_chunks_inherit_the_registered_access_labels(pipeline: IngestionPipeline) -> None:
    pipeline.ingest(registration("acme-forecast", access_labels=frozenset({"finance"})), FORECAST)
    repository = CatalogChunkRepository(pipeline.catalog)
    employee = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))
    analyst = AccessContext(tenant_id="tenant-a", labels=frozenset({"public", "finance"}))

    assert repository.lexical_search("revenue growth", employee, limit=5) == []
    assert repository.lexical_search("revenue growth", analyst, limit=5)


def test_sensitive_values_never_reach_the_index(pipeline: IngestionPipeline) -> None:
    content = "# Contacts\n\n## Support\n\nEmail ada@example.com for leave questions.\n"
    result = pipeline.ingest(registration("acme-contacts"), content)
    texts = " ".join(chunk.text for chunk in pipeline.catalog.active_chunks("tenant-a"))

    assert "ada@example.com" not in texts
    assert "[EMAIL_" in texts
    assert result.report.pii_entities_processed == 1
    assert "pii_pseudonymized" in result.report.validation_checks


def test_pii_can_be_disabled_per_source(pipeline: IngestionPipeline) -> None:
    content = "# Contacts\n\n## Support\n\nEmail ada@example.com for leave questions.\n"
    result = pipeline.ingest(
        registration("acme-contacts", pii_policy=PiiPolicy(mode="off")), content
    )
    texts = " ".join(chunk.text for chunk in pipeline.catalog.active_chunks("tenant-a"))

    assert "ada@example.com" in texts
    assert "pii_pseudonymized" not in result.report.validation_checks


def test_parent_chunks_are_stored_but_not_retrievable(catalog: IndexCatalog) -> None:
    pipeline = IngestionPipeline(
        catalog=catalog, chunking=ChunkingConfig(max_characters=120, overlap_characters=20)
    )
    long_source = "# Handbook\n\n## Leave\n\n" + ("Leave requests use the HR portal. " * 20)
    pipeline.ingest(registration(), long_source)

    chunks = catalog.active_chunks("tenant-a")
    parents = [chunk for chunk in chunks if not chunk.retrievable]
    children = [chunk for chunk in chunks if chunk.parent_chunk_id is not None]
    assert parents and children
    assert {chunk.parent_chunk_id for chunk in children} == {parents[0].chunk_id}

    repository = CatalogChunkRepository(catalog)
    access = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))
    found = repository.lexical_search("leave requests", access, limit=20)
    assert parents[0].chunk_id not in {result.chunk.chunk_id for result in found}


def test_repeated_identical_windows_are_deduplicated(catalog: IndexCatalog) -> None:
    pipeline = IngestionPipeline(catalog=catalog, chunking=ChunkingConfig(max_characters=200))
    duplicated = "# Notes\n\n## One\n\nSame paragraph text.\n\n## Two\n\nSame paragraph text.\n"
    report = pipeline.ingest(registration("acme-notes", source_type="text"), duplicated).report
    assert report.duplicate_chunks_removed == 1


def test_empty_content_is_rejected_and_no_index_is_activated(
    pipeline: IngestionPipeline, catalog: IndexCatalog
) -> None:
    with pytest.raises(IngestionValidationError, match="retrievable"):
        pipeline.ingest(registration(), "   \n\n  ")
    assert catalog.active_index_version("tenant-a") is None
    assert all(version.status == "retired" for version in catalog.index_versions())


def test_document_type_falls_back_to_text_without_headings() -> None:
    assert detect_document_type("plain sentence", "markdown") == "text"
    assert detect_document_type("# Title", "markdown") == "markdown"
    assert detect_document_type("# Title", "text") == "text"


def test_language_detection_uses_function_word_profiles() -> None:
    assert detect_language("the leave policy of the company") == "en"
    assert detect_language("le document doit pour les employes") == "fr"
    assert detect_language("!!!") == "en"


def test_validation_rejects_a_chunk_from_another_tenant(
    pipeline: IngestionPipeline, catalog: IndexCatalog
) -> None:
    result = pipeline.ingest(registration(), HANDBOOK)
    index_version = result.index_version
    chunks = catalog.chunks_for(index_version.index_version_id)
    foreign = chunks[0].model_copy(update={"tenant_id": "tenant-b"})

    with pytest.raises(IngestionValidationError, match="crosses tenants"):
        pipeline.validate(index_version, (foreign,), PiiPolicy())
    assert catalog.index_version(index_version.index_version_id).status == "retired"


def test_validation_rejects_a_chunk_without_access_labels(
    pipeline: IngestionPipeline, catalog: IndexCatalog
) -> None:
    result = pipeline.ingest(registration(), HANDBOOK)
    unlabelled = catalog.chunks_for(result.index_version.index_version_id)[0].model_copy(
        update={"access_labels": frozenset()}
    )
    with pytest.raises(IngestionValidationError, match="access labels"):
        pipeline.validate(result.index_version, (unlabelled,), PiiPolicy())


def test_validation_rejects_an_index_that_cannot_be_retrieved(
    pipeline: IngestionPipeline, catalog: IndexCatalog
) -> None:
    result = pipeline.ingest(registration(), HANDBOOK)
    unsearchable = catalog.chunks_for(result.index_version.index_version_id)[0].model_copy(
        update={"text": "!!!"}
    )
    with pytest.raises(IngestionValidationError, match="smoke check"):
        pipeline.validate(result.index_version, (unsearchable,), PiiPolicy())


def test_content_restored_by_rollback_can_be_reingested(
    pipeline: IngestionPipeline, catalog: IndexCatalog
) -> None:
    first = pipeline.ingest(registration(), HANDBOOK)
    pipeline.ingest(registration(), UPDATED_HANDBOOK)
    catalog.rollback("tenant-a")

    reingested = pipeline.ingest(registration(), UPDATED_HANDBOOK)
    assert reingested.report.reused_existing_source_version is False
    assert reingested.index_version.index_version_id != first.index_version.index_version_id

from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_platform.adapters.originals import (
    InMemoryOriginalStore,
    MinioOriginalStore,
    original_key,
)
from rag_platform.catalog import IndexCatalog
from rag_platform.ingestion import IngestionPipeline, content_hash
from rag_platform.models import RetentionPolicy, SourceRegistration

HANDBOOK = "# Handbook\n\n## Annual leave\n\nAnnual leave requests use the HR portal.\n"


def registration(*, delete_original: bool = False) -> SourceRegistration:
    return SourceRegistration(
        source_id="handbook",
        tenant_id="tenant-a",
        owner="people-operations",
        source_uri="https://example.test/handbook",
        source_title="Handbook",
        retention=RetentionPolicy(delete_original_after_indexing=delete_original),
    )


def test_the_key_is_a_function_of_tenant_source_and_content() -> None:
    assert original_key("tenant-a", "handbook", "abc") == "tenant-a/handbook/abc"


def test_the_in_memory_store_round_trips_content() -> None:
    store = InMemoryOriginalStore()
    key = store.put("tenant-a", "handbook", "abc", "content")
    assert store.get(key) == "content"


def test_an_original_is_never_overwritten() -> None:
    store = InMemoryOriginalStore()
    key = store.put("tenant-a", "handbook", "abc", "first")
    store.put("tenant-a", "handbook", "abc", "second")
    assert store.get(key) == "first"


def test_ingestion_stores_the_original_under_its_content_hash() -> None:
    store = InMemoryOriginalStore()
    pipeline = IngestionPipeline(catalog=IndexCatalog(), originals=store)

    result = pipeline.ingest(registration(), HANDBOOK)

    key = original_key("tenant-a", "handbook", content_hash(HANDBOOK))
    assert store.get(key) == HANDBOOK
    assert f"original_stored:{key}" in result.report.validation_checks


def test_the_stored_original_is_the_unredacted_source() -> None:
    """Pseudonymization applies to what is indexed; the original stays immutable."""
    store = InMemoryOriginalStore()
    content = HANDBOOK + "\nContact jane.doe@example.com for help.\n"
    IngestionPipeline(catalog=IndexCatalog(), originals=store).ingest(registration(), content)
    assert "jane.doe@example.com" in next(iter(store.objects.values()))


def test_the_retention_policy_can_delete_the_original_after_indexing() -> None:
    store = InMemoryOriginalStore()
    pipeline = IngestionPipeline(catalog=IndexCatalog(), originals=store)

    result = pipeline.ingest(registration(delete_original=True), HANDBOOK)

    assert store.objects == {}
    assert "original_deleted_by_retention_policy" in result.report.validation_checks


def test_ingestion_without_an_original_store_is_unchanged() -> None:
    result = IngestionPipeline(catalog=IndexCatalog()).ingest(registration(), HANDBOOK)
    assert not any("original" in check for check in result.report.validation_checks)


# MinIO -------------------------------------------------------------------------


@dataclass
class FakeResponse:
    data: bytes
    closed: bool = False
    released: bool = False

    def read(self) -> bytes:
        return self.data

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


@dataclass
class FakeMinio:
    buckets: set[str] = field(default_factory=set)
    objects: dict[tuple[str, str], bytes] = field(default_factory=dict)
    puts: int = 0
    responses: list[FakeResponse] = field(default_factory=list)
    stat_error_code: str = "NoSuchKey"

    def bucket_exists(self, bucket: str) -> bool:
        return bucket in self.buckets

    def make_bucket(self, bucket: str) -> None:
        self.buckets.add(bucket)

    def stat_object(self, bucket: str, key: str) -> None:
        if (bucket, key) not in self.objects:
            from minio.error import S3Error

            raise S3Error(None, self.stat_error_code, "missing", key, "request", "host")  # type: ignore[arg-type]

    def put_object(
        self, bucket: str, key: str, data: Any, length: int, content_type: str
    ) -> None:
        self.puts += 1
        self.objects[(bucket, key)] = data.read()

    def get_object(self, bucket: str, key: str) -> FakeResponse:
        response = FakeResponse(self.objects[(bucket, key)])
        self.responses.append(response)
        return response

    def remove_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)


def minio_store() -> tuple[MinioOriginalStore, FakeMinio]:
    pytest.importorskip("minio")
    client = FakeMinio()
    return MinioOriginalStore(client=client), client  # type: ignore[arg-type]


def test_the_bucket_is_created_once() -> None:
    store, client = minio_store()
    store.ensure_bucket()
    store.ensure_bucket()
    assert client.buckets == {"rag-originals"}


def test_minio_round_trips_utf8_content_and_releases_the_connection() -> None:
    store, client = minio_store()
    key = store.put("tenant-a", "handbook", "abc", "Übersicht — annual leave")

    assert store.get(key) == "Übersicht — annual leave"
    assert client.responses[0].closed and client.responses[0].released


def test_minio_never_rewrites_an_existing_original() -> None:
    store, client = minio_store()
    store.put("tenant-a", "handbook", "abc", "first")
    store.put("tenant-a", "handbook", "abc", "second")
    assert client.puts == 1


def test_minio_delete_removes_the_object() -> None:
    store, client = minio_store()
    key = store.put("tenant-a", "handbook", "abc", "content")
    store.delete(key)
    assert client.objects == {}


def test_an_unexpected_store_error_is_not_mistaken_for_a_missing_object() -> None:
    from minio.error import S3Error

    store, client = minio_store()
    client.stat_error_code = "AccessDenied"
    with pytest.raises(S3Error):
        store.put("tenant-a", "handbook", "abc", "content")

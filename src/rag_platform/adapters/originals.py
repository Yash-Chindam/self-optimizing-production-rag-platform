"""Immutable originals in object storage (specification section 7, "store immutable original").

An original is addressed by tenant, source and content hash, so the key is a function of the
bytes: writing the same content twice is a no-op, and a changed document is a new object rather
than an overwrite. That is what makes a `SourceVersion` reproducible from its `content_hash`.

`RetentionPolicy.delete_original_after_indexing` is honoured by the ingestion pipeline through
`delete`, which is the only way an original leaves the store.
"""

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from minio import Minio

CONTENT_TYPE = "text/plain; charset=utf-8"


class OriginalStore(Protocol):
    def put(self, tenant_id: str, source_id: str, content_hash: str, content: str) -> str: ...

    def get(self, key: str) -> str: ...

    def delete(self, key: str) -> None: ...


def original_key(tenant_id: str, source_id: str, content_hash: str) -> str:
    return f"{tenant_id}/{source_id}/{content_hash}"


@dataclass(slots=True)
class InMemoryOriginalStore:
    """Deterministic stand-in with the same content-addressed contract."""

    objects: dict[str, str] = field(default_factory=dict)

    def put(self, tenant_id: str, source_id: str, content_hash: str, content: str) -> str:
        key = original_key(tenant_id, source_id, content_hash)
        self.objects.setdefault(key, content)
        return key

    def get(self, key: str) -> str:
        return self.objects[key]

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)


@dataclass(slots=True)
class MinioOriginalStore:
    client: "Minio"
    bucket: str = "rag-originals"

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put(self, tenant_id: str, source_id: str, content_hash: str, content: str) -> str:
        key = original_key(tenant_id, source_id, content_hash)
        if self._exists(key):
            return key
        data = content.encode("utf-8")
        self.client.put_object(
            self.bucket, key, io.BytesIO(data), length=len(data), content_type=CONTENT_TYPE
        )
        return key

    def get(self, key: str) -> str:
        response = self.client.get_object(self.bucket, key)
        try:
            return bytes(response.read()).decode("utf-8")
        finally:
            response.close()
            response.release_conn()

    def delete(self, key: str) -> None:
        self.client.remove_object(self.bucket, key)

    def _exists(self, key: str) -> bool:
        from minio.error import S3Error

        try:
            self.client.stat_object(self.bucket, key)
        except S3Error as error:
            if error.code in {"NoSuchKey", "NoSuchObject"}:
                return False
            raise
        return True

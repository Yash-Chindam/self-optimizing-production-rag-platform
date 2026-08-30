"""Versioned source and index registry.

The catalog is the in-process stand-in for the PostgreSQL metadata store plus the physical
Qdrant, OpenSearch and Neo4j indexes. It exists so index activation is staged rather than
in-place: chunks become visible to retrieval only when their index version is activated, and
the previous configuration can be restored without reingestion.
"""

from collections.abc import Iterable

from rag_platform.models import DocumentChunk, IndexVersion, SourceVersion


class UnknownIndexVersionError(LookupError):
    pass


class IndexActivationError(RuntimeError):
    pass


class IndexCatalog:
    def __init__(self) -> None:
        self._source_versions: dict[str, SourceVersion] = {}
        self._index_versions: dict[str, IndexVersion] = {}
        self._chunks: dict[str, tuple[DocumentChunk, ...]] = {}
        self._active: dict[str, str] = {}
        self._activation_history: dict[str, list[str]] = {}

    # Source versions -------------------------------------------------------
    def register_source_version(self, source_version: SourceVersion) -> SourceVersion:
        existing = self._source_versions.get(source_version.source_version_id)
        if existing is not None:
            return existing
        self._source_versions[source_version.source_version_id] = source_version
        return source_version

    def find_source_version(self, source_id: str, content_hash: str) -> SourceVersion | None:
        for source_version in self._source_versions.values():
            if (
                source_version.source_id == source_id
                and source_version.content_hash == content_hash
            ):
                return source_version
        return None

    def source_version(self, source_version_id: str) -> SourceVersion:
        return self._source_versions[source_version_id]

    # Index versions --------------------------------------------------------
    def stage_index_version(
        self, index_version: IndexVersion, chunks: Iterable[DocumentChunk]
    ) -> IndexVersion:
        staged = index_version.with_status("building")
        self._index_versions[staged.index_version_id] = staged
        self._chunks[staged.index_version_id] = tuple(chunks)
        return staged

    def mark_validated(self, index_version_id: str) -> IndexVersion:
        validated = self._require(index_version_id).with_status("validated")
        self._index_versions[index_version_id] = validated
        return validated

    def retire(self, index_version_id: str) -> IndexVersion:
        retired = self._require(index_version_id).with_status("retired")
        self._index_versions[index_version_id] = retired
        return retired

    def activate(self, index_version_id: str) -> IndexVersion:
        candidate = self._require(index_version_id)
        if candidate.status != "validated":
            raise IndexActivationError(
                f"index version {index_version_id} must be validated before activation"
            )
        previous = self._active.get(candidate.tenant_id)
        if previous is not None and previous != index_version_id:
            self._activation_history.setdefault(candidate.tenant_id, []).append(previous)
            self.retire(previous)
        active = candidate.with_status("active")
        self._index_versions[index_version_id] = active
        self._active[candidate.tenant_id] = index_version_id
        return active

    def rollback(self, tenant_id: str) -> IndexVersion:
        history = self._activation_history.get(tenant_id) or []
        if not history:
            raise IndexActivationError(f"no previous index version for tenant {tenant_id}")
        previous_id = history.pop()
        current_id = self._active.get(tenant_id)
        if current_id is not None:
            self.retire(current_id)
        restored = self._require(previous_id).with_status("active")
        self._index_versions[previous_id] = restored
        self._active[tenant_id] = previous_id
        return restored

    def active_index_version(self, tenant_id: str) -> IndexVersion | None:
        index_version_id = self._active.get(tenant_id)
        return None if index_version_id is None else self._index_versions[index_version_id]

    def index_version(self, index_version_id: str) -> IndexVersion:
        return self._require(index_version_id)

    def index_versions(self) -> tuple[IndexVersion, ...]:
        return tuple(self._index_versions.values())

    # Chunks ----------------------------------------------------------------
    def chunks_for(self, index_version_id: str) -> tuple[DocumentChunk, ...]:
        self._require(index_version_id)
        return self._chunks[index_version_id]

    def active_chunks(self, tenant_id: str) -> tuple[DocumentChunk, ...]:
        active = self.active_index_version(tenant_id)
        return () if active is None else self._chunks[active.index_version_id]

    def chunk(self, tenant_id: str, chunk_id: str) -> DocumentChunk | None:
        for candidate in self.active_chunks(tenant_id):
            if candidate.chunk_id == chunk_id:
                return candidate
        return None

    def _require(self, index_version_id: str) -> IndexVersion:
        try:
            return self._index_versions[index_version_id]
        except KeyError as error:
            raise UnknownIndexVersionError(index_version_id) from error

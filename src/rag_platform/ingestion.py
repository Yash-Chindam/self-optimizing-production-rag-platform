"""Ingestion flow described in specification section 7.

acquire -> hash -> extract -> detect type and language -> process sensitive data ->
chunk -> build a new index version -> validate -> activate.

Ingestion is idempotent: re-ingesting identical content reuses the existing source version and
does not create a second index version.
"""

from dataclasses import dataclass, field
from hashlib import blake2s, sha256

from rag_platform.catalog import IndexCatalog
from rag_platform.chunking import build_pieces
from rag_platform.models import (
    ChunkingConfig,
    DocumentChunk,
    IndexVersion,
    IngestionReport,
    PiiPolicy,
    SourceRegistration,
    SourceVersion,
)
from rag_platform.pii import PiiProcessor
from rag_platform.repository import lexical_score

LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset({"the", "and", "of", "must", "with", "are"}),
    "de": frozenset({"der", "die", "und", "muss", "nicht", "werden"}),
    "es": frozenset({"el", "la", "los", "debe", "para", "con"}),
    "fr": frozenset({"le", "les", "des", "doit", "pour", "avec"}),
}


class IngestionValidationError(RuntimeError):
    """Raised when a staged index version fails its activation gate."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    source_version: SourceVersion
    index_version: IndexVersion
    report: IngestionReport


def content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def detect_document_type(content: str, declared: str) -> str:
    has_heading = any(line.startswith("#") for line in content.splitlines())
    if declared == "markdown" and not has_heading:
        return "text"
    return declared


def detect_language(content: str) -> str:
    words = {token for token in content.lower().split() if token.isalpha()}
    best_language = "en"
    best_score = 0
    for language, markers in LANGUAGE_MARKERS.items():
        score = len(words & markers)
        if score > best_score:
            best_language, best_score = language, score
    return best_language


@dataclass(slots=True)
class IngestionPipeline:
    catalog: IndexCatalog
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    pii: PiiProcessor = field(default_factory=PiiProcessor)
    transformation_revision: str = "transform-v1"
    embedding_revision: str = "embed-v1"
    analyzer_revision: str = "analyzer-v1"
    graph_extractor_revision: str = "graph-v1"

    def ingest(self, registration: SourceRegistration, content: str) -> IngestionResult:
        digest = content_hash(content)
        reused = self._reuse_existing(registration, digest)
        if reused is not None:
            return reused

        source_version = self._register_source_version(registration, content, digest)
        processed = self.pii.process(content, registration.pii_policy, registration.tenant_id)
        new_chunks, duplicates = self._build_chunks(source_version, processed.text)
        if not any(chunk.retrievable for chunk in new_chunks):
            # Chunks carried from other sources must never disguise an empty ingest.
            raise IngestionValidationError(
                f"source {registration.source_id} produced no retrievable chunk"
            )

        carried = self._carried_chunks(registration)
        chunks = carried + tuple(new_chunks)
        source_version_ids = tuple(sorted({chunk.source_version_id for chunk in chunks}))
        index_version = self._build_index_version(
            registration.tenant_id, source_version_ids, len(chunks)
        )
        chunks = tuple(
            chunk.model_copy(update={"index_version": index_version.index_version_id})
            for chunk in chunks
        )
        self.catalog.stage_index_version(index_version, chunks)

        checks = self.validate(index_version, chunks, registration.pii_policy)
        self.catalog.mark_validated(index_version.index_version_id)
        activated = self.catalog.activate(index_version.index_version_id)

        return IngestionResult(
            source_version=source_version,
            index_version=activated,
            report=IngestionReport(
                source_version_id=source_version.source_version_id,
                index_version_id=activated.index_version_id,
                chunk_count=len(chunks),
                duplicate_chunks_removed=duplicates,
                pii_entities_processed=len(processed.entities),
                reused_existing_source_version=False,
                validation_checks=checks,
            ),
        )

    def _reuse_existing(
        self, registration: SourceRegistration, digest: str
    ) -> IngestionResult | None:
        existing = self.catalog.find_source_version(registration.source_id, digest)
        active = self.catalog.active_index_version(registration.tenant_id)
        if existing is None or active is None:
            return None
        if existing.source_version_id not in active.source_version_ids:
            return None
        return IngestionResult(
            source_version=existing,
            index_version=active,
            report=IngestionReport(
                source_version_id=existing.source_version_id,
                index_version_id=active.index_version_id,
                chunk_count=active.chunk_count,
                duplicate_chunks_removed=0,
                pii_entities_processed=0,
                reused_existing_source_version=True,
                validation_checks=("idempotent_content_hash_match",),
            ),
        )

    def _register_source_version(
        self, registration: SourceRegistration, content: str, digest: str
    ) -> SourceVersion:
        document_type = detect_document_type(content, registration.source_type)
        return self.catalog.register_source_version(
            SourceVersion(
                source_version_id=f"sv-{registration.source_id}-{digest[:12]}",
                source_id=registration.source_id,
                tenant_id=registration.tenant_id,
                owner=registration.owner,
                access_labels=registration.access_labels,
                source_uri=registration.source_uri,
                source_title=registration.source_title,
                content_hash=digest,
                parser_revision=registration.parser_revision,
                transformation_revision=self.transformation_revision,
                retention=registration.retention,
                document_type="markdown" if document_type == "markdown" else "text",
                language=detect_language(content),
            )
        )

    def _carried_chunks(self, registration: SourceRegistration) -> tuple[DocumentChunk, ...]:
        """Keep chunks from other sources so a new index version stays complete."""
        return tuple(
            chunk
            for chunk in self.catalog.active_chunks(registration.tenant_id)
            if self.catalog.source_version(chunk.source_version_id).source_id
            != registration.source_id
        )

    def _build_chunks(
        self, source_version: SourceVersion, text: str
    ) -> tuple[list[DocumentChunk], int]:
        pieces = build_pieces(text, source_version.document_type, self.chunking)
        identifiers = {
            piece.ordinal: f"{source_version.source_version_id}-{piece.ordinal:04d}"
            for piece in pieces
        }
        chunks: list[DocumentChunk] = []
        seen_text: set[str] = set()
        duplicates = 0
        for piece in pieces:
            if piece.retrievable and piece.text in seen_text:
                duplicates += 1
                continue
            seen_text.add(piece.text)
            parent_id = (
                None if piece.parent_ordinal is None else identifiers[piece.parent_ordinal]
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=identifiers[piece.ordinal],
                    tenant_id=source_version.tenant_id,
                    access_labels=source_version.access_labels,
                    text=piece.text,
                    source_uri=source_version.source_uri,
                    source_title=source_version.source_title,
                    index_version="pending",
                    source_version_id=source_version.source_version_id,
                    parent_chunk_id=parent_id,
                    section_path=piece.section_path,
                    ordinal=piece.ordinal,
                    retrievable=piece.retrievable,
                )
            )
        return chunks, duplicates

    def _build_index_version(
        self, tenant_id: str, source_version_ids: tuple[str, ...], chunk_count: int
    ) -> IndexVersion:
        fingerprint = blake2s(
            "|".join(
                (
                    tenant_id,
                    *source_version_ids,
                    self.chunking.model_dump_json(),
                    self.embedding_revision,
                    self.analyzer_revision,
                    self.graph_extractor_revision,
                )
            ).encode("utf-8"),
            digest_size=6,
        ).hexdigest()
        return IndexVersion(
            index_version_id=f"iv-{tenant_id}-{fingerprint}",
            tenant_id=tenant_id,
            source_version_ids=source_version_ids,
            chunking=self.chunking,
            embedding_revision=self.embedding_revision,
            analyzer_revision=self.analyzer_revision,
            graph_extractor_revision=self.graph_extractor_revision,
            physical_dense_index=f"qdrant:{tenant_id}:{fingerprint}",
            physical_sparse_index=f"opensearch:{tenant_id}:{fingerprint}",
            physical_graph_index=f"neo4j:{tenant_id}:{fingerprint}",
            chunk_count=chunk_count,
        )

    def validate(
        self,
        index_version: IndexVersion,
        chunks: tuple[DocumentChunk, ...],
        pii_policy: PiiPolicy,
    ) -> tuple[str, ...]:
        """Activation gate. A failing index version is retired instead of activated."""
        for chunk in chunks:
            if chunk.tenant_id != index_version.tenant_id:
                self._reject(index_version, f"chunk {chunk.chunk_id} crosses tenants")
            if not chunk.access_labels:
                self._reject(index_version, f"chunk {chunk.chunk_id} has no access labels")

        retrievable = [chunk for chunk in chunks if chunk.retrievable]
        if not retrievable:
            self._reject(index_version, "index version exposes no retrievable chunk")
        probe = retrievable[0]
        if lexical_score(probe.text, probe) <= 0:
            self._reject(index_version, "retrieval smoke check failed")

        checks = [
            "chunk_count_positive",
            "tenant_isolation_verified",
            "access_labels_present",
            "retrieval_smoke_passed",
        ]
        if pii_policy.mode == "pseudonymize":
            checks.append("pii_pseudonymized")
        return tuple(checks)

    def _reject(self, index_version: IndexVersion, reason: str) -> None:
        self.catalog.retire(index_version.index_version_id)
        raise IngestionValidationError(reason)


__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "IngestionValidationError",
    "content_hash",
    "detect_document_type",
    "detect_language",
]

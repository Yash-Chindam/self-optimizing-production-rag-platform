"""Dense retrieval over Qdrant (specification sections 6 and 8).

One Qdrant collection per `IndexVersion`, named by the `physical_dense_index` the catalog
already records. That is what makes staged activation real rather than nominal: a new index
version is written into its own collection, validated there, and only then activated, so
activation and rollback never mutate the collection serving live traffic.

Every query sends a mandatory tenant filter plus a subset filter on `required_labels`, and the
authoritative `is_authorized` check still runs in Python on whatever comes back. Qdrant cannot
express "every label on this document is held by the caller" directly, so the store filter is a
narrowing step and the application keeps the final say (specification section 4).
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rag_platform.adapters.embeddings import HashingEmbedder, TextEmbedder
from rag_platform.adapters.payload import from_payload, to_payload
from rag_platform.models import AccessContext, DocumentChunk
from rag_platform.repository import ScoredChunk, is_authorized

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from qdrant_client import QdrantClient

OVERFETCH = 4
"""Query more than requested, because the Python authorization recheck can drop hits."""


@dataclass(slots=True)
class QdrantDenseIndex:
    client: "QdrantClient"
    collection: str
    embedder: TextEmbedder = field(default_factory=HashingEmbedder)

    # Write path ------------------------------------------------------------
    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.embedder.dimensions, distance=Distance.COSINE
            ),
        )
        # Every query filters on these three, so each one is worth a payload index.
        for name, schema in (
            ("tenant_id", PayloadSchemaType.KEYWORD),
            ("required_labels", PayloadSchemaType.KEYWORD),
            ("retrievable", PayloadSchemaType.BOOL),
        ):
            self.client.create_payload_index(
                collection_name=self.collection, field_name=name, field_schema=schema
            )

    def upsert(self, chunks: Iterable[DocumentChunk]) -> int:
        from qdrant_client.models import PointStruct

        batch = list(chunks)
        if not batch:
            return 0
        vectors = self.embedder.embed([chunk.text for chunk in batch])
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(id=_point_id(chunk), vector=vector, payload=to_payload(chunk))
                for chunk, vector in zip(batch, vectors, strict=True)
            ],
        )
        return len(batch)

    def drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    # Read path -------------------------------------------------------------
    def semantic_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        if limit <= 0:
            return []
        vector = self.embedder.embed([question])[0]
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=self.access_filter(access),
            limit=limit * OVERFETCH,
            with_payload=True,
        ).points
        return _authorized_results(hits, access, limit)

    def access_filter(self, access: AccessContext) -> Any:
        """Mandatory tenant match, plus the labels the caller holds. Never optional."""
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchAny,
            MatchValue,
            PayloadField,
        )

        conditions: list[Any] = [
            FieldCondition(key="tenant_id", match=MatchValue(value=access.tenant_id)),
            FieldCondition(key="retrievable", match=MatchValue(value=True)),
        ]
        # A chunk with no required labels is public; one with required labels must have all of
        # them held by the caller. Qdrant cannot express the subset test, so this narrows to
        # "requires nothing, or mentions at least one held label" and `is_authorized` finishes
        # the job. Beside `must`, at least one `should` condition has to match.
        should: list[Any] = [IsEmptyCondition(is_empty=PayloadField(key="required_labels"))]
        held = sorted(access.labels)
        if held:
            should.append(FieldCondition(key="required_labels", match=MatchAny(any=held)))
        return Filter(must=conditions, should=should)


def _authorized_results(
    hits: Sequence[Any], access: AccessContext, limit: int
) -> list[ScoredChunk]:
    results: list[ScoredChunk] = []
    for hit in hits:
        if not hit.payload:
            continue
        chunk = from_payload(dict(hit.payload))
        if not chunk.retrievable or not is_authorized(chunk, access):
            continue
        results.append(ScoredChunk(chunk=chunk, score=float(hit.score)))
        if len(results) == limit:
            break
    return results


def _point_id(chunk: DocumentChunk) -> str:
    import uuid

    # Qdrant ids must be a UUID or an unsigned integer, and chunk ids are neither.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))

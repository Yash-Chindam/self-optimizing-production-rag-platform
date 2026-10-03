"""Chunk-to-store payload codec shared by every retrieval adapter.

`access_labels` is stored twice on purpose. `access_labels` is the full set, kept so a chunk
round-trips exactly. `required_labels` is the set minus `public` — the labels a caller must
actually hold — because that is the field a store can filter on with subset semantics. Deriving
it at write time keeps the query side simple and keeps both in one place.
"""

from typing import Any

from rag_platform.models import DocumentChunk

PUBLIC_LABEL = "public"


def to_payload(chunk: DocumentChunk) -> dict[str, Any]:
    payload = chunk.model_dump(mode="json")
    payload["access_labels"] = sorted(chunk.access_labels)
    payload["required_labels"] = sorted(chunk.access_labels - {PUBLIC_LABEL})
    payload["section_path"] = list(chunk.section_path)
    return payload


def from_payload(payload: dict[str, Any]) -> DocumentChunk:
    fields = dict(payload)
    fields.pop("required_labels", None)
    fields["access_labels"] = frozenset(fields.get("access_labels") or ())
    fields["section_path"] = tuple(fields.get("section_path") or ())
    return DocumentChunk.model_validate(fields)


def required_labels(chunk: DocumentChunk) -> list[str]:
    return sorted(chunk.access_labels - {PUBLIC_LABEL})

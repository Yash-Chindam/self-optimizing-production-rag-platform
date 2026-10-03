"""Sparse retrieval over OpenSearch (specification sections 6 and 8).

BM25 for exact identifiers, acronyms, terminology and rare vocabulary — the vocabulary dense
retrieval is worst at. One index per `IndexVersion`, named by the `physical_sparse_index` the
catalog records, so staged activation and rollback work the same way they do for Qdrant.

Unlike Qdrant, OpenSearch can express the authorization subset test exactly: `terms_set` with
`minimum_should_match_field` requires every one of a document's `required_labels` to appear in
the caller's label set. The Python `is_authorized` recheck still runs, because the specification
puts the authorization boundary in the application rather than in a retrieval database.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_platform.adapters.payload import from_payload, to_payload
from rag_platform.models import AccessContext, DocumentChunk
from rag_platform.repository import ScoredChunk, is_authorized

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from opensearchpy import OpenSearch

MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "tenant_id": {"type": "keyword"},
            "access_labels": {"type": "keyword"},
            "required_labels": {"type": "keyword"},
            "required_label_count": {"type": "integer"},
            "text": {"type": "text", "analyzer": "english"},
            "source_uri": {"type": "keyword"},
            "source_title": {"type": "text"},
            "index_version": {"type": "keyword"},
            "source_version_id": {"type": "keyword"},
            "parent_chunk_id": {"type": "keyword"},
            "section_path": {"type": "keyword"},
            "ordinal": {"type": "integer"},
            "retrievable": {"type": "boolean"},
        }
    }
}


@dataclass(slots=True)
class OpenSearchSparseIndex:
    client: "OpenSearch"
    index: str

    # Write path ------------------------------------------------------------
    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        self.client.indices.create(index=self.index, body=MAPPING)

    def upsert(self, chunks: Iterable[DocumentChunk]) -> int:
        batch = list(chunks)
        if not batch:
            return 0
        operations: list[dict[str, Any]] = []
        for chunk in batch:
            document = to_payload(chunk)
            document["required_label_count"] = len(document["required_labels"])
            operations.append({"index": {"_index": self.index, "_id": chunk.chunk_id}})
            operations.append(document)
        self.client.bulk(body=operations, refresh=True)
        return len(batch)

    def drop(self) -> None:
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)

    # Read path -------------------------------------------------------------
    def lexical_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        if limit <= 0:
            return []
        response = self.client.search(
            index=self.index, body=self.query(question, access, limit=limit)
        )
        hits = response.get("hits", {}).get("hits", [])
        return _authorized_results(hits, access, limit)

    def query(self, question: str, access: AccessContext, *, limit: int) -> dict[str, Any]:
        return {
            "size": limit,
            "query": {
                "bool": {
                    "must": [{"match": {"text": {"query": question}}}],
                    "filter": self.access_filter(access),
                }
            },
        }

    def access_filter(self, access: AccessContext) -> list[dict[str, Any]]:
        """Mandatory tenant term, plus an exact subset test on the required labels."""
        return [
            {"term": {"tenant_id": access.tenant_id}},
            {"term": {"retrievable": True}},
            {
                "bool": {
                    # `terms_set` never matches a document whose required count is zero, so a
                    # chunk that requires no label is admitted by its own clause.
                    "should": [
                        {"term": {"required_label_count": 0}},
                        {
                            "terms_set": {
                                "required_labels": {
                                    "terms": sorted(access.labels),
                                    "minimum_should_match_field": "required_label_count",
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
        ]


def _authorized_results(
    hits: Sequence[dict[str, Any]], access: AccessContext, limit: int
) -> list[ScoredChunk]:
    results: list[ScoredChunk] = []
    for hit in hits:
        source = hit.get("_source")
        if not source:
            continue
        chunk = from_payload(dict(source))
        if not chunk.retrievable or not is_authorized(chunk, access):
            continue
        results.append(ScoredChunk(chunk=chunk, score=float(hit.get("_score") or 0.0)))
        if len(results) == limit:
            break
    return results


def document_for(chunk: DocumentChunk) -> dict[str, Any]:
    """The indexed document, exposed for tests and for bulk loaders outside this adapter."""
    document = to_payload(chunk)
    document["required_label_count"] = len(document["required_labels"])
    return document

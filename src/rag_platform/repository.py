import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from rag_platform.catalog import IndexCatalog
from rag_platform.models import AccessContext, DocumentChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "say",
    "the",
    "to",
    "what",
}
SEMANTIC_ALIASES = {
    "holiday": "leave",
    "holidays": "leave",
    "vacation": "leave",
    "pto": "leave",
    "requesting": "submit",
    "requests": "submit",
    "policy": "handbook",
}


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: DocumentChunk
    score: float


class ChunkRepository(Protocol):
    def lexical_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]: ...

    def semantic_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]: ...


def tokenize(text: str, *, semantic: bool = False) -> list[str]:
    tokens = [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOP_WORDS]
    if semantic:
        return [SEMANTIC_ALIASES.get(token, token) for token in tokens]
    return tokens


def is_authorized(chunk: DocumentChunk, access: AccessContext) -> bool:
    if chunk.tenant_id != access.tenant_id:
        return False
    required = chunk.access_labels - {"public"}
    return required.issubset(access.labels)


def lexical_score(question: str, chunk: DocumentChunk) -> float:
    query_counts = Counter(tokenize(question))
    if not query_counts:
        return 0.0
    document_counts = Counter(tokenize(chunk.text))
    return float(sum(min(count, document_counts[token]) for token, count in query_counts.items()))


def semantic_score(question: str, chunk: DocumentChunk) -> float:
    query_counts = Counter(tokenize(question, semantic=True))
    if not query_counts:
        return 0.0
    return _cosine_similarity(query_counts, Counter(tokenize(chunk.text, semantic=True)))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    dot_product = sum(value * right[token] for token, value in left.items())
    if dot_product == 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot_product / (left_norm * right_norm)


def rank(results: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
    return sorted(results, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit]


class ScoringChunkRepository:
    """Shared mandatory-filter and scoring contract for every chunk source."""

    def candidates(self, access: AccessContext) -> Iterable[DocumentChunk]:
        raise NotImplementedError

    def lexical_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        return self._search(question, access, limit=limit, score=lexical_score)

    def semantic_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        return self._search(question, access, limit=limit, score=semantic_score)

    def _search(
        self,
        question: str,
        access: AccessContext,
        *,
        limit: int,
        score: Callable[[str, DocumentChunk], float],
    ) -> list[ScoredChunk]:
        results = [
            ScoredChunk(chunk=chunk, score=value)
            for chunk in self.candidates(access)
            if (value := score(question, chunk)) > 0
        ]
        return rank(results, limit)


class InMemoryChunkRepository(ScoringChunkRepository):
    """Deterministic starter adapter with the same mandatory filter contract as real stores."""

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self._chunks = tuple(chunks)

    def candidates(self, access: AccessContext) -> Iterable[DocumentChunk]:
        return _authorized(self._chunks, access)


class CatalogChunkRepository(ScoringChunkRepository):
    """Serves only the chunks of the tenant's currently active index version."""

    def __init__(self, catalog: IndexCatalog) -> None:
        self._catalog = catalog

    def candidates(self, access: AccessContext) -> Iterable[DocumentChunk]:
        return _authorized(self._catalog.active_chunks(access.tenant_id), access)


def _authorized(
    chunks: Iterable[DocumentChunk], access: AccessContext
) -> Iterable[DocumentChunk]:
    return (
        chunk
        for chunk in chunks
        if chunk.retrievable and is_authorized(chunk, access)
    )

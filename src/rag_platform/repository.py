import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

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


class InMemoryChunkRepository:
    """Deterministic starter adapter with the same mandatory filter contract as real stores."""

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self._chunks = tuple(chunks)

    def lexical_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        query_counts = Counter(tokenize(question))
        if not query_counts:
            return []
        results: list[ScoredChunk] = []
        for chunk in self._authorized_chunks(access):
            document_counts = Counter(tokenize(chunk.text))
            overlap = sum(
                min(count, document_counts[token]) for token, count in query_counts.items()
            )
            if overlap:
                results.append(ScoredChunk(chunk=chunk, score=float(overlap)))
        return self._rank(results, limit)

    def semantic_search(
        self, question: str, access: AccessContext, *, limit: int
    ) -> list[ScoredChunk]:
        query_counts = Counter(tokenize(question, semantic=True))
        if not query_counts:
            return []
        results: list[ScoredChunk] = []
        for chunk in self._authorized_chunks(access):
            document_counts = Counter(tokenize(chunk.text, semantic=True))
            score = self._cosine_similarity(query_counts, document_counts)
            if score > 0:
                results.append(ScoredChunk(chunk=chunk, score=score))
        return self._rank(results, limit)

    def _authorized_chunks(self, access: AccessContext) -> Iterable[DocumentChunk]:
        return (chunk for chunk in self._chunks if is_authorized(chunk, access))

    @staticmethod
    def _rank(results: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        return sorted(results, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit]

    @staticmethod
    def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
        dot_product = sum(value * right[token] for token, value in left.items())
        if dot_product == 0:
            return 0.0
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot_product / (left_norm * right_norm)

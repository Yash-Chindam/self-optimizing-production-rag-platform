"""Reranking of a bounded candidate set.

The reranker is versioned independently of the retrieval configuration so a reranker change can
be evaluated and rolled back on its own. `LexicalCrossEncoder` is a deterministic stand-in with
the interface a cross-encoder adapter implements: it scores query-term coverage first and uses
term proximity only to break ties, which is the signal a cross-encoder approximates.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rag_platform.models import DocumentChunk
from rag_platform.repository import ScoredChunk, tokenize


class Reranker(Protocol):
    revision: str

    def rerank(
        self, question: str, candidates: Iterable[DocumentChunk], *, limit: int
    ) -> list[ScoredChunk]: ...


@dataclass(slots=True)
class LexicalCrossEncoder:
    revision: str = "reranker-lexical-v1"
    proximity_weight: float = 0.25

    def rerank(
        self, question: str, candidates: Iterable[DocumentChunk], *, limit: int
    ) -> list[ScoredChunk]:
        query_tokens = tuple(dict.fromkeys(tokenize(question, semantic=True)))
        scored = [
            ScoredChunk(chunk=chunk, score=self.score(query_tokens, chunk))
            for chunk in candidates
        ]
        ranked = sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))
        return [item for item in ranked if item.score > 0][:limit]

    def score(self, query_tokens: tuple[str, ...], chunk: DocumentChunk) -> float:
        if not query_tokens:
            return 0.0
        document_tokens = tokenize(chunk.text, semantic=True)
        matched = [token for token in query_tokens if token in set(document_tokens)]
        if not matched:
            return 0.0
        coverage = len(matched) / len(query_tokens)
        return coverage + self.proximity_weight * _proximity(matched, document_tokens)


def _proximity(matched: list[str], document_tokens: list[str]) -> float:
    """1.0 when every matched term sits in the tightest possible window, approaching 0 apart."""
    if len(matched) < 2:
        return 1.0
    wanted = set(matched)
    positions = [
        (index, token) for index, token in enumerate(document_tokens) if token in wanted
    ]
    best_span: int | None = None
    for start, (start_index, _token) in enumerate(positions):
        covered: set[str] = set()
        for end_index, token in positions[start:]:
            covered.add(token)
            if covered == wanted:
                span = end_index - start_index + 1
                best_span = span if best_span is None else min(best_span, span)
                break
    if best_span is None:
        return 0.0
    return len(wanted) / best_span

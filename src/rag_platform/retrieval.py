"""Hybrid retrieval: dense and sparse candidates, versioned fusion, graph expansion, reranking.

Fusion weights, graph depth and the reranker revision are configuration rather than code, so a
change to any of them is a new candidate configuration that evaluation can accept or reject.

Graph expansion and reranking are optional enrichments, not the retrieval boundary itself: dense
and sparse search already produced an authorized, if unranked, candidate set before either runs.
A failing graph or reranking dependency therefore degrades the result to what came before it
(specification section 17) instead of failing the query, through the `CircuitBreaker` each is
called behind.
"""

from dataclasses import dataclass

from rag_platform.graph import GraphExpander
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.reliability import CircuitBreaker, DegradedCall
from rag_platform.repository import ChunkRepository, ScoredChunk
from rag_platform.rerank import Reranker


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    fused_score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunks: tuple[RetrievedChunk, ...]
    strategy: str
    graph_expanded_chunk_ids: tuple[str, ...]
    degraded_dependencies: tuple[str, ...] = ()

    def scored(self) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=item.chunk, score=item.fused_score) for item in self.chunks]


class HybridRetriever:
    def __init__(
        self,
        repository: ChunkRepository,
        config: PipelineConfig,
        graph_retriever: GraphExpander | None = None,
        reranker: Reranker | None = None,
        graph_breaker: CircuitBreaker | None = None,
        reranker_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._graph = graph_retriever
        self._reranker = reranker
        self._graph_breaker = graph_breaker or CircuitBreaker(name="graph_expansion")
        self._reranker_breaker = reranker_breaker or CircuitBreaker(name="reranker")

    @property
    def strategy(self) -> str:
        parts = [f"hybrid_{self._config.fusion_method}"]
        if self._graph is not None and self._config.graph_expansion_depth > 0:
            parts.append(f"graph_depth_{self._config.graph_expansion_depth}")
        if self._reranker is not None:
            parts.append(self._reranker.revision)
        return "+".join(parts)

    def retrieve(self, question: str, access: AccessContext) -> RetrievalResult:
        dense = self._repository.semantic_search(
            question, access, limit=self._config.dense_top_k
        )
        sparse = self._repository.lexical_search(
            question, access, limit=self._config.sparse_top_k
        )
        fused = self._fuse(dense, sparse)
        expanded, graph_note = self._expand_with_graph(fused, access)
        candidates = fused + expanded
        ranked, rerank_note = self._rerank(question, candidates)
        degraded = tuple(note.detail for note in (graph_note, rerank_note) if not note.ok)
        return RetrievalResult(
            chunks=tuple(ranked[: self._config.final_top_k]),
            strategy=self.strategy,
            graph_expanded_chunk_ids=tuple(item.chunk.chunk_id for item in expanded),
            degraded_dependencies=degraded,
        )

    def _fuse(
        self, dense: list[ScoredChunk], sparse: list[ScoredChunk]
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, DocumentChunk] = {}
        if self._config.fusion_method == "weighted":
            self._add_weighted(dense, self._config.dense_weight, scores, chunks)
            self._add_weighted(sparse, self._config.sparse_weight, scores, chunks)
        else:
            self._add_reciprocal_rank(dense, scores, chunks)
            self._add_reciprocal_rank(sparse, scores, chunks)
        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        return [
            RetrievedChunk(chunk=chunks[chunk_id], fused_score=scores[chunk_id])
            for chunk_id in ranked
        ]

    def _add_reciprocal_rank(
        self,
        results: list[ScoredChunk],
        scores: dict[str, float],
        chunks: dict[str, DocumentChunk],
    ) -> None:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1 / (self._config.reciprocal_rank_constant + rank)
            )
            chunks[chunk_id] = result.chunk

    @staticmethod
    def _add_weighted(
        results: list[ScoredChunk],
        weight: float,
        scores: dict[str, float],
        chunks: dict[str, DocumentChunk],
    ) -> None:
        """Normalized weighted fusion: each strategy contributes on its own 0-1 scale."""
        if not results:
            return
        best = max(result.score for result in results)
        if best <= 0:
            return
        for result in results:
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * (result.score / best)
            chunks[chunk_id] = result.chunk

    def _expand_with_graph(
        self, fused: list[RetrievedChunk], access: AccessContext
    ) -> tuple[list[RetrievedChunk], DegradedCall]:
        graph = self._graph
        if graph is None or not fused:
            return [], DegradedCall(ok=True, detail="graph_expansion_skipped")
        seeds = [item.chunk for item in fused[: self._config.final_top_k]]
        no_expansion: list[DocumentChunk] = []
        expanded, note = self._graph_breaker.call(
            lambda: graph.expand(
                seeds,
                access,
                depth=self._config.graph_expansion_depth,
                limit=self._config.graph_candidate_limit,
            ),
            fallback=no_expansion,
        )
        lowest = min((item.fused_score for item in fused), default=0.0)
        # Graph candidates enter the pool below every directly retrieved chunk; the reranker,
        # not the traversal, decides whether they belong in the answer.
        return [RetrievedChunk(chunk=chunk, fused_score=lowest / 2) for chunk in expanded], note

    def _rerank(
        self, question: str, candidates: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], DegradedCall]:
        reranker = self._reranker
        if reranker is None or not candidates:
            return candidates, DegradedCall(ok=True, detail="reranker_skipped")
        bounded = candidates[: self._config.rerank_candidate_count]
        fallback = [ScoredChunk(chunk=item.chunk, score=item.fused_score) for item in bounded]
        reranked, note = self._reranker_breaker.call(
            lambda: reranker.rerank(
                question, [item.chunk for item in bounded], limit=len(bounded)
            ),
            fallback=fallback,
        )
        return [
            RetrievedChunk(chunk=item.chunk, fused_score=item.score) for item in reranked
        ], note

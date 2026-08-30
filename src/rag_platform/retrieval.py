from dataclasses import dataclass

from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.repository import ChunkRepository, ScoredChunk


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: DocumentChunk
    fused_score: float


class HybridRetriever:
    def __init__(self, repository: ChunkRepository, config: PipelineConfig) -> None:
        self._repository = repository
        self._config = config

    def retrieve(self, question: str, access: AccessContext) -> list[RetrievedChunk]:
        dense = self._repository.semantic_search(
            question, access, limit=self._config.dense_top_k
        )
        sparse = self._repository.lexical_search(
            question, access, limit=self._config.sparse_top_k
        )
        scores: dict[str, float] = {}
        chunks: dict[str, DocumentChunk] = {}
        self._add_reciprocal_rank_scores(dense, scores, chunks)
        self._add_reciprocal_rank_scores(sparse, scores, chunks)
        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        return [
            RetrievedChunk(chunk=chunks[chunk_id], fused_score=scores[chunk_id])
            for chunk_id in ranked[: self._config.final_top_k]
        ]

    def _add_reciprocal_rank_scores(
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


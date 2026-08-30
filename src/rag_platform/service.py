from rag_platform.models import (
    AccessContext,
    AnswerTrace,
    Citation,
    PipelineConfig,
    QueryResponse,
)
from rag_platform.retrieval import HybridRetriever, RetrievedChunk


class QueryService:
    def __init__(self, retriever: HybridRetriever, config: PipelineConfig) -> None:
        self._retriever = retriever
        self.config = config

    def answer(self, question: str, access: AccessContext) -> QueryResponse:
        retrieved = self._retriever.retrieve(question, access)
        selected = self._within_context_budget(retrieved)
        trace = AnswerTrace(
            config_version=self.config.version,
            index_version=selected[0].chunk.index_version if selected else None,
            retrieval_strategy="hybrid_rrf",
            retrieved_chunk_ids=[item.chunk.chunk_id for item in selected],
            policy="tenant_and_access_labels_required_before_scoring",
        )
        if not selected:
            return QueryResponse(
                status="insufficient_evidence",
                answer="I could not find authorized evidence for that question.",
                citations=[],
                trace=trace,
            )

        best = selected[0].chunk
        return QueryResponse(
            status="answered",
            answer=best.text,
            citations=[
                Citation(
                    citation_id="C1",
                    source_title=best.source_title,
                    source_uri=best.source_uri,
                    chunk_id=best.chunk_id,
                )
            ],
            trace=trace,
        )

    def _within_context_budget(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        used = 0
        for item in retrieved:
            size = len(item.chunk.text)
            if used + size > self.config.context_character_budget:
                continue
            selected.append(item)
            used += size
        return selected


from rag_platform.context import ContextBuilder, ContextBundle
from rag_platform.models import (
    AccessContext,
    AnswerTrace,
    Citation,
    PipelineConfig,
    QueryResponse,
)
from rag_platform.retrieval import HybridRetriever, RetrievalResult


class QueryService:
    def __init__(
        self,
        retriever: HybridRetriever,
        config: PipelineConfig,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._retriever = retriever
        self.config = config
        self._context = context_builder or ContextBuilder(config)

    def answer(self, question: str, access: AccessContext) -> QueryResponse:
        retrieved = self._retriever.retrieve(question, access)
        bundle = self._context.build(retrieved.scored(), access)
        trace = self._trace(retrieved, bundle)
        if not bundle:
            return QueryResponse(
                status="insufficient_evidence",
                answer="I could not find authorized evidence for that question.",
                citations=[],
                trace=trace,
            )

        best = bundle.items[0]
        return QueryResponse(
            status="answered",
            answer=best.chunk.text,
            citations=[
                Citation(
                    citation_id=best.citation_id,
                    source_title=best.chunk.source_title,
                    source_uri=best.chunk.source_uri,
                    chunk_id=best.chunk.chunk_id,
                )
            ],
            trace=trace,
        )

    def _trace(self, retrieved: RetrievalResult, bundle: ContextBundle) -> AnswerTrace:
        return AnswerTrace(
            config_version=self.config.version,
            index_version=bundle.items[0].chunk.index_version if bundle else None,
            retrieval_strategy=retrieved.strategy,
            retrieved_chunk_ids=[item.chunk.chunk_id for item in retrieved.chunks],
            policy="tenant_and_access_labels_required_before_scoring_and_before_context",
            context_chunk_ids=[item.chunk.chunk_id for item in bundle.items],
            graph_expanded_chunk_ids=list(retrieved.graph_expanded_chunk_ids),
            dropped_chunk_ids=list(bundle.dropped_chunk_ids),
            context_characters=bundle.used_characters,
            policy_notes=list(bundle.policy_notes),
        )

"""Explicit query workflow (specification section 9).

The workflow is a state machine, not a linear chain: classification, clarification, retrieval,
generation, verification, repair and fallback are separate states, and the path a query took is
recorded in its trace. This is the structure LangGraph owns in the deployed topology; keeping
the state explicit here means swapping the executor does not change the states or their tests.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from rag_platform.context import ContextBuilder, ContextBundle
from rag_platform.models import (
    AccessContext,
    AnswerTrace,
    Citation,
    PipelineConfig,
    QueryResponse,
)
from rag_platform.programs import (
    ProgramSuite,
    QueryClassification,
    SynthesizedAnswer,
    VerificationResult,
)
from rag_platform.reliability import CircuitBreaker
from rag_platform.retrieval import HybridRetriever, RetrievalResult, RetrievedChunk

INSUFFICIENT_EVIDENCE = "I could not find authorized evidence for that question."
MAX_STEPS = 24


@dataclass(slots=True)
class WorkflowState:
    question: str
    access: AccessContext
    classification: QueryClassification | None = None
    rewritten: str = ""
    subqueries: tuple[str, ...] = ()
    retrieval: RetrievalResult | None = None
    bundle: ContextBundle | None = None
    answer: SynthesizedAnswer | None = None
    verification: VerificationResult | None = None
    repairs: int = 0
    sentence_limit: int = 1
    clarification: str = ""
    path: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    degraded_dependencies: list[str] = field(default_factory=list)


class QueryWorkflow:
    def __init__(
        self,
        retriever: HybridRetriever,
        config: PipelineConfig,
        context_builder: ContextBuilder,
        programs: ProgramSuite | None = None,
        generator_breaker: CircuitBreaker | None = None,
        verifier_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._retriever = retriever
        self._config = config
        self._context = context_builder
        self._programs = programs or ProgramSuite()
        self._generator_breaker = generator_breaker or CircuitBreaker(name="generator")
        self._verifier_breaker = verifier_breaker or CircuitBreaker(name="verifier")

    def run(self, question: str, access: AccessContext) -> QueryResponse:
        state = WorkflowState(
            question=question,
            access=access,
            sentence_limit=self._config.answer_sentence_limit,
        )
        nodes: dict[str, Callable[[WorkflowState], str]] = {
            "classify": self._classify,
            "clarify": self._clarify,
            "rewrite": self._rewrite,
            "decompose": self._decompose,
            "retrieve": self._retrieve,
            "build_context": self._build_context,
            "generate": self._generate,
            "verify": self._verify,
            "repair": self._repair,
        }
        node = "classify"
        for _step in range(MAX_STEPS):
            state.path.append(node)
            node = nodes[node](state)
            if node == "answer":
                return self._answered(state)
            if node == "clarification":
                return self._clarification(state)
            if node == "fallback":
                return self._fallback(state)
        return self._fallback(state)

    # States ----------------------------------------------------------------
    def _classify(self, state: WorkflowState) -> str:
        state.classification = self._programs.classifier.classify(state.question)
        if (
            state.classification.intent == "ambiguous"
            and self._config.clarify_ambiguous_queries
        ):
            return "clarify"
        return "rewrite"

    def _clarify(self, state: WorkflowState) -> str:
        assert state.classification is not None
        state.clarification = self._programs.clarifier.generate(
            state.question, state.classification
        )
        return "clarification"

    def _rewrite(self, state: WorkflowState) -> str:
        assert state.classification is not None
        state.rewritten = self._programs.rewriter.rewrite(state.question, state.classification)
        if state.rewritten != state.question:
            state.transformations.append(f"rewrite:{state.rewritten}")
        return "decompose"

    def _decompose(self, state: WorkflowState) -> str:
        parts = self._programs.decomposer.decompose(
            state.rewritten, limit=self._config.max_subqueries
        )
        state.subqueries = tuple(parts)
        if len(parts) > 1:
            state.transformations.append(f"decompose:{' | '.join(parts)}")
        return "retrieve"

    def _retrieve(self, state: WorkflowState) -> str:
        results = [
            self._retriever.retrieve(subquery, state.access) for subquery in state.subqueries
        ]
        state.retrieval = _merge(results, limit=self._config.final_top_k)
        state.degraded_dependencies.extend(state.retrieval.degraded_dependencies)
        return "build_context"

    def _build_context(self, state: WorkflowState) -> str:
        assert state.retrieval is not None
        state.bundle = self._context.build(state.retrieval.scored(), state.access)
        return "generate" if state.bundle.items else "fallback"

    def _generate(self, state: WorkflowState) -> str:
        assert state.bundle is not None
        bundle = state.bundle
        no_answer = SynthesizedAnswer(text="", citation_ids=())
        answer, note = self._generator_breaker.call(
            lambda: self._programs.synthesizer.synthesize(
                state.question, bundle.items, sentence_limit=state.sentence_limit
            ),
            fallback=no_answer,
        )
        state.answer = answer
        if not note.ok:
            state.degraded_dependencies.append(note.detail)
        return "verify"

    def _verify(self, state: WorkflowState) -> str:
        assert state.answer is not None and state.bundle is not None
        answer, bundle = state.answer, state.bundle
        # A verifier that cannot run must never be treated as having confirmed grounding.
        claims = (answer.text,) if answer.text.strip() else ()
        unverifiable = VerificationResult(grounded=False, unsupported_claims=claims)
        verification, note = self._verifier_breaker.call(
            lambda: self._programs.verifier.verify(answer, bundle.items), fallback=unverifiable
        )
        state.verification = verification
        if not note.ok:
            state.degraded_dependencies.append(note.detail)
        if state.verification.grounded:
            return "answer"
        if state.repairs < self._config.max_repair_attempts:
            return "repair"
        return "fallback"

    def _repair(self, state: WorkflowState) -> str:
        """Narrow the answer to its single best-supported sentence and regenerate."""
        state.repairs += 1
        state.sentence_limit = 1
        state.transformations.append(f"repair:attempt_{state.repairs}")
        return "generate"

    # Responses -------------------------------------------------------------
    def _answered(self, state: WorkflowState) -> QueryResponse:
        assert state.answer is not None and state.bundle is not None
        cited = {item.citation_id: item for item in state.bundle.items}
        citations = [
            Citation(
                citation_id=citation_id,
                source_title=cited[citation_id].chunk.source_title,
                source_uri=cited[citation_id].chunk.source_uri,
                chunk_id=cited[citation_id].chunk.chunk_id,
            )
            for citation_id in state.answer.citation_ids
        ]
        return QueryResponse(
            status="answered",
            answer=state.answer.text,
            citations=citations,
            trace=self._trace(state),
        )

    def _clarification(self, state: WorkflowState) -> QueryResponse:
        return QueryResponse(
            status="clarification_needed",
            answer=state.clarification,
            citations=[],
            trace=self._trace(state),
        )

    def _fallback(self, state: WorkflowState) -> QueryResponse:
        return QueryResponse(
            status="insufficient_evidence",
            answer=INSUFFICIENT_EVIDENCE,
            citations=[],
            trace=self._trace(state),
        )

    def _trace(self, state: WorkflowState) -> AnswerTrace:
        bundle = state.bundle
        retrieval = state.retrieval
        verification = state.verification
        return AnswerTrace(
            config_version=self._config.version,
            index_version=(
                bundle.items[0].chunk.index_version if bundle is not None and bundle.items else None
            ),
            retrieval_strategy=retrieval.strategy if retrieval is not None else "none",
            retrieved_chunk_ids=(
                [item.chunk.chunk_id for item in retrieval.chunks] if retrieval is not None else []
            ),
            policy="tenant_and_access_labels_required_before_scoring_and_before_context",
            context_chunk_ids=(
                [item.chunk.chunk_id for item in bundle.items] if bundle is not None else []
            ),
            graph_expanded_chunk_ids=(
                list(retrieval.graph_expanded_chunk_ids) if retrieval is not None else []
            ),
            dropped_chunk_ids=list(bundle.dropped_chunk_ids) if bundle is not None else [],
            context_characters=bundle.used_characters if bundle is not None else 0,
            policy_notes=list(bundle.policy_notes) if bundle is not None else [],
            intent=state.classification.intent if state.classification else None,
            query_transformations=list(state.transformations),
            workflow_path=list(state.path),
            program_revisions=list(self._programs.revisions()),
            repair_attempts=state.repairs,
            unsupported_claims=(
                list(verification.unsupported_claims) if verification is not None else []
            ),
            degraded_dependencies=list(state.degraded_dependencies),
        )


def _merge(results: list[RetrievalResult], *, limit: int) -> RetrievalResult:
    """Combine sub-query results, keeping the best score seen for each chunk."""
    if not results:
        return RetrievalResult(chunks=(), strategy="none", graph_expanded_chunk_ids=())
    best: dict[str, RetrievedChunk] = {}
    for result in results:
        for item in result.chunks:
            current = best.get(item.chunk.chunk_id)
            if current is None or item.fused_score > current.fused_score:
                best[item.chunk.chunk_id] = item
    ranked = sorted(best.values(), key=lambda item: (-item.fused_score, item.chunk.chunk_id))
    expanded = tuple(
        dict.fromkeys(
            chunk_id for result in results for chunk_id in result.graph_expanded_chunk_ids
        )
    )
    degraded = tuple(
        dict.fromkeys(
            detail for result in results for detail in result.degraded_dependencies
        )
    )
    return RetrievalResult(
        chunks=tuple(ranked[:limit]),
        strategy=results[0].strategy,
        graph_expanded_chunk_ids=expanded,
        degraded_dependencies=degraded,
    )

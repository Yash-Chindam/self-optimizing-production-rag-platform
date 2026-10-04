"""OpenTelemetry tracing of the query workflow, in OpenInference conventions.

One query is one trace: a root CHAIN span and one child span per workflow state, so Phoenix (or
any OTLP backend) shows the path a query took, what was retrieved, and where it degraded
(specification sections 5 and 18).

Traces leave the trusted boundary for an observability stack that operators outside the tenant
can read, so this adapter is deliberately narrow about what it exports (section 15, "redact or
sample traces to prevent observability leakage"):

- chunk text is never exported; retrieval spans carry chunk ids, scores and versions only;
- the question and the answer are exported only when the policy allows it, and then only after
  the redactor has replaced detected identifiers and the text has been truncated;
- sampling is decided once, at the root span, so a query is traced whole or not at all.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from rag_platform.models import AccessContext, PipelineConfig, QueryResponse
from rag_platform.pii import redact_identifiers
from rag_platform.workflow import QueryObservation, WorkflowState

Redactor = Callable[[str], str]

PROJECT_NAME = "rag-platform"
ROOT_SPAN = "rag.query"

_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_CHAIN = OpenInferenceSpanKindValues.CHAIN.value
STATE_KINDS: dict[str, str] = {
    "retrieve": OpenInferenceSpanKindValues.RETRIEVER.value,
    "generate": OpenInferenceSpanKindValues.LLM.value,
    "verify": OpenInferenceSpanKindValues.GUARDRAIL.value,
}


@dataclass(frozen=True, slots=True)
class TracePolicy:
    """What a trace may carry out of the trusted boundary."""

    capture_text: bool = True
    """Export the question and the answer. Off, spans still carry ids, versions and outcomes."""
    redactor: Redactor = redact_identifiers
    max_text_characters: int = 2_000

    def text(self, value: str) -> str | None:
        if not self.capture_text:
            return None
        return self.redactor(value)[: self.max_text_characters]


def build_tracer_provider(
    exporter: SpanExporter | None = None,
    *,
    sample_ratio: float = 1.0,
    service_name: str = PROJECT_NAME,
    project_name: str = PROJECT_NAME,
    processor: Callable[[SpanExporter], Any] = BatchSpanProcessor,
) -> TracerProvider:
    """A provider that samples whole traces and, given an exporter, ships them.

    The provider is returned rather than installed globally: the platform passes it to the
    observer explicitly, so two differently configured services can coexist in one process.
    """
    if not 0.0 <= sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be between 0 and 1")
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": service_name, "openinference.project.name": project_name}
        ),
        sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
    )
    if exporter is not None:
        provider.add_span_processor(processor(exporter))
    return provider


def otlp_exporter(endpoint: str, headers: dict[str, str] | None = None) -> SpanExporter:
    """OTLP over HTTP, the protocol both Phoenix and the OpenTelemetry Collector accept."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=traces_endpoint(endpoint), headers=headers)


def degraded_names(details: Sequence[str]) -> list[str]:
    """Which dependency degraded and how, without the error text that may quote user input."""
    return list(dict.fromkeys(detail.split(":", 1)[0] for detail in details))


def traces_endpoint(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    return base if base.endswith("/v1/traces") else f"{base}/v1/traces"


@dataclass(slots=True)
class _TracedQuery:
    tracer: Tracer
    policy: TracePolicy
    root: Span
    open_spans: dict[str, Span] = field(default_factory=dict)

    @property
    def trace_id(self) -> str | None:
        context = self.root.get_span_context()
        if not context.trace_flags.sampled:
            return None
        return format(context.trace_id, "032x")

    def enter(self, node: str) -> None:
        span = self.tracer.start_span(f"rag.{node}", context=trace.set_span_in_context(self.root))
        span.set_attribute(_KIND, STATE_KINDS.get(node, _CHAIN))
        self.open_spans[node] = span

    def leave(self, node: str, state: WorkflowState, next_node: str) -> None:
        span = self.open_spans.pop(node)
        span.set_attribute("rag.next_state", next_node)
        describe = _DESCRIBERS.get(node)
        if describe is not None:
            describe(span, state, self.policy)
        span.set_status(Status(StatusCode.OK))
        span.end()

    def finish(self, response: QueryResponse) -> None:
        answer = self.policy.text(response.answer)
        if answer is not None:
            self.root.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
        summary = response.trace
        self.root.set_attribute("rag.status", response.status)
        self.root.set_attribute("rag.index_version", summary.index_version or "none")
        self.root.set_attribute("rag.retrieval_strategy", summary.retrieval_strategy)
        self.root.set_attribute("rag.workflow_path", summary.workflow_path)
        self.root.set_attribute("rag.program_revisions", summary.program_revisions)
        self.root.set_attribute("rag.model_route", summary.model_route or "none")
        self.root.set_attribute("rag.repair_attempts", summary.repair_attempts)
        self.root.set_attribute(
            "rag.degraded_dependencies", degraded_names(summary.degraded_dependencies)
        )
        self.root.set_attribute("rag.citation_count", len(response.citations))
        self.root.set_attribute("rag.unsupported_claim_count", len(summary.unsupported_claims))
        self.root.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, summary.prompt_tokens)
        self.root.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, summary.completion_tokens
        )
        self.root.set_status(Status(StatusCode.OK))
        self.root.end()

    def fail(self, error: BaseException) -> None:
        # The message may quote user input, so only the exception type is exported.
        description = type(error).__name__
        for span in self.open_spans.values():
            span.set_status(Status(StatusCode.ERROR, description))
            span.end()
        self.open_spans.clear()
        self.root.set_attribute("rag.status", "error")
        self.root.set_status(Status(StatusCode.ERROR, description))
        self.root.end()


@dataclass(slots=True)
class OpenTelemetryObserver:
    """A `WorkflowObserver` that records each query as an OpenInference trace."""

    provider: TracerProvider
    policy: TracePolicy = field(default_factory=TracePolicy)

    def start(
        self, question: str, access: AccessContext, config: PipelineConfig
    ) -> QueryObservation:
        tracer = self.provider.get_tracer("rag_platform.workflow")
        # A new root context: a query is never parented to whatever span happens to be current.
        root = tracer.start_span(ROOT_SPAN, context=trace.set_span_in_context(trace.INVALID_SPAN))
        root.set_attribute(_KIND, _CHAIN)
        root.set_attribute("rag.tenant_id", access.tenant_id)
        root.set_attribute("rag.access_labels", sorted(access.labels))
        root.set_attribute("rag.config_version", config.version)
        text = self.policy.text(question)
        if text is not None:
            root.set_attribute(SpanAttributes.INPUT_VALUE, text)
        return _TracedQuery(tracer=tracer, policy=self.policy, root=root)


# State descriptions ------------------------------------------------------------


def _classify(span: Span, state: WorkflowState, policy: TracePolicy) -> None:
    if state.classification is not None:
        span.set_attribute("rag.intent", state.classification.intent)


def _rewrite(span: Span, state: WorkflowState, policy: TracePolicy) -> None:
    span.set_attribute("rag.rewritten", state.rewritten != state.question)
    text = policy.text(state.rewritten)
    if text is not None:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, text)


def _decompose(span: Span, state: WorkflowState, policy: TracePolicy) -> None:
    span.set_attribute("rag.subquery_count", len(state.subqueries))


def _retrieve(span: Span, state: WorkflowState, policy: TracePolicy) -> None:
    retrieval = state.retrieval
    if retrieval is None:
        return
    span.set_attribute("rag.retrieval_strategy", retrieval.strategy)
    span.set_attribute("rag.graph_expanded_count", len(retrieval.graph_expanded_chunk_ids))
    span.set_attribute("rag.degraded_dependencies", degraded_names(retrieval.degraded_dependencies))
    for position, item in enumerate(retrieval.chunks):
        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{position}"
        span.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_ID}", item.chunk.chunk_id)
        span.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_SCORE}", item.fused_score)
        span.set_attribute(
            f"{prefix}.{DocumentAttributes.DOCUMENT_METADATA}",
            json.dumps(
                {
                    "index_version": item.chunk.index_version,
                    "source_version_id": item.chunk.source_version_id,
                },
                sort_keys=True,
            ),
        )


def _build_context(span: Span, state: WorkflowState, policy: TracePolicy) -> None:
    bundle = state.bundle
    if bundle is None:
        return
    span.set_attribute("rag.context_chunk_ids", [item.chunk.chunk_id for item in bundle.items])
    span.set_attribute("rag.dropped_chunk_ids", list(bundle.dropped_chunk_ids))
    span.set_attribute("rag.context_characters", bundle.used_characters)
    span.set_attribute("rag.policy_notes", list(bundle.policy_notes))


def _generate(span: Span, state: WorkflowState, policy: TracePolicy) -> None:
    if state.answer is None:
        return
    span.set_attribute("rag.citation_ids", list(state.answer.citation_ids))
    span.set_attribute("rag.sentence_limit", state.sentence_limit)
    text = policy.text(state.answer.text)
    if text is not None:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, text)


def _verify(span: Span, state: WorkflowState, policy: TracePolicy) -> None:
    if state.verification is None:
        return
    span.set_attribute("rag.grounded", state.verification.grounded)
    span.set_attribute("rag.unsupported_claim_count", len(state.verification.unsupported_claims))


def _repair(span: Span, state: WorkflowState, policy: TracePolicy) -> None:
    span.set_attribute("rag.repair_attempt", state.repairs)


_DESCRIBERS: dict[str, Callable[[Span, WorkflowState, TracePolicy], None]] = {
    "classify": _classify,
    "rewrite": _rewrite,
    "decompose": _decompose,
    "retrieve": _retrieve,
    "build_context": _build_context,
    "generate": _generate,
    "verify": _verify,
    "repair": _repair,
}

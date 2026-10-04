"""Tracing behavior, verified against an in-memory span exporter.

The properties that matter are structural (one trace per query, one span per state) and about
what does not leave the boundary: chunk text never, identifiers never, and nothing at all for a
query that was not sampled. tests/integration/test_observability.py proves the OTLP path through
the collector into Phoenix.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from rag_platform.context import ContextBuilder, EvidenceItem
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.programs import ProgramSuite, SynthesizedAnswer
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("openinference.semconv")

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from rag_platform.adapters.tracing import (
    OpenTelemetryObserver,
    TracePolicy,
    build_tracer_provider,
    traces_endpoint,
)

ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))
CHUNK_TEXT = "Annual leave requests use the HR portal. Managers approve within two working days."
CORPUS = [
    DocumentChunk(
        chunk_id="leave",
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text=CHUNK_TEXT,
        source_uri="https://example.test/leave",
        source_title="Leave policy",
        index_version="index-v1",
        source_version_id="sv-leave",
    )
]


@dataclass(slots=True)
class FailingSynthesizer:
    revision: str = "synthesizer-failing"

    def synthesize(
        self, question: str, evidence: Sequence[EvidenceItem], *, sentence_limit: int
    ) -> SynthesizedAnswer:
        raise RuntimeError("model said jane.doe@example.com")


class Exploding:
    """A retriever that fails outside any circuit breaker."""

    def retrieve(self, question: str, access: AccessContext) -> None:
        raise RuntimeError("store unreachable for jane.doe@example.com")


def traced_service(
    *,
    policy: TracePolicy | None = None,
    sample_ratio: float = 1.0,
    programs: ProgramSuite | None = None,
) -> tuple[QueryService, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider(
        exporter, sample_ratio=sample_ratio, processor=SimpleSpanProcessor
    )
    config = PipelineConfig()
    service = QueryService(
        HybridRetriever(InMemoryChunkRepository(CORPUS), config),
        config,
        ContextBuilder(config),
        programs,
        observer=OpenTelemetryObserver(provider, policy or TracePolicy()),
    )
    return service, exporter


def spans_by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {span.name: span for span in exporter.get_finished_spans()}


def exported_text(exporter: InMemorySpanExporter) -> str:
    return " ".join(
        f"{span.name} {dict(span.attributes or {})} {span.status.description}"
        for span in exporter.get_finished_spans()
    )


# Structure ---------------------------------------------------------------------


def test_a_query_is_one_trace_with_a_span_per_workflow_state() -> None:
    service, exporter = traced_service()
    response = service.answer("How do I request annual leave?", ACCESS)

    spans = exporter.get_finished_spans()
    assert len({span.context.trace_id for span in spans}) == 1
    assert [span.name for span in spans if span.name != "rag.query"] == [
        f"rag.{state}" for state in response.trace.workflow_path
    ]


def test_every_state_span_is_a_child_of_the_query_span() -> None:
    service, exporter = traced_service()
    service.answer("How do I request annual leave?", ACCESS)

    spans = spans_by_name(exporter)
    root = spans.pop("rag.query")
    assert root.parent is None
    assert all(
        span.parent is not None and span.parent.span_id == root.context.span_id
        for span in spans.values()
    )


def test_the_response_carries_the_id_of_its_exported_trace() -> None:
    service, exporter = traced_service()
    response = service.answer("How do I request annual leave?", ACCESS)

    root = spans_by_name(exporter)["rag.query"]
    assert response.trace.trace_id == format(root.context.trace_id, "032x")


def test_spans_use_openinference_kinds() -> None:
    service, exporter = traced_service()
    service.answer("How do I request annual leave?", ACCESS)

    kinds = {
        name: (span.attributes or {})["openinference.span.kind"]
        for name, span in spans_by_name(exporter).items()
    }
    assert kinds["rag.query"] == "CHAIN"
    assert kinds["rag.retrieve"] == "RETRIEVER"
    assert kinds["rag.generate"] == "LLM"
    assert kinds["rag.verify"] == "GUARDRAIL"


def test_the_retrieval_span_lists_documents_by_id_score_and_version() -> None:
    service, exporter = traced_service()
    service.answer("How do I request annual leave?", ACCESS)

    attributes = spans_by_name(exporter)["rag.retrieve"].attributes or {}
    assert attributes["retrieval.documents.0.document.id"] == "leave"
    assert float(attributes["retrieval.documents.0.document.score"]) > 0  # type: ignore[arg-type]
    assert "index-v1" in str(attributes["retrieval.documents.0.document.metadata"])


def test_the_query_span_records_the_outcome_and_the_versions_that_produced_it() -> None:
    service, exporter = traced_service()
    service.answer("How do I request annual leave?", ACCESS)

    attributes = spans_by_name(exporter)["rag.query"].attributes or {}
    assert attributes["rag.status"] == "answered"
    assert attributes["rag.tenant_id"] == "tenant-a"
    assert attributes["rag.config_version"] == "pipeline-v1"
    assert attributes["rag.index_version"] == "index-v1"
    assert attributes["rag.model_route"] == "synthesizer-extractive-v1"
    assert int(attributes["llm.token_count.prompt"]) > 0  # type: ignore[arg-type]


def test_a_clarification_is_traced_without_retrieval_spans() -> None:
    service, exporter = traced_service()
    service.answer("What about it?", ACCESS)

    spans = spans_by_name(exporter)
    assert (spans["rag.query"].attributes or {})["rag.status"] == "clarification_needed"
    assert "rag.retrieve" not in spans


# Leakage -----------------------------------------------------------------------


def test_chunk_text_is_never_exported_when_text_capture_is_off() -> None:
    service, exporter = traced_service(policy=TracePolicy(capture_text=False))
    service.answer("How do I request annual leave?", ACCESS)

    text = exported_text(exporter)
    assert "HR portal" not in text
    assert "annual leave" not in text.lower()


def test_retrieval_spans_never_carry_chunk_text() -> None:
    service, exporter = traced_service()
    service.answer("How do I request annual leave?", ACCESS)

    retrieval = dict(spans_by_name(exporter)["rag.retrieve"].attributes or {})
    assert "HR portal" not in str(retrieval)


def test_identifiers_in_the_question_are_redacted_before_export() -> None:
    service, exporter = traced_service()
    service.answer("How do I request annual leave? Reply to jane.doe@example.com", ACCESS)

    text = exported_text(exporter)
    assert "jane.doe@example.com" not in text
    assert "[EMAIL]" in str((spans_by_name(exporter)["rag.query"].attributes or {})["input.value"])


def test_exported_text_is_truncated_to_the_policy_limit() -> None:
    service, exporter = traced_service(policy=TracePolicy(max_text_characters=12))
    service.answer("How do I request annual leave?", ACCESS)

    attributes = spans_by_name(exporter)["rag.query"].attributes or {}
    assert attributes["input.value"] == "How do I req"


def test_an_unsampled_query_exports_nothing_and_has_no_trace_id() -> None:
    service, exporter = traced_service(sample_ratio=0.0)
    response = service.answer("How do I request annual leave?", ACCESS)

    assert exporter.get_finished_spans() == ()
    assert response.trace.trace_id is None
    assert response.status == "answered"


def test_a_sample_ratio_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_ratio"):
        build_tracer_provider(sample_ratio=1.5)


# Failure -----------------------------------------------------------------------


def test_a_failing_state_closes_every_span_as_an_error_without_quoting_the_message() -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider(exporter, processor=SimpleSpanProcessor)
    config = PipelineConfig()
    service = QueryService(
        Exploding(),  # type: ignore[arg-type]
        config,
        ContextBuilder(config),
        observer=OpenTelemetryObserver(provider),
    )

    with pytest.raises(RuntimeError):
        service.answer("How do I request annual leave?", ACCESS)

    spans = spans_by_name(exporter)
    assert spans["rag.retrieve"].status.status_code is StatusCode.ERROR
    assert spans["rag.query"].status.status_code is StatusCode.ERROR
    assert spans["rag.query"].status.description == "RuntimeError"
    assert "jane.doe@example.com" not in exported_text(exporter)


def test_a_degraded_generator_is_visible_on_the_query_span() -> None:
    service, exporter = traced_service(programs=ProgramSuite(synthesizer=FailingSynthesizer()))
    response = service.answer("How do I request annual leave?", ACCESS)

    attributes = spans_by_name(exporter)["rag.query"].attributes or {}
    assert response.status == "insufficient_evidence"
    assert list(attributes["rag.degraded_dependencies"]) == ["generator_failed"]  # type: ignore[arg-type]
    assert "jane.doe@example.com" not in exported_text(exporter)


# Export ------------------------------------------------------------------------


def test_the_traces_path_is_appended_exactly_once() -> None:
    assert traces_endpoint("http://collector:4318") == "http://collector:4318/v1/traces"
    assert traces_endpoint("http://collector:4318/") == "http://collector:4318/v1/traces"
    assert traces_endpoint("http://phoenix:6006/v1/traces") == "http://phoenix:6006/v1/traces"


def test_a_provider_without_an_exporter_still_assigns_trace_ids() -> None:
    provider = build_tracer_provider()
    config = PipelineConfig()
    service = QueryService(
        HybridRetriever(InMemoryChunkRepository(CORPUS), config),
        config,
        observer=OpenTelemetryObserver(provider),
    )
    assert service.answer("How do I request annual leave?", ACCESS).trace.trace_id is not None


def test_the_otlp_exporter_targets_the_traces_path_of_the_configured_endpoint() -> None:
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    from rag_platform.adapters.tracing import otlp_exporter

    exporter = otlp_exporter("http://collector:4318", headers={"x-tenant": "ops"})
    assert exporter._endpoint == "http://collector:4318/v1/traces"  # type: ignore[attr-defined]

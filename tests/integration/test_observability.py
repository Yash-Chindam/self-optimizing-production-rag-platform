"""Tracing and experiment governance against the real services in docker-compose.yml.

Skipped unless the matching environment variable is set. The trace test follows the deployed
path: the application exports OTLP to the OpenTelemetry Collector, the collector forwards to
Phoenix, and the test reads the spans back from Phoenix.
"""

import json
import os
import time
import urllib.request
import uuid
from typing import Any

import pytest

from rag_platform.bootstrap import build_platform
from rag_platform.models import AccessContext, PipelineConfig

OTLP_ENDPOINT = os.environ.get("RAG_OTLP_ENDPOINT")
PHOENIX_URL = os.environ.get("RAG_PHOENIX_URL")
MLFLOW_URI = os.environ.get("RAG_MLFLOW_TRACKING_URI")

ACCESS = AccessContext(tenant_id="tenant-acme", labels=frozenset({"public"}))


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Phoenix may return attributes nested by dotted key; compare on the flat form."""
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            flat.update(flatten(item, f"{name}."))
        else:
            flat[name] = item
    return flat


def phoenix_spans(project: str) -> list[dict[str, Any]]:
    try:
        payload = get_json(f"{PHOENIX_URL}/v1/projects/{project}/spans?limit=100")
    except OSError:
        # The project does not exist until Phoenix has ingested its first span.
        return []
    return list(payload.get("data", []))


def wait_for_trace(project: str, trace_id: str, *, expected: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for _attempt in range(60):
        found = [span for span in phoenix_spans(project) if span["context"]["trace_id"] == trace_id]
        if len(found) >= expected:
            return found
        time.sleep(1)
    raise AssertionError(f"Phoenix has {len(found)} of {expected} spans for trace {trace_id}")


@pytest.mark.integration
@pytest.mark.skipif(not (OTLP_ENDPOINT and PHOENIX_URL), reason="collector and Phoenix not set")
def test_a_query_trace_reaches_phoenix_through_the_collector() -> None:
    from rag_platform.adapters.tracing import (
        OpenTelemetryObserver,
        build_tracer_provider,
        otlp_exporter,
    )

    assert OTLP_ENDPOINT is not None
    project = f"rag-ci-{uuid.uuid4().hex[:8]}"
    provider = build_tracer_provider(otlp_exporter(OTLP_ENDPOINT), project_name=project)
    platform = build_platform(observer=OpenTelemetryObserver(provider))

    response = platform.query_service.answer(
        "How do I request annual leave? Reply to jane.doe@example.com", ACCESS
    )
    assert provider.force_flush(timeout_millis=30_000)
    assert response.trace.trace_id is not None

    expected = len(response.trace.workflow_path) + 1
    spans = wait_for_trace(project, response.trace.trace_id, expected=expected)

    names = {span["name"] for span in spans}
    assert names == {"rag.query"} | {f"rag.{state}" for state in response.trace.workflow_path}
    root = next(span for span in spans if span["name"] == "rag.query")
    attributes = flatten(root["attributes"])
    assert attributes["rag.status"] == "answered"
    assert attributes["rag.tenant_id"] == "tenant-acme"
    assert (attributes.get("openinference.span.kind") or root.get("span_kind")) == "CHAIN"
    assert "jane.doe@example.com" not in json.dumps(spans)
    provider.shutdown()


@pytest.fixture
def mlflow_client() -> Any:
    from rag_platform.adapters.mlflow_tracking import client_for

    assert MLFLOW_URI is not None
    return client_for(MLFLOW_URI)


@pytest.mark.integration
@pytest.mark.skipif(not MLFLOW_URI, reason="RAG_MLFLOW_TRACKING_URI not set")
def test_an_optimization_is_recorded_on_the_mlflow_server(mlflow_client: Any) -> None:
    from pathlib import Path

    from rag_platform.adapters.mlflow_tracking import MlflowRunLogger
    from rag_platform.cli import optimize_dataset

    dataset = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "cases.jsonl"
    runs = optimize_dataset(build_platform(), dataset, max_candidates=2)
    experiment = f"rag-ci-{uuid.uuid4().hex[:8]}"

    mapping = MlflowRunLogger(mlflow_client, experiment=experiment).log_optimization(runs)

    for run in runs:
        logged = mlflow_client.get_run(mapping[run.run_id])
        assert logged.data.tags["rag.promotion"] == run.promotion
        assert logged.data.tags["rag.dataset_revision"] == run.dataset_revision
        assert logged.data.metrics["pass_rate"] == run.summary.pass_rate
        artifacts = [item.path for item in mlflow_client.list_artifacts(logged.info.run_id)]
        assert artifacts == ["pipeline_config.json"]


@pytest.mark.integration
@pytest.mark.skipif(not MLFLOW_URI, reason="RAG_MLFLOW_TRACKING_URI not set")
def test_promotion_and_rollback_survive_in_the_mlflow_registry(mlflow_client: Any) -> None:
    from rag_platform.adapters.mlflow_tracking import MlflowConfigRegistry

    baseline = PipelineConfig()
    candidate = baseline.model_copy(update={"version": "pipeline-v1-cand-1", "final_top_k": 5})
    name = f"rag-ci-config-{uuid.uuid4().hex[:8]}"

    registry = MlflowConfigRegistry(mlflow_client, name=name)
    assert registry.ensure(baseline) == baseline
    registry.register(candidate)
    assert registry.promote(candidate.version) == candidate

    # A second registry object stands in for a restarted process.
    restarted = MlflowConfigRegistry(mlflow_client, name=name)
    assert restarted.ensure(baseline) == candidate
    assert restarted.rollback() == baseline
    assert MlflowConfigRegistry(mlflow_client, name=name).active() == baseline

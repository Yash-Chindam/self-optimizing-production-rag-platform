"""Prefect flows, run for real against Prefect's ephemeral test API.

The flows hold no logic of their own, so what is tested is the orchestration: that a flow run
produces what a direct call would, that the gate fails the run, that retries apply to transient
failures only, and that Prefect recorded the run.
"""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag_platform.bootstrap import ACME_HANDBOOK, Platform, build_platform
from rag_platform.cli import evaluate_dataset
from rag_platform.events import InMemoryEventPublisher
from rag_platform.ingestion import IngestionValidationError
from rag_platform.models import AccessContext, SourceRegistration

pytest.importorskip("prefect")
pytest.importorskip("ragas")
pytest.importorskip("deepeval")

import rag_platform.adapters.prefect_flows as flows
from rag_platform.adapters.prefect_flows import (
    GateFailedError,
    SourceDocument,
    best_candidate,
    evaluation_flow,
    ingestion_flow,
    optimization_flow,
)

DATASET = str(Path(__file__).resolve().parents[2] / "data" / "evaluation" / "cases.jsonl")

SECURITY = SourceDocument(
    registration=SourceRegistration(
        source_id="acme-security",
        tenant_id="tenant-acme",
        owner="security",
        source_uri="https://knowledge.example/acme/security",
        source_title="ACME Security Standard",
    ),
    content=(
        "# ACME Security Standard\n\n## Laptop encryption\n\n"
        "Every ACME laptop must use full disk encryption before it leaves the office.\n"
    ),
)


@pytest.fixture(scope="module", autouse=True)
def prefect_api() -> Iterator[None]:
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness():
        yield


def flow_runs() -> list[tuple[str, str]]:
    from prefect.client.orchestration import get_client

    async def read() -> list[tuple[str, str]]:
        async with get_client() as client:
            flows_by_id = {item.id: item.name for item in await client.read_flows()}
            return [
                (flows_by_id[run.flow_id], str(run.state_name))
                for run in await client.read_flow_runs()
            ]

    return asyncio.run(read())


def failing_dataset(tmp_path: Path) -> str:
    path = tmp_path / "cases.jsonl"
    row = {
        "case_id": "expects-the-impossible",
        "question": "How do I request annual leave?",
        "tenant_id": "tenant-acme",
        "required_evidence_chunk_ids": ["no-such-chunk"],
        "reviewer": "people-operations-review",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return str(path)


# Ingestion ---------------------------------------------------------------------


def test_the_ingestion_flow_activates_an_index_the_platform_can_answer_from() -> None:
    platform = build_platform()
    [ingested] = ingestion_flow([SECURITY], platform=platform)

    assert ingested.reused_existing_source_version is False
    answer = platform.query_service.answer(
        "What must every laptop use?", AccessContext(tenant_id="tenant-acme")
    )
    assert answer.status == "answered"
    assert "full disk encryption" in answer.answer
    assert ("rag-ingestion", "Completed") in flow_runs()


def test_a_transient_ingestion_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = build_platform()
    real = platform.ingestion.ingest
    attempts: list[int] = []

    def flaky(registration: SourceRegistration, content: str) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("store unreachable")
        return real(registration, content)

    monkeypatch.setattr(type(platform.ingestion), "ingest", lambda _self, r, c: flaky(r, c))
    [ingested] = ingestion_flow([SECURITY], platform=platform)

    assert len(attempts) == 2
    assert ingested.chunk_count > 0


def test_a_validation_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = build_platform()
    attempts: list[int] = []

    def rejecting(_self: Any, registration: SourceRegistration, content: str) -> Any:
        attempts.append(1)
        raise IngestionValidationError("source produced no retrievable chunk")

    monkeypatch.setattr(type(platform.ingestion), "ingest", rejecting)
    with pytest.raises(IngestionValidationError):
        ingestion_flow([SECURITY], platform=platform)
    assert len(attempts) == 1


def test_the_flow_uses_the_configured_platform_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = build_platform(seed_demo_sources=False)
    monkeypatch.setattr(flows, "PLATFORM_FACTORY", lambda: platform)

    handbook = SourceDocument(
        registration=SourceRegistration(
            source_id="acme-handbook",
            tenant_id="tenant-acme",
            owner="people-operations",
            source_uri="https://knowledge.example/acme/handbook",
            source_title="ACME Employee Handbook",
        ),
        content=ACME_HANDBOOK,
    )
    ingestion_flow([handbook])

    assert platform.catalog.active_index_version("tenant-acme") is not None


# Evaluation --------------------------------------------------------------------


def test_the_evaluation_flow_agrees_with_a_direct_evaluation() -> None:
    platform = build_platform()
    outcome = evaluation_flow(DATASET, platform=platform)
    direct = evaluate_dataset(platform, Path(DATASET))

    assert outcome.dataset_revision == direct.dataset_revision
    assert outcome.passed_count == direct.summary.passed_count == outcome.case_count
    assert outcome.gate_failures == ()
    assert (outcome.ragas_context_recall, outcome.deepeval_failures) == (1.0, 0)
    assert ("rag-evaluation", "Completed") in flow_runs()


def test_a_failed_gate_fails_the_flow_run(tmp_path: Path) -> None:
    with pytest.raises(GateFailedError) as raised:
        evaluation_flow(failing_dataset(tmp_path), platform=build_platform())

    message = str(raised.value)
    assert "expects-the-impossible" in message
    assert "RAGAS context recall" in message
    assert "DeepEval Evidence recall" in message
    assert ("rag-evaluation", "Failed") in flow_runs()


def test_the_gate_can_be_reported_without_being_enforced(tmp_path: Path) -> None:
    outcome = evaluation_flow(
        failing_dataset(tmp_path), enforce_gate=False, platform=build_platform()
    )
    assert outcome.passed_count == 0
    assert len(outcome.gate_failures) >= 3


def test_the_evaluation_flow_announces_its_result(monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = InMemoryEventPublisher()
    monkeypatch.setattr(flows, "EVENT_PUBLISHER", publisher)

    outcome = evaluation_flow(DATASET, platform=build_platform())

    [event] = publisher.events
    assert event.event_type == "evaluation.completed"
    assert event.subject == outcome.dataset_revision
    assert event.attributes["passed"] == str(outcome.passed_count)


# Optimization ------------------------------------------------------------------


def mlflow_uri(tmp_path: Path) -> str:
    pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    from rag_platform.adapters.mlflow_tracking import client_for

    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    client = client_for(uri)
    for name in ("rag-platform", "rag-pipeline-config"):
        client.create_experiment(name, artifact_location=(tmp_path / name).as_uri())
    return uri


def test_the_optimization_flow_canaries_the_best_candidate() -> None:
    platform: Platform = build_platform()
    outcome = optimization_flow(DATASET, max_candidates=4, platform=platform)

    assert len(outcome.candidates) == 4
    assert outcome.decided is not None
    assert outcome.decided.promotion in {"promoted", "rolled_back"}
    assert outcome.active_config_version == "pipeline-v1"
    assert ("rag-optimization", "Completed") in flow_runs()


def test_a_promoted_candidate_becomes_the_registry_champion(tmp_path: Path) -> None:
    from rag_platform.adapters.mlflow_tracking import MlflowConfigRegistry, client_for

    uri = mlflow_uri(tmp_path)
    outcome = optimization_flow(
        DATASET, max_candidates=3, mlflow_tracking_uri=uri, promote=True, platform=build_platform()
    )

    client = client_for(uri)
    registry = MlflowConfigRegistry(client)
    assert outcome.decided is not None
    if outcome.decided.promotion == "promoted":
        assert registry.active() == outcome.decided.config
        assert registry.rollback().version == "pipeline-v1"
    else:
        assert registry.active().version == "pipeline-v1"
    assert outcome.active_config_version in {"pipeline-v1", outcome.decided.config.version}

    experiment = client.get_experiment_by_name("rag-platform")
    dispositions = {
        run.data.tags["rag.run_id"]: run.data.tags["rag.promotion"]
        for run in client.search_runs([experiment.experiment_id])
    }
    assert len(dispositions) == 3
    assert dispositions[outcome.decided.run_id] == outcome.decided.promotion


def test_without_promotion_the_champion_is_left_alone(tmp_path: Path) -> None:
    from rag_platform.adapters.mlflow_tracking import MlflowConfigRegistry, client_for

    uri = mlflow_uri(tmp_path)
    outcome = optimization_flow(
        DATASET, max_candidates=2, mlflow_tracking_uri=uri, platform=build_platform()
    )
    assert outcome.active_config_version == "pipeline-v1"
    assert MlflowConfigRegistry(client_for(uri)).active().version == "pipeline-v1"


def test_nothing_is_canaried_when_there_is_no_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_platform.optimization import CandidateSpace

    empty = CandidateSpace(
        dense_top_k=(),
        sparse_top_k=(),
        final_top_k=(),
        fusion_method=(),
        graph_expansion_depth=(),
        rerank_candidate_count=(),
        context_character_budget=(),
        answer_sentence_limit=(),
    )
    monkeypatch.setattr(flows, "CandidateSpace", lambda: empty)

    outcome = optimization_flow(DATASET, platform=build_platform())

    assert outcome.candidates == ()
    assert outcome.decided is None
    assert outcome.active_config_version == "pipeline-v1"
    assert best_candidate([]) is None

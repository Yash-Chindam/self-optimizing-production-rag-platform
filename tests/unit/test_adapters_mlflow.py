"""MLflow governance against a real MLflow tracking store and model registry on SQLite.

No fake client: the adapter is thin, so the useful question is whether MLflow accepts what it
sends and gives it back. tests/integration/test_observability.py repeats the round trip against
an MLflow server over HTTP.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from rag_platform.config_registry import ConfigPromotionError, UnknownConfigVersionError
from rag_platform.evaluation import CaseResult, EvaluationReport
from rag_platform.models import (
    CandidateRun,
    ConstraintViolation,
    EvaluationSummary,
    PipelineConfig,
)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
pytest.importorskip("mlflow")
pytest.importorskip("sqlalchemy")

from rag_platform.adapters.mlflow_tracking import (
    CHAMPION,
    MlflowConfigRegistry,
    MlflowRunLogger,
    client_for,
    config_params,
    summary_metrics,
)

BASELINE = PipelineConfig()
CANDIDATE = BASELINE.model_copy(update={"version": "pipeline-v1-cand-1", "final_top_k": 5})
OTHER = BASELINE.model_copy(update={"version": "pipeline-v1-cand-2", "dense_top_k": 10})

SUMMARY = EvaluationSummary(
    case_count=4,
    passed_count=3,
    mean_recall_at_k=0.75,
    mean_reciprocal_rank=0.5,
    grounded_rate=1.0,
    authorization_violations=0,
    p50_latency_ms=1.5,
    p95_latency_ms=4.0,
)


@pytest.fixture
def client(tmp_path: Path) -> Any:
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    tracking = client_for(uri)
    # Artifacts go beside the database rather than into the working directory.
    tracking.create_experiment("rag-platform", artifact_location=(tmp_path / "artifacts").as_uri())
    return tracking


def candidate_run(
    config: PipelineConfig = CANDIDATE,
    *,
    violations: tuple[ConstraintViolation, ...] = (),
    promotion: Any = "pareto_optimal",
) -> CandidateRun:
    return CandidateRun(
        run_id=f"run-{config.version}",
        baseline_config_version=BASELINE.version,
        config=config,
        config_diff=("final_top_k: 4 -> 5",),
        dataset_revision="59deebbdcfb2",
        program_revisions=("programs-v1", "classifier-v1"),
        summary=SUMMARY,
        constraint_violations=violations,
        promotion=promotion,
    )


def report() -> EvaluationReport:
    failed = CaseResult(
        case_id="acme-leave-alias",
        passed=False,
        status="insufficient_evidence",
        recall_at_k=0.0,
        reciprocal_rank=0.0,
        grounded=True,
        authorization_violations=(),
        latency_ms=2.0,
        failure_category="retrieval",
        notes=(),
    )
    return EvaluationReport(
        dataset_revision="59deebbdcfb2",
        config_version=BASELINE.version,
        program_revisions=("programs-v1",),
        summary=SUMMARY,
        results=(failed,),
    )


# Run logging -------------------------------------------------------------------


def test_a_candidate_is_logged_with_its_configuration_metrics_and_evidence(client: Any) -> None:
    violation = ConstraintViolation(constraint="latency", detail="p95 latency too high")
    logger = MlflowRunLogger(client)
    run_id = logger.log_candidate(candidate_run(violations=(violation,), promotion="rejected"))

    logged = client.get_run(run_id)
    assert logged.data.params["final_top_k"] == "5"
    assert logged.data.metrics["pass_rate"] == 0.75
    assert logged.data.metrics["p95_latency_ms"] == 4.0
    assert logged.data.tags["rag.promotion"] == "rejected"
    assert logged.data.tags["rag.dataset_revision"] == "59deebbdcfb2"
    assert logged.data.tags["rag.baseline_config_version"] == "pipeline-v1"
    assert logged.data.tags["rag.config_diff"] == "final_top_k: 4 -> 5"
    assert json.loads(logged.data.tags["rag.constraint_violations"]) == [
        {"constraint": "latency", "detail": "p95 latency too high"}
    ]
    assert logged.info.status == "FINISHED"


def test_the_exact_configuration_is_kept_as_a_run_artifact(client: Any, tmp_path: Path) -> None:
    run_id = MlflowRunLogger(client).log_candidate(candidate_run())
    downloaded = client.download_artifacts(run_id, "pipeline_config.json", str(tmp_path))
    stored = PipelineConfig.model_validate_json(Path(downloaded).read_text(encoding="utf-8"))
    assert stored == CANDIDATE


def test_every_candidate_of_an_optimization_is_logged_including_the_rejected(client: Any) -> None:
    runs = [candidate_run(CANDIDATE), candidate_run(OTHER, promotion="rejected")]
    mapping = MlflowRunLogger(client).log_optimization(runs)

    assert set(mapping) == {"run-pipeline-v1-cand-1", "run-pipeline-v1-cand-2"}
    experiment = client.get_experiment_by_name("rag-platform")
    assert len(client.search_runs([experiment.experiment_id])) == 2


def test_a_canary_decision_updates_the_recorded_disposition(client: Any) -> None:
    logger = MlflowRunLogger(client)
    run = candidate_run()
    run_id = logger.log_candidate(run)

    logger.record_promotion(run_id, run.with_promotion("rolled_back"))

    assert client.get_run(run_id).data.tags["rag.promotion"] == "rolled_back"


def test_an_evaluation_records_which_cases_failed_and_why(client: Any) -> None:
    run_id = MlflowRunLogger(client).log_evaluation(report(), BASELINE)

    tags = client.get_run(run_id).data.tags
    assert tags["rag.kind"] == "evaluation"
    assert tags["rag.failed_case_ids"] == "acme-leave-alias"
    assert tags["rag.failure_categories"] == "retrieval"


def test_the_experiment_is_created_on_first_use(client: Any, tmp_path: Path) -> None:
    logger = MlflowRunLogger(
        client, experiment="rag-nightly", artifact_location=(tmp_path / "nightly").as_uri()
    )
    logger.log_candidate(candidate_run())
    assert client.get_experiment_by_name("rag-nightly") is not None


def test_metrics_and_parameters_cover_the_whole_summary_and_configuration() -> None:
    assert set(summary_metrics(SUMMARY)) >= {"pass_rate", "grounded_rate", "p95_latency_ms"}
    assert config_params(BASELINE)["version"] == "pipeline-v1"
    assert set(config_params(BASELINE)) == set(PipelineConfig.model_fields)


# Registry ----------------------------------------------------------------------


def test_the_first_configuration_becomes_the_champion(client: Any) -> None:
    registry = MlflowConfigRegistry(client)
    assert registry.ensure(BASELINE) == BASELINE
    assert registry.active() == BASELINE
    assert str(client.get_model_version_by_alias(registry.name, CHAMPION).version) == "1"


def test_an_existing_champion_survives_a_restart(client: Any) -> None:
    registry = MlflowConfigRegistry(client)
    registry.ensure(BASELINE)
    registry.register(CANDIDATE)
    registry.promote(CANDIDATE.version)

    # A new process constructs a new registry and offers its defaults again.
    assert MlflowConfigRegistry(client).ensure(BASELINE) == CANDIDATE


def test_promotion_moves_the_champion_and_rollback_restores_it(client: Any) -> None:
    registry = MlflowConfigRegistry(client)
    registry.ensure(BASELINE)
    registry.register(CANDIDATE)
    registry.register(OTHER)

    assert registry.promote(CANDIDATE.version) == CANDIDATE
    assert registry.promote(OTHER.version) == OTHER
    assert registry.rollback() == CANDIDATE
    assert registry.active() == CANDIDATE
    assert registry.rollback() == BASELINE
    assert registry.active() == BASELINE


def test_rollback_without_history_is_refused(client: Any) -> None:
    registry = MlflowConfigRegistry(client)
    registry.ensure(BASELINE)
    with pytest.raises(ConfigPromotionError):
        registry.rollback()


def test_promoting_the_active_configuration_adds_no_history(client: Any) -> None:
    registry = MlflowConfigRegistry(client)
    registry.ensure(BASELINE)
    registry.promote(BASELINE.version)
    with pytest.raises(ConfigPromotionError):
        registry.rollback()


def test_an_unregistered_configuration_cannot_be_promoted(client: Any) -> None:
    registry = MlflowConfigRegistry(client)
    registry.ensure(BASELINE)
    with pytest.raises(UnknownConfigVersionError):
        registry.promote("pipeline-v9")


def test_registering_a_version_twice_keeps_one_model_version(client: Any) -> None:
    registry = MlflowConfigRegistry(client)
    registry.ensure(BASELINE)
    registry.register(CANDIDATE)
    registry.register(CANDIDATE)
    assert registry.versions() == (BASELINE, CANDIDATE)


def test_a_registered_configuration_links_back_to_the_run_that_evaluated_it(client: Any) -> None:
    run_id = MlflowRunLogger(client).log_candidate(candidate_run())
    registry = MlflowConfigRegistry(client)
    registry.register(CANDIDATE, mlflow_run_id=run_id)

    [version] = client.search_model_versions(f"name='{registry.name}'")
    assert version.run_id == run_id
    assert version.source == f"runs:/{run_id}/pipeline_config.json"


def test_nothing_is_active_before_anything_is_promoted(client: Any) -> None:
    registry = MlflowConfigRegistry(client)
    registry.register(CANDIDATE)
    with pytest.raises(ConfigPromotionError):
        registry.active()

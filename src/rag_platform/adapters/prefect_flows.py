"""Prefect orchestration of ingestion, evaluation and optimization (specification sections 5-7, 12).

The flows add scheduling, retries and a run history around steps the platform already
implements; they contain no business logic of their own. Each flow is a thin sequence of tasks
over `IngestionPipeline`, `Evaluator`, the framework metrics and `OptimizationRun`, so a flow
run and a direct call cannot disagree.

Flows take plain, serializable parameters (paths, URIs, source documents). The platform they
act on comes from `PLATFORM_FACTORY`, which a deployment points at its configured platform, so
a scheduled run needs no live object as a parameter. A caller in the same process may pass a
platform directly instead.

A flow fails, visibly, when its gate fails: an evaluation that does not clear `ReleaseGate` and
the framework gate raises `GateFailedError`, and an optimization only promotes a candidate that
survived its canary.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prefect import flow, task
from pydantic import BaseModel

from rag_platform.adapters.eval_frameworks import (
    DeepEvalResult,
    RagasScores,
    deepeval_regression,
    framework_gate,
    ragas_scores,
)
from rag_platform.bootstrap import Platform
from rag_platform.evaluation import (
    EvaluationReport,
    Evaluator,
    ReleaseGate,
    dataset_revision,
    load_cases,
)
from rag_platform.events import EventPublisher, evaluation_completed, publish_safely
from rag_platform.ingestion import IngestionValidationError
from rag_platform.models import CandidateRun, IngestionReport, SourceRegistration
from rag_platform.optimization import CandidateSpace, OptimizationRun
from rag_platform.runtime import build_platform_from_settings
from rag_platform.settings import Settings


def platform_from_environment() -> Platform:
    """The platform a worker acts on: assembled from its own environment."""
    return build_platform_from_settings(Settings.from_env(os.environ))


PLATFORM_FACTORY: Callable[[], Platform] = platform_from_environment
EVENT_PUBLISHER: EventPublisher | None = None


class GateFailedError(RuntimeError):
    """A release gate rejected the run. The message lists every finding."""


class SourceDocument(BaseModel):
    registration: SourceRegistration
    content: str


class EvaluationOutcome(BaseModel):
    dataset_revision: str
    config_version: str
    case_count: int
    passed_count: int
    grounded_rate: float
    authorization_violations: int
    ragas_context_recall: float
    ragas_context_precision: float
    deepeval_failures: int
    gate_failures: tuple[str, ...]
    mlflow_run_id: str | None = None


class OptimizationOutcome(BaseModel):
    baseline_config_version: str
    candidates: tuple[CandidateRun, ...]
    decided: CandidateRun | None
    """The canaried candidate with its final disposition, or None if nothing was approved."""
    active_config_version: str


# Ingestion ---------------------------------------------------------------------


def _retryable(_task: Any, _task_run: Any, state: Any) -> bool:
    """Retry transient failures. Content that fails validation will fail again."""
    try:
        state.result()
    except IngestionValidationError:
        return False
    except Exception:
        return True
    return True


@task(name="ingest-source", retries=2, retry_delay_seconds=1, retry_condition_fn=_retryable)
def ingest_source(platform: Platform, document: SourceDocument) -> IngestionReport:
    return platform.ingestion.ingest(document.registration, document.content).report


@flow(name="rag-ingestion", validate_parameters=False)
def ingestion_flow(documents: list[SourceDocument], platform: Any = None) -> list[IngestionReport]:
    """Ingest sources one after another: each builds on the index version before it."""
    resolved: Platform = platform or PLATFORM_FACTORY()
    return [ingest_source(resolved, document) for document in documents]


# Evaluation --------------------------------------------------------------------


@task(name="run-evaluation-set")
def run_evaluation(platform: Platform, dataset: str) -> EvaluationReport:
    path = Path(dataset)
    evaluator = Evaluator(platform.query_service, dataset_revision=dataset_revision(path))
    return evaluator.evaluate(load_cases(path))


@task(name="score-with-frameworks")
def score_with_frameworks(
    report: EvaluationReport, dataset: str
) -> tuple[RagasScores, DeepEvalResult]:
    cases = load_cases(Path(dataset))
    return ragas_scores(report, cases), deepeval_regression(report, cases)


@task(name="record-evaluation")
def record_evaluation(platform: Platform, report: EvaluationReport, tracking_uri: str) -> str:
    from rag_platform.adapters.mlflow_tracking import MlflowRunLogger, client_for

    return MlflowRunLogger(client_for(tracking_uri)).log_evaluation(report, platform.config)


@flow(name="rag-evaluation", validate_parameters=False)
def evaluation_flow(
    dataset: str,
    mlflow_tracking_uri: str | None = None,
    enforce_gate: bool = True,
    platform: Any = None,
) -> EvaluationOutcome:
    """Deterministic evaluation, then RAGAS and DeepEval on the same responses, then the gate."""
    resolved: Platform = platform or PLATFORM_FACTORY()
    report = run_evaluation(resolved, dataset)
    ragas, deepeval = score_with_frameworks(report, dataset)
    failures = ReleaseGate().failures(report) + framework_gate(ragas, deepeval)
    run_id = (
        record_evaluation(resolved, report, mlflow_tracking_uri) if mlflow_tracking_uri else None
    )
    summary = report.summary
    publish_safely(
        EVENT_PUBLISHER,
        evaluation_completed(
            report.dataset_revision,
            report.config_version,
            passed=summary.passed_count,
            total=summary.case_count,
            authorization_violations=summary.authorization_violations,
        ),
    )
    outcome = EvaluationOutcome(
        dataset_revision=report.dataset_revision,
        config_version=report.config_version,
        case_count=summary.case_count,
        passed_count=summary.passed_count,
        grounded_rate=summary.grounded_rate,
        authorization_violations=summary.authorization_violations,
        ragas_context_recall=ragas.context_recall,
        ragas_context_precision=ragas.context_precision,
        deepeval_failures=len(deepeval.failures),
        gate_failures=failures,
        mlflow_run_id=run_id,
    )
    if failures and enforce_gate:
        raise GateFailedError("; ".join(failures))
    return outcome


# Optimization ------------------------------------------------------------------


@task(name="evaluate-candidates")
def evaluate_candidates(
    platform: Platform, dataset: str, max_candidates: int
) -> list[CandidateRun]:
    path = Path(dataset)
    run = OptimizationRun(platform.optimization, dataset_revision=dataset_revision(path))
    return run.run(
        platform.config, load_cases(path), CandidateSpace(), max_candidates=max_candidates
    )


@task(name="canary-candidate")
def canary_candidate(platform: Platform, dataset: str, candidate: CandidateRun) -> CandidateRun:
    path = Path(dataset)
    run = OptimizationRun(platform.optimization, dataset_revision=dataset_revision(path))
    return run.canary(candidate, load_cases(path), platform.config)


def best_candidate(runs: list[CandidateRun]) -> CandidateRun | None:
    """The Pareto-optimal candidate with the best pass rate, then the lowest latency."""
    front = [run for run in runs if run.promotion == "pareto_optimal"]
    if not front:
        return None
    return min(
        front, key=lambda run: (-run.summary.pass_rate, run.summary.p95_latency_ms, run.run_id)
    )


@flow(name="rag-optimization", validate_parameters=False)
def optimization_flow(
    dataset: str,
    max_candidates: int = 8,
    mlflow_tracking_uri: str | None = None,
    promote: bool = False,
    platform: Any = None,
) -> OptimizationOutcome:
    """Section 12's loop: evaluate candidates, canary the best, promote or roll back."""
    resolved: Platform = platform or PLATFORM_FACTORY()
    runs = evaluate_candidates(resolved, dataset, max_candidates)
    chosen = best_candidate(runs)
    decided = canary_candidate(resolved, dataset, chosen) if chosen is not None else None
    active = resolved.config.version

    if mlflow_tracking_uri:
        from rag_platform.adapters.mlflow_tracking import (
            MlflowConfigRegistry,
            MlflowRunLogger,
            client_for,
        )

        client = client_for(mlflow_tracking_uri)
        logger = MlflowRunLogger(client)
        mapping = logger.log_optimization(runs)
        registry = MlflowConfigRegistry(client)
        active = registry.ensure(resolved.config).version
        if decided is not None:
            logger.record_promotion(mapping[decided.run_id], decided)
            if promote and decided.promotion == "promoted":
                registry.register(decided.config, mlflow_run_id=mapping[decided.run_id])
                active = registry.promote(decided.config.version).version

    return OptimizationOutcome(
        baseline_config_version=resolved.config.version,
        candidates=tuple(runs),
        decided=decided,
        active_config_version=active,
    )

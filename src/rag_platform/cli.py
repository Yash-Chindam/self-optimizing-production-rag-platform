"""Command line entry point.

`serve` runs the API. `evaluate` and `optimize` are the offline half of the platform
(specification sections 12 and 16): they run outside production, against a versioned evaluation
set, and `evaluate` is the release gate CI and the DVC pipeline both call.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_platform.bootstrap import Platform, build_platform
from rag_platform.evaluation import (
    EvaluationReport,
    Evaluator,
    ReleaseGate,
    dataset_revision,
    load_cases,
    write_report,
)
from rag_platform.models import CandidateRun
from rag_platform.optimization import CandidateSpace, OptimizationRun

DEFAULT_DATASET = Path("data/evaluation/cases.jsonl")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = arguments.command or "serve"
    if command == "serve":
        return _serve(arguments)
    if command == "evaluate":
        return _evaluate(arguments)
    return _optimize(arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-platform", description=__doc__)
    commands = parser.add_subparsers(dest="command")

    serve = commands.add_parser("serve", help="run the API and the demo UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    evaluate = commands.add_parser("evaluate", help="run the evaluation set and apply the gate")
    _offline_arguments(evaluate)
    evaluate.add_argument("--metrics", type=Path, default=Path("reports/evaluation/metrics.json"))
    evaluate.add_argument("--results", type=Path, default=Path("reports/evaluation/results.jsonl"))
    evaluate.add_argument("--min-pass-rate", type=float, default=1.0)
    evaluate.add_argument("--min-grounded-rate", type=float, default=1.0)

    optimize = commands.add_parser("optimize", help="evaluate bounded candidates offline")
    _offline_arguments(optimize)
    optimize.add_argument("--max-candidates", type=int, default=8)
    optimize.add_argument(
        "--output", type=Path, default=Path("reports/optimization/candidates.json")
    )
    return parser


def _offline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="record the run in MLflow (requires the governance extra)",
    )


def _serve(arguments: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "rag_platform.api:app",
        host=getattr(arguments, "host", "127.0.0.1"),
        port=getattr(arguments, "port", 8000),
        reload=False,
    )
    return 0


def _evaluate(arguments: argparse.Namespace) -> int:
    platform = build_platform()
    report = evaluate_dataset(platform, arguments.dataset)
    write_report(report, metrics=arguments.metrics, results=arguments.results)
    if arguments.mlflow_tracking_uri:
        from rag_platform.adapters.mlflow_tracking import MlflowRunLogger, client_for

        logger = MlflowRunLogger(client_for(arguments.mlflow_tracking_uri))
        logger.log_evaluation(report, platform.config)

    summary = report.summary
    print(
        f"dataset {report.dataset_revision}: {summary.passed_count}/{summary.case_count} passed, "
        f"grounded {summary.grounded_rate:.2f}, "
        f"authorization violations {summary.authorization_violations}"
    )
    gate = ReleaseGate(
        min_pass_rate=arguments.min_pass_rate, min_grounded_rate=arguments.min_grounded_rate
    )
    failures = gate.failures(report)
    for failure in failures:
        print(f"gate failed: {failure}", file=sys.stderr)
    return 1 if failures else 0


def _optimize(arguments: argparse.Namespace) -> int:
    platform = build_platform()
    runs = optimize_dataset(platform, arguments.dataset, max_candidates=arguments.max_candidates)
    if arguments.mlflow_tracking_uri:
        from rag_platform.adapters.mlflow_tracking import MlflowRunLogger, client_for

        MlflowRunLogger(client_for(arguments.mlflow_tracking_uri)).log_optimization(runs)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps([run.model_dump(mode="json") for run in runs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for run in runs:
        print(f"{run.run_id}: {run.promotion} ({'; '.join(run.config_diff)})")
    return 0


def evaluate_dataset(platform: Platform, dataset: Path) -> EvaluationReport:
    evaluator = Evaluator(platform.query_service, dataset_revision=dataset_revision(dataset))
    return evaluator.evaluate(load_cases(dataset))


def optimize_dataset(
    platform: Platform, dataset: Path, *, max_candidates: int
) -> list[CandidateRun]:
    run = OptimizationRun(platform.optimization, dataset_revision=dataset_revision(dataset))
    return run.run(
        platform.config, load_cases(dataset), CandidateSpace(), max_candidates=max_candidates
    )

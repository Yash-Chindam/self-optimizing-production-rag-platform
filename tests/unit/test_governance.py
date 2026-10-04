"""The versioned evaluation set, the release gate, the offline commands and the observer hooks."""

import json
from pathlib import Path

import pytest

from rag_platform.bootstrap import build_platform
from rag_platform.cli import evaluate_dataset, main, optimize_dataset
from rag_platform.context import ContextBuilder
from rag_platform.evaluation import (
    ReleaseGate,
    dataset_revision,
    load_cases,
    reproducible_metrics,
    write_report,
)
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig, QueryResponse
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService
from rag_platform.workflow import QueryObservation, WorkflowState, estimate_tokens

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "evaluation" / "cases.jsonl"
ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))

FAILING_CASE = {
    "case_id": "expects-the-impossible",
    "question": "How do I request annual leave?",
    "tenant_id": "tenant-acme",
    "required_evidence_chunk_ids": ["no-such-chunk"],
    "reviewer": "people-operations-review",
}


def write_dataset(path: Path, *rows: dict[str, object]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


# Dataset -----------------------------------------------------------------------


def test_the_committed_evaluation_set_loads_and_covers_every_outcome() -> None:
    cases = load_cases(DATASET)
    assert len(cases) >= 10
    assert {case.expected_status for case in cases} == {
        "answered",
        "insufficient_evidence",
        "clarification_needed",
    }
    assert {case.tenant_id for case in cases} >= {"tenant-acme", "tenant-globex"}
    assert any(case.forbidden_chunk_ids for case in cases)


def test_the_dataset_revision_is_the_hash_dvc_recorded() -> None:
    """dvc.lock pins the dataset by MD5; the platform must report that same revision."""
    lock = (ROOT / "dvc.lock").read_text(encoding="utf-8")
    assert f"md5: {dataset_revision(DATASET)}" in lock


def test_the_revision_changes_with_the_content(tmp_path: Path) -> None:
    first = write_dataset(tmp_path / "a.jsonl", FAILING_CASE)
    second = write_dataset(tmp_path / "b.jsonl", {**FAILING_CASE, "difficulty": "hard"})
    assert dataset_revision(first) != dataset_revision(second)
    assert dataset_revision(first) == dataset_revision(first)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n" + json.dumps(FAILING_CASE) + "\n\n", encoding="utf-8")
    assert [case.case_id for case in load_cases(path)] == ["expects-the-impossible"]


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "cases.jsonl", FAILING_CASE, FAILING_CASE)
    with pytest.raises(ValueError, match="expects-the-impossible"):
        load_cases(path)


# Reports and the gate -----------------------------------------------------------


def test_the_demo_platform_passes_its_own_evaluation_set() -> None:
    report = evaluate_dataset(build_platform(), DATASET)
    assert report.failures() == ()
    assert report.summary.authorization_violations == 0
    assert ReleaseGate().failures(report) == ()


def test_the_written_report_is_byte_identical_across_runs(tmp_path: Path) -> None:
    outputs = []
    for run in ("first", "second"):
        report = evaluate_dataset(build_platform(), DATASET)
        metrics, results = tmp_path / run / "metrics.json", tmp_path / run / "results.jsonl"
        write_report(report, metrics=metrics, results=results)
        outputs.append((metrics.read_bytes(), results.read_bytes()))
    assert outputs[0] == outputs[1]
    assert b"\r\n" not in outputs[0][0] + outputs[0][1]


def test_the_report_dvc_tracks_carries_no_timing() -> None:
    metrics = reproducible_metrics(evaluate_dataset(build_platform(), DATASET))
    assert not any("latency" in key for key in metrics)
    assert metrics["dataset_revision"] == dataset_revision(DATASET)


def test_the_committed_metrics_are_the_ones_the_platform_produces() -> None:
    committed = json.loads((ROOT / "reports/evaluation/metrics.json").read_text(encoding="utf-8"))
    assert committed == reproducible_metrics(evaluate_dataset(build_platform(), DATASET))


def test_the_gate_names_the_failed_cases(tmp_path: Path) -> None:
    report = evaluate_dataset(
        build_platform(), write_dataset(tmp_path / "cases.jsonl", FAILING_CASE)
    )
    [failure] = ReleaseGate().failures(report)
    assert "expects-the-impossible" in failure


def test_the_gate_refuses_an_empty_evaluation_set(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("", encoding="utf-8")
    failures = ReleaseGate().failures(evaluate_dataset(build_platform(), path))
    assert "the evaluation set is empty" in failures


def test_the_gate_blocks_any_authorization_violation(tmp_path: Path) -> None:
    leak = {
        "case_id": "leak",
        "question": "How do I request annual leave?",
        "tenant_id": "tenant-acme",
        "forbidden_chunk_ids": ["sv-acme-handbook-c35f8c5e52cd-0000"],
        "reviewer": "people-operations-review",
    }
    report = evaluate_dataset(build_platform(), write_dataset(tmp_path / "cases.jsonl", leak))
    assert any("unauthorized" in failure for failure in ReleaseGate().failures(report))


def test_the_gate_can_require_grounding_separately(tmp_path: Path) -> None:
    report = evaluate_dataset(build_platform(), DATASET)
    assert ReleaseGate(min_grounded_rate=1.0).failures(report) == ()
    lowered = ReleaseGate(min_pass_rate=0.0, min_grounded_rate=1.1).failures(report)
    assert any("grounded rate" in failure for failure in lowered)


# Commands ----------------------------------------------------------------------


def test_evaluate_exits_zero_and_writes_both_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metrics, results = tmp_path / "metrics.json", tmp_path / "results.jsonl"
    code = main(
        [
            "evaluate",
            "--dataset",
            str(DATASET),
            "--metrics",
            str(metrics),
            "--results",
            str(results),
        ]
    )

    assert code == 0
    assert json.loads(metrics.read_text(encoding="utf-8"))["pass_rate"] == 1.0
    assert len(results.read_text(encoding="utf-8").splitlines()) == len(load_cases(DATASET))
    assert "10/10 passed" in capsys.readouterr().out


def test_evaluate_exits_non_zero_when_the_gate_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path / "cases.jsonl", FAILING_CASE)
    code = main(
        [
            "evaluate",
            "--dataset",
            str(dataset),
            "--metrics",
            str(tmp_path / "metrics.json"),
            "--results",
            str(tmp_path / "results.jsonl"),
        ]
    )
    assert code == 1
    assert "gate failed" in capsys.readouterr().err


def test_a_lower_threshold_lets_a_known_failure_through(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "cases.jsonl", FAILING_CASE)
    code = main(
        [
            "evaluate",
            "--dataset",
            str(dataset),
            "--metrics",
            str(tmp_path / "metrics.json"),
            "--results",
            str(tmp_path / "results.jsonl"),
            "--min-pass-rate",
            "0",
        ]
    )
    assert code == 0


def test_optimize_writes_every_candidate_with_its_disposition(tmp_path: Path) -> None:
    output = tmp_path / "candidates.json"
    code = main(
        ["optimize", "--dataset", str(DATASET), "--max-candidates", "3", "--output", str(output)]
    )

    candidates = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert len(candidates) == 3
    assert all(item["dataset_revision"] == dataset_revision(DATASET) for item in candidates)
    assert all(item["summary"]["authorization_violations"] == 0 for item in candidates)


def test_optimization_never_approves_a_candidate_that_fails_the_evaluation_set() -> None:
    runs = optimize_dataset(build_platform(), DATASET, max_candidates=6)
    for run in runs:
        if run.summary.passed_count < run.summary.case_count:
            assert run.promotion == "rejected"


def test_both_offline_commands_can_record_to_mlflow(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    from rag_platform.adapters.mlflow_tracking import client_for

    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    client_for(uri).create_experiment(
        "rag-platform", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    common = ["--dataset", str(DATASET), "--mlflow-tracking-uri", uri]
    assert (
        main(
            [
                "evaluate",
                *common,
                "--metrics",
                str(tmp_path / "metrics.json"),
                "--results",
                str(tmp_path / "results.jsonl"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "optimize",
                *common,
                "--max-candidates",
                "2",
                "--output",
                str(tmp_path / "candidates.json"),
            ]
        )
        == 0
    )

    client = client_for(uri)
    experiment = client.get_experiment_by_name("rag-platform")
    kinds = sorted(
        run.data.tags["rag.kind"] for run in client.search_runs([experiment.experiment_id])
    )
    assert kinds == ["candidate", "candidate", "evaluation"]


def test_serve_is_the_default_command(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    started: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: started.update(app=app, **options))

    assert main([]) == 0
    assert started == {
        "app": "rag_platform.api:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": False,
    }
    assert main(["serve", "--port", "9100"]) == 0
    assert started["port"] == 9100


# Observer hooks ----------------------------------------------------------------


class Recording:
    """An observer that keeps what it was told, to pin down the hook contract."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.states: list[WorkflowState] = []

    @property
    def trace_id(self) -> str | None:
        return "trace-0001"

    def start(
        self, question: str, access: AccessContext, config: PipelineConfig
    ) -> QueryObservation:
        self.events.append(f"start:{access.tenant_id}:{config.version}")
        return self

    def enter(self, node: str) -> None:
        self.events.append(f"enter:{node}")

    def leave(self, node: str, state: WorkflowState, next_node: str) -> None:
        self.events.append(f"leave:{node}->{next_node}")
        self.states.append(state)

    def finish(self, response: QueryResponse) -> None:
        self.events.append(f"finish:{response.status}:{response.trace.trace_id}")

    def fail(self, error: BaseException) -> None:
        self.events.append(f"fail:{type(error).__name__}")


def observed_service(observer: Recording, chunks: list[DocumentChunk]) -> QueryService:
    config = PipelineConfig()
    return QueryService(
        HybridRetriever(InMemoryChunkRepository(chunks), config),
        config,
        ContextBuilder(config),
        observer=observer,
    )


def test_the_observer_sees_every_state_in_order_then_the_response(
    chunks: list[DocumentChunk],
) -> None:
    observer = Recording()
    response = observed_service(observer, chunks).answer("How do I request annual leave?", ACCESS)

    assert observer.events[0] == "start:tenant-a:pipeline-v1"
    assert observer.events[1:3] == ["enter:classify", "leave:classify->rewrite"]
    assert observer.events[-2] == "leave:verify->answer"
    assert observer.events[-1] == "finish:answered:trace-0001"
    assert response.trace.trace_id == "trace-0001"


def test_an_unobserved_query_has_no_trace_id_but_still_measures_itself(
    chunks: list[DocumentChunk],
) -> None:
    config = PipelineConfig()
    service = QueryService(HybridRetriever(InMemoryChunkRepository(chunks), config), config)
    trace = service.answer("How do I request annual leave?", ACCESS).trace

    assert trace.trace_id is None
    assert trace.latency_ms > 0
    assert trace.model_route == "synthesizer-extractive-v1"
    assert trace.prompt_tokens > trace.completion_tokens > 0


def test_a_query_that_never_generates_reports_no_tokens(chunks: list[DocumentChunk]) -> None:
    config = PipelineConfig()
    service = QueryService(HybridRetriever(InMemoryChunkRepository(chunks), config), config)
    trace = service.answer("What about it?", ACCESS).trace
    assert (trace.prompt_tokens, trace.completion_tokens) == (0, 0)


def test_a_failure_is_reported_to_the_observer_and_still_raised() -> None:
    class Exploding:
        def retrieve(self, question: str, access: AccessContext) -> None:
            raise RuntimeError("store unreachable")

    observer = Recording()
    config = PipelineConfig()
    service = QueryService(Exploding(), config, observer=observer)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        service.answer("How do I request annual leave?", ACCESS)
    assert observer.events[-1] == "fail:RuntimeError"
    assert not any(event.startswith("finish") for event in observer.events)


def test_token_estimates_count_whitespace_delimited_words() -> None:
    assert estimate_tokens("  annual   leave requests ") == 3
    assert estimate_tokens("") == 0

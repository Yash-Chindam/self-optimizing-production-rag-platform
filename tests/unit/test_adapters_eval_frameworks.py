"""RAGAS and DeepEval scoring of an evaluation run.

Both are reference-based here, so each test builds a run with a known defect and checks that
the framework reports that defect and nothing else.
"""

import json
from pathlib import Path

import pytest

from rag_platform.bootstrap import build_platform
from rag_platform.cli import evaluate_dataset
from rag_platform.evaluation import CaseResult, EvaluationReport, load_cases
from rag_platform.models import AnswerTrace, EvaluationCase, EvaluationSummary, QueryResponse

pytest.importorskip("ragas")
pytest.importorskip("deepeval")

from rag_platform.adapters.eval_frameworks import (
    AuthorizationMetric,
    EvidenceRecallMetric,
    ExpectedOutcomeMetric,
    GroundednessMetric,
    MissingResponseError,
    RequiredTermsMetric,
    deepeval_regression,
    framework_gate,
    ragas_scores,
    regression_metrics,
    to_test_case,
)

DATASET = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "cases.jsonl"


def case(case_id: str = "leave", **overrides: object) -> EvaluationCase:
    values: dict[str, object] = {
        "case_id": case_id,
        "question": "How do I request annual leave?",
        "tenant_id": "tenant-a",
        "required_evidence_chunk_ids": ["leave"],
        "acceptable_answer_terms": ["HR portal"],
        "reviewer": "people-operations-review",
    }
    values.update(overrides)
    return EvaluationCase.model_validate(values)


def response(
    *,
    status: str = "answered",
    answer: str = "Annual leave requests use the HR portal.",
    retrieved: list[str] | None = None,
    context: list[str] | None = None,
    unsupported: list[str] | None = None,
) -> QueryResponse:
    retrieved_ids = ["leave"] if retrieved is None else retrieved
    return QueryResponse.model_validate(
        {
            "status": status,
            "answer": answer,
            "citations": [],
            "trace": AnswerTrace(
                config_version="pipeline-v1",
                index_version="index-v1",
                retrieval_strategy="hybrid",
                retrieved_chunk_ids=retrieved_ids,
                policy="tenant_and_access_labels",
                context_chunk_ids=retrieved_ids if context is None else context,
                unsupported_claims=unsupported or [],
            ),
        }
    )


def report(*pairs: tuple[EvaluationCase, QueryResponse | None]) -> EvaluationReport:
    results = tuple(
        CaseResult(
            case_id=item.case_id,
            passed=True,
            status=answer.status if answer else "answered",
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            grounded=True,
            authorization_violations=(),
            latency_ms=1.0,
            failure_category=None,
            notes=(),
            response=answer,
        )
        for item, answer in pairs
    )
    summary = EvaluationSummary(
        case_count=len(results),
        passed_count=len(results),
        mean_recall_at_k=1.0,
        mean_reciprocal_rank=1.0,
        grounded_rate=1.0,
        authorization_violations=0,
        p50_latency_ms=1.0,
        p95_latency_ms=1.0,
    )
    return EvaluationReport("rev", "pipeline-v1", ("programs-v1",), summary, results)


def run(
    item: EvaluationCase, answer: QueryResponse
) -> tuple[EvaluationReport, list[EvaluationCase]]:
    return report((item, answer)), [item]


# RAGAS -------------------------------------------------------------------------


def test_ragas_gives_full_marks_when_exactly_the_required_evidence_is_retrieved() -> None:
    scores = ragas_scores(*run(case(), response()))
    assert (scores.context_recall, scores.context_precision, scores.scored_cases) == (1.0, 1.0, 1)


def test_ragas_recall_drops_when_required_evidence_is_missing() -> None:
    item = case(required_evidence_chunk_ids=["leave", "notice-period"])
    scores = ragas_scores(*run(item, response()))
    assert scores.context_recall == 0.5
    assert scores.context_precision == 1.0


def test_ragas_precision_drops_when_irrelevant_chunks_are_retrieved() -> None:
    scores = ragas_scores(
        *run(case(), response(retrieved=["leave", "expenses", "parking", "badge"]))
    )
    assert scores.context_recall == 1.0
    assert scores.context_precision == 0.25
    assert scores.per_case == {"leave": (1.0, 0.25)}


def test_ragas_scores_nothing_retrieved_as_zero() -> None:
    scores = ragas_scores(*run(case(), response(status="insufficient_evidence", retrieved=[])))
    assert (scores.context_recall, scores.context_precision) == (0.0, 0.0)


def test_ragas_skips_cases_that_require_no_evidence() -> None:
    item = case(required_evidence_chunk_ids=[], expected_status="insufficient_evidence")
    scores = ragas_scores(*run(item, response(status="insufficient_evidence", retrieved=[])))
    assert scores.scored_cases == 0
    assert (scores.context_recall, scores.context_precision) == (1.0, 1.0)


# DeepEval ----------------------------------------------------------------------


def test_a_correct_grounded_answer_passes_every_regression_metric() -> None:
    result = deepeval_regression(*run(case(), response()))
    assert result.passed
    assert result.case_count == 1
    assert set(result.metric_means.values()) == {1.0}
    assert len(result.metric_means) == len(regression_metrics())


def test_the_test_case_carries_chunk_ids_and_never_chunk_text() -> None:
    test_case = to_test_case(case(), report((case(), response())).results[0])
    assert test_case.retrieval_context == ["leave"]
    assert test_case.context == ["leave"]
    assert test_case.expected_output == "answered"
    assert test_case.name == "leave"


@pytest.mark.parametrize(
    ("metric", "item", "answer", "score"),
    [
        (
            ExpectedOutcomeMetric(),
            case(),
            response(status="insufficient_evidence"),
            0.0,
        ),
        (
            EvidenceRecallMetric(),
            case(required_evidence_chunk_ids=["leave", "notice-period"]),
            response(),
            0.5,
        ),
        (
            GroundednessMetric(),
            case(),
            response(unsupported=["Annual leave is unlimited."]),
            0.0,
        ),
        (
            GroundednessMetric(),
            case(forbidden_claims=["HR portal"]),
            response(),
            0.0,
        ),
        (
            AuthorizationMetric(),
            case(forbidden_chunk_ids=["forecast"]),
            response(context=["leave", "forecast"]),
            0.0,
        ),
        (
            RequiredTermsMetric(),
            case(acceptable_answer_terms=["HR portal", "five working days"]),
            response(),
            0.5,
        ),
    ],
)
def test_each_metric_detects_its_own_defect(
    metric: ExpectedOutcomeMetric, item: EvaluationCase, answer: QueryResponse, score: float
) -> None:
    test_case = to_test_case(item, report((item, answer)).results[0])
    assert metric.measure(test_case) == score
    assert not metric.is_successful()
    assert metric.reason


def test_a_defect_is_attributed_to_its_case_and_metric() -> None:
    good, bad = case("good"), case("bad", forbidden_chunk_ids=["forecast"])
    result = deepeval_regression(
        report((good, response()), (bad, response(context=["leave", "forecast"]))), [good, bad]
    )

    [failure] = result.failures
    assert (failure.case_id, failure.metric, failure.score) == ("bad", "Authorization", 0.0)
    assert result.metric_means["Authorization"] == 0.5
    assert not result.passed


def test_metrics_with_nothing_to_check_pass() -> None:
    item = case(required_evidence_chunk_ids=[], acceptable_answer_terms=[])
    test_case = to_test_case(item, report((item, response())).results[0])
    assert EvidenceRecallMetric().measure(test_case) == 1.0
    assert RequiredTermsMetric().measure(test_case) == 1.0


def test_metrics_work_through_deepevals_own_assertion() -> None:
    from deepeval import assert_test

    test_case = to_test_case(case(), report((case(), response())).results[0])
    assert_test(test_case, regression_metrics(), run_async=False)


def test_a_report_without_responses_cannot_be_scored() -> None:
    bare = report((case(), None))
    with pytest.raises(MissingResponseError):
        ragas_scores(bare, [case()])
    with pytest.raises(MissingResponseError):
        deepeval_regression(bare, [case()])
    with pytest.raises(MissingResponseError):
        to_test_case(case(), bare.results[0])


# Gate and the real dataset -----------------------------------------------------


def test_the_gate_reports_both_frameworks_findings() -> None:
    item = case(required_evidence_chunk_ids=["leave", "notice-period"])
    arguments = run(item, response(retrieved=["leave", "a", "b", "c"]))
    failures = framework_gate(ragas_scores(*arguments), deepeval_regression(*arguments))

    assert any("RAGAS context recall 0.50" in failure for failure in failures)
    assert any("RAGAS context precision 0.25" in failure for failure in failures)
    assert any("DeepEval Evidence recall failed for leave" in failure for failure in failures)


def test_the_demo_platform_clears_the_framework_gate_on_the_committed_dataset() -> None:
    evaluated = evaluate_dataset(build_platform(), DATASET)
    cases = load_cases(DATASET)

    ragas, deepeval = ragas_scores(evaluated, cases), deepeval_regression(evaluated, cases)

    assert framework_gate(ragas, deepeval) == ()
    assert ragas.scored_cases == sum(bool(item.required_evidence_chunk_ids) for item in cases)
    assert deepeval.case_count == len(cases)
    assert json.dumps(deepeval.metric_means)

import pytest

from rag_platform.evaluation import EvaluationReport
from rag_platform.models import (
    CandidatePromotion,
    CandidateRun,
    DocumentChunk,
    EvaluationCase,
    EvaluationSummary,
    PipelineConfig,
)
from rag_platform.optimization import (
    CandidateSpace,
    Constraints,
    OptimizationContext,
    OptimizationRun,
    dominates,
    mark_pareto_optimal,
)
from rag_platform.programs import ProgramSuite
from rag_platform.repository import InMemoryChunkRepository

TENANT = "tenant-a"
BASELINE = PipelineConfig()

CORPUS = [
    DocumentChunk(
        chunk_id="leave",
        tenant_id=TENANT,
        access_labels=frozenset({"public"}),
        text="Annual leave requests use the HR portal.",
        source_uri="https://example.test/leave",
        source_title="leave",
        index_version="index-v1",
        source_version_id="sv-leave",
    )
]


def summary(*, pass_rate: float, p95_latency_ms: float = 10.0) -> EvaluationSummary:
    return EvaluationSummary(
        case_count=4,
        passed_count=round(pass_rate * 4),
        mean_recall_at_k=pass_rate,
        mean_reciprocal_rank=pass_rate,
        grounded_rate=1.0,
        authorization_violations=0,
        p50_latency_ms=p95_latency_ms,
        p95_latency_ms=p95_latency_ms,
    )


def candidate_run(
    run_id: str,
    *,
    pass_rate: float,
    p95_latency_ms: float = 10.0,
    promotion: CandidatePromotion = "approved",
) -> CandidateRun:
    return CandidateRun(
        run_id=run_id,
        baseline_config_version=BASELINE.version,
        config=BASELINE,
        config_diff=(),
        dataset_revision="eval-v1",
        program_revisions=("programs-v1",),
        summary=summary(pass_rate=pass_rate, p95_latency_ms=p95_latency_ms),
        promotion=promotion,
    )


def pinned_space(**overrides: tuple[object, ...]) -> CandidateSpace:
    """A CandidateSpace where every field defaults to generating no candidate at all."""
    pinned: dict[str, tuple[object, ...]] = {
        "dense_top_k": (BASELINE.dense_top_k,),
        "sparse_top_k": (BASELINE.sparse_top_k,),
        "final_top_k": (BASELINE.final_top_k,),
        "fusion_method": (BASELINE.fusion_method,),
        "graph_expansion_depth": (BASELINE.graph_expansion_depth,),
        "rerank_candidate_count": (BASELINE.rerank_candidate_count,),
        "context_character_budget": (BASELINE.context_character_budget,),
        "answer_sentence_limit": (BASELINE.answer_sentence_limit,),
    }
    pinned.update(overrides)
    return CandidateSpace(**pinned)  # type: ignore[arg-type]


def report(
    pass_rate: float, *, authorization_violations: int = 0, p95: float = 10.0
) -> EvaluationReport:
    return EvaluationReport(
        dataset_revision="eval-v1",
        config_version="pipeline-v1",
        program_revisions=("programs-v1",),
        summary=EvaluationSummary(
            case_count=4,
            passed_count=round(pass_rate * 4),
            mean_recall_at_k=pass_rate,
            mean_reciprocal_rank=pass_rate,
            grounded_rate=1.0,
            authorization_violations=authorization_violations,
            p50_latency_ms=p95,
            p95_latency_ms=p95,
        ),
        results=(),
    )


# CandidateSpace ----------------------------------------------------------------


def test_a_value_equal_to_the_baselines_is_never_generated_as_a_candidate() -> None:
    space = pinned_space(dense_top_k=(BASELINE.dense_top_k, 10))
    generated = space.candidates(BASELINE, limit=10)
    assert [candidate.dense_top_k for candidate in generated] == [10]


def test_candidates_are_bounded_by_the_limit() -> None:
    generated = CandidateSpace().candidates(BASELINE, limit=3)
    assert len(generated) == 3


def test_each_candidate_gets_a_distinct_version() -> None:
    space = CandidateSpace(dense_top_k=(6, 10))
    versions = [candidate.version for candidate in space.candidates(BASELINE, limit=5)]
    assert len(versions) == len(set(versions))
    assert all(version != BASELINE.version for version in versions)


def test_generation_order_is_deterministic() -> None:
    space = CandidateSpace(dense_top_k=(6, 10), sparse_top_k=(6, 10))
    first = [c.dense_top_k for c in space.candidates(BASELINE, limit=10)]
    second = [c.dense_top_k for c in space.candidates(BASELINE, limit=10)]
    assert first == second


# Constraints ---------------------------------------------------------------------


def test_a_candidate_with_no_regression_has_no_violations() -> None:
    assert Constraints().check(report(1.0), report(1.0)) == ()


def test_an_authorization_violation_is_always_rejected() -> None:
    violations = Constraints().check(report(1.0, authorization_violations=1), report(1.0))
    assert [v.constraint for v in violations] == ["authorization"]


def test_latency_above_the_bound_is_rejected() -> None:
    constraints = Constraints(max_p95_latency_ms=100.0)
    violations = constraints.check(report(1.0, p95=150.0), report(1.0, p95=50.0))
    assert [v.constraint for v in violations] == ["latency"]


def test_a_quality_regression_against_the_baseline_is_rejected() -> None:
    violations = Constraints().check(report(0.5), report(1.0))
    assert [v.constraint for v in violations] == ["quality_regression"]


def test_a_quality_improvement_is_never_rejected() -> None:
    assert Constraints().check(report(1.0), report(0.5)) == ()


# Pareto front ----------------------------------------------------------------------


def test_a_candidate_dominated_on_both_axes_is_excluded_from_the_front() -> None:
    better = candidate_run("better", pass_rate=1.0, p95_latency_ms=10.0)
    worse = candidate_run("worse", pass_rate=0.5, p95_latency_ms=20.0)
    assert dominates(better, worse) is True
    assert dominates(worse, better) is False

    by_id = {run.run_id: run for run in mark_pareto_optimal([better, worse])}
    assert by_id["better"].promotion == "pareto_optimal"
    assert by_id["worse"].promotion == "approved"


def test_candidates_that_trade_off_quality_for_speed_are_both_kept() -> None:
    faster = candidate_run("faster", pass_rate=0.5, p95_latency_ms=5.0)
    higher_quality = candidate_run("higher_quality", pass_rate=1.0, p95_latency_ms=20.0)
    assert dominates(faster, higher_quality) is False
    assert dominates(higher_quality, faster) is False

    marked = mark_pareto_optimal([faster, higher_quality])
    assert {run.promotion for run in marked} == {"pareto_optimal"}


def test_rejected_candidates_never_enter_the_front() -> None:
    rejected = candidate_run("rejected", pass_rate=0.1, p95_latency_ms=1.0, promotion="rejected")
    approved = candidate_run("approved", pass_rate=0.5, p95_latency_ms=50.0)
    by_id = {run.run_id: run for run in mark_pareto_optimal([rejected, approved])}
    assert by_id["rejected"].promotion == "rejected"
    assert by_id["approved"].promotion == "pareto_optimal"


def test_a_tie_does_not_dominate() -> None:
    one = candidate_run("one", pass_rate=1.0, p95_latency_ms=10.0)
    two = candidate_run("two", pass_rate=1.0, p95_latency_ms=10.0)
    assert dominates(one, two) is False
    assert dominates(two, one) is False


# OptimizationRun end-to-end ---------------------------------------------------------


def build_context() -> OptimizationContext:
    return OptimizationContext(repository=InMemoryChunkRepository(CORPUS), programs=ProgramSuite())


def eval_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="leave",
            question="How do I request annual leave?",
            tenant_id=TENANT,
            required_evidence_chunk_ids=frozenset({"leave"}),
            reviewer="reviewer-1",
        )
    ]


def test_run_evaluates_every_generated_candidate() -> None:
    space = CandidateSpace(dense_top_k=(6, 10), sparse_top_k=(6, 10))
    constraints = Constraints(max_p95_latency_ms=10_000.0)
    run = OptimizationRun(build_context(), dataset_revision="eval-v1", constraints=constraints)

    runs = run.run(BASELINE, eval_cases(), space, max_candidates=4)

    assert len(runs) == 4
    assert all(candidate.baseline_config_version == BASELINE.version for candidate in runs)
    assert all(candidate.dataset_revision == "eval-v1" for candidate in runs)
    assert all(candidate.promotion != "rejected" for candidate in runs)
    assert any(candidate.promotion == "pareto_optimal" for candidate in runs)


def test_run_rejects_a_candidate_that_breaks_a_passing_case() -> None:
    space = pinned_space(context_character_budget=(10,))
    constraints = Constraints(max_p95_latency_ms=10_000.0)
    run = OptimizationRun(build_context(), dataset_revision="eval-v1", constraints=constraints)

    [candidate] = run.run(BASELINE, eval_cases(), space, max_candidates=1)

    assert candidate.promotion == "rejected"
    assert any(v.constraint == "quality_regression" for v in candidate.constraint_violations)


def test_config_diff_reports_only_the_changed_fields() -> None:
    space = pinned_space(dense_top_k=(10,))
    run = OptimizationRun(build_context(), dataset_revision="eval-v1")

    [candidate] = run.run(BASELINE, eval_cases(), space, max_candidates=1)

    assert candidate.config_diff == (f"dense_top_k: {BASELINE.dense_top_k!r} -> 10",)


def test_canary_promotes_a_pareto_optimal_candidate_that_still_passes() -> None:
    space = pinned_space(dense_top_k=(10,))
    constraints = Constraints(max_p95_latency_ms=10_000.0)
    run = OptimizationRun(build_context(), dataset_revision="eval-v1", constraints=constraints)
    [candidate] = run.run(BASELINE, eval_cases(), space, max_candidates=1)
    assert candidate.promotion == "pareto_optimal"

    decided = run.canary(candidate, eval_cases(), BASELINE)

    assert decided.promotion == "promoted"


def test_canary_refuses_a_candidate_that_is_not_pareto_optimal() -> None:
    run = OptimizationRun(build_context(), dataset_revision="eval-v1")
    not_optimal = candidate_run("not-optimal", pass_rate=0.5, promotion="approved")

    with pytest.raises(ValueError, match="pareto-optimal"):
        run.canary(not_optimal, eval_cases(), BASELINE)

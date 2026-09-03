"""Optimization control loop (section 12) over a bounded candidate space (section 11).

`CandidateSpace` perturbs one `PipelineConfig` field at a time within reviewer-approved bounds;
only parameters a query-time reconfiguration can change are bounded here; chunk size, embedding
model and DSPy program revision require reingestion or recompilation and stay frozen for one
optimization run. `OptimizationRun` evaluates the baseline and every candidate against the same
versioned evaluation set, rejects any candidate that lets an unauthorized chunk into context,
regresses latency, or regresses quality against the frozen baseline, and marks the surviving,
non-dominated candidates Pareto-optimal. `canary` performs the loop's last two steps: one
approved candidate is evaluated again and promoted only if it still clears every constraint,
otherwise it is rolled back.

Optimization runs outside production (section 3, "Optimization service"): nothing here changes
which configuration answers a live query. `PipelineConfigRegistry` is what a control-plane
process calls to actually promote or roll back the active configuration.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rag_platform.context import ContextBuilder, ParentLookup
from rag_platform.evaluation import EvaluationReport, Evaluator
from rag_platform.graph import GraphExpander
from rag_platform.models import (
    CandidateRun,
    ConstraintViolation,
    EvaluationCase,
    PipelineConfig,
)
from rag_platform.programs import ProgramSuite
from rag_platform.repository import ChunkRepository
from rag_platform.rerank import Reranker
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService

_CandidateField = tuple[str, tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class CandidateSpace:
    """Bounds a candidate `PipelineConfig` may differ from the baseline within (section 11)."""

    dense_top_k: tuple[int, ...] = (6, 8, 10)
    sparse_top_k: tuple[int, ...] = (6, 8, 10)
    final_top_k: tuple[int, ...] = (3, 4, 5)
    fusion_method: tuple[str, ...] = ("reciprocal_rank", "weighted")
    graph_expansion_depth: tuple[int, ...] = (0, 1, 2)
    rerank_candidate_count: tuple[int, ...] = (8, 12, 16)
    context_character_budget: tuple[int, ...] = (3_000, 4_000, 5_000)
    answer_sentence_limit: tuple[int, ...] = (1, 2, 3)

    def candidates(self, baseline: PipelineConfig, *, limit: int) -> list[PipelineConfig]:
        """One-parameter-at-a-time perturbations of the baseline, deterministically ordered."""
        generated: list[PipelineConfig] = []
        for name, values in self._fields():
            for value in values:
                if value == getattr(baseline, name):
                    continue
                generated.append(
                    baseline.model_copy(
                        update={
                            name: value,
                            "version": f"{baseline.version}-cand-{len(generated) + 1}",
                        }
                    )
                )
                if len(generated) >= limit:
                    return generated
        return generated

    def _fields(self) -> tuple[_CandidateField, ...]:
        return (
            ("dense_top_k", self.dense_top_k),
            ("sparse_top_k", self.sparse_top_k),
            ("final_top_k", self.final_top_k),
            ("fusion_method", self.fusion_method),
            ("graph_expansion_depth", self.graph_expansion_depth),
            ("rerank_candidate_count", self.rerank_candidate_count),
            ("context_character_budget", self.context_character_budget),
            ("answer_sentence_limit", self.answer_sentence_limit),
        )


@dataclass(frozen=True, slots=True)
class Constraints:
    """Rejection thresholds checked against the frozen baseline (sections 12 and 17)."""

    max_p95_latency_ms: float = 250.0
    max_quality_regression: float = 0.0
    """How far a candidate's pass rate may fall below the baseline's before it is rejected."""

    def check(
        self, candidate: EvaluationReport, baseline: EvaluationReport
    ) -> tuple[ConstraintViolation, ...]:
        violations: list[ConstraintViolation] = []
        if candidate.summary.authorization_violations > 0:
            violations.append(
                ConstraintViolation(
                    constraint="authorization",
                    detail=(
                        f"{candidate.summary.authorization_violations} unauthorized chunk(s) "
                        "reached context"
                    ),
                )
            )
        if candidate.summary.p95_latency_ms > self.max_p95_latency_ms:
            violations.append(
                ConstraintViolation(
                    constraint="latency",
                    detail=(
                        f"p95 latency {candidate.summary.p95_latency_ms:.1f}ms exceeds "
                        f"{self.max_p95_latency_ms:.1f}ms"
                    ),
                )
            )
        regression = baseline.summary.pass_rate - candidate.summary.pass_rate
        if regression > self.max_quality_regression:
            violations.append(
                ConstraintViolation(
                    constraint="quality_regression",
                    detail=(
                        f"pass rate {candidate.summary.pass_rate:.2f} is below baseline "
                        f"{baseline.summary.pass_rate:.2f}"
                    ),
                )
            )
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class OptimizationContext:
    """The config-independent components a candidate's query service is assembled from."""

    repository: ChunkRepository
    programs: ProgramSuite
    graph_retriever: GraphExpander | None = None
    reranker: Reranker | None = None
    parent_lookup: ParentLookup | None = None

    def build_service(self, config: PipelineConfig) -> QueryService:
        retriever = HybridRetriever(
            self.repository, config, graph_retriever=self.graph_retriever, reranker=self.reranker
        )
        return QueryService(
            retriever, config, ContextBuilder(config, self.parent_lookup), self.programs
        )


class OptimizationRun:
    """Runs the control loop in specification section 12 against a bounded candidate space."""

    def __init__(
        self,
        context: OptimizationContext,
        *,
        dataset_revision: str,
        constraints: Constraints | None = None,
    ) -> None:
        self._context = context
        self._dataset_revision = dataset_revision
        self._constraints = constraints or Constraints()

    def run(
        self,
        baseline: PipelineConfig,
        cases: Sequence[EvaluationCase],
        space: CandidateSpace,
        *,
        max_candidates: int = 8,
    ) -> list[CandidateRun]:
        baseline_report = self._evaluate(baseline, cases)
        runs = [
            self._to_candidate_run(baseline, candidate, cases, baseline_report)
            for candidate in space.candidates(baseline, limit=max_candidates)
        ]
        return mark_pareto_optimal(runs)

    def canary(
        self, run: CandidateRun, cases: Sequence[EvaluationCase], baseline: PipelineConfig
    ) -> CandidateRun:
        """The control loop's last two steps: canary one approved candidate, then decide."""
        if run.promotion != "pareto_optimal":
            raise ValueError(
                f"only a pareto-optimal candidate can be canaried, not {run.promotion!r}"
            )
        canarying = run.with_promotion("canary")
        baseline_report = self._evaluate(baseline, cases)
        report = self._evaluate(run.config, cases)
        violations = self._constraints.check(report, baseline_report)
        return canarying.with_promotion("rolled_back" if violations else "promoted")

    def _evaluate(
        self, config: PipelineConfig, cases: Sequence[EvaluationCase]
    ) -> EvaluationReport:
        service = self._context.build_service(config)
        return Evaluator(service, dataset_revision=self._dataset_revision).evaluate(cases)

    def _to_candidate_run(
        self,
        baseline: PipelineConfig,
        candidate: PipelineConfig,
        cases: Sequence[EvaluationCase],
        baseline_report: EvaluationReport,
    ) -> CandidateRun:
        report = self._evaluate(candidate, cases)
        violations = self._constraints.check(report, baseline_report)
        return CandidateRun(
            run_id=f"run-{candidate.version}",
            baseline_config_version=baseline.version,
            config=candidate,
            config_diff=_diff(baseline, candidate),
            dataset_revision=self._dataset_revision,
            program_revisions=report.program_revisions,
            summary=report.summary,
            constraint_violations=violations,
            promotion="rejected" if violations else "approved",
        )


def _diff(baseline: PipelineConfig, candidate: PipelineConfig) -> tuple[str, ...]:
    baseline_fields = baseline.model_dump(exclude={"version"})
    candidate_fields = candidate.model_dump(exclude={"version"})
    return tuple(
        f"{name}: {baseline_fields[name]!r} -> {value!r}"
        for name, value in candidate_fields.items()
        if value != baseline_fields[name]
    )


def mark_pareto_optimal(runs: list[CandidateRun]) -> list[CandidateRun]:
    approved = [run for run in runs if run.promotion == "approved"]
    front = {
        run.run_id
        for run in approved
        if not any(dominates(other, run) for other in approved if other is not run)
    }
    return [run.with_promotion("pareto_optimal") if run.run_id in front else run for run in runs]


def dominates(a: CandidateRun, b: CandidateRun) -> bool:
    at_least_as_good = (
        a.summary.pass_rate >= b.summary.pass_rate
        and a.summary.p95_latency_ms <= b.summary.p95_latency_ms
    )
    strictly_better = (
        a.summary.pass_rate > b.summary.pass_rate
        or a.summary.p95_latency_ms < b.summary.p95_latency_ms
    )
    return at_least_as_good and strictly_better

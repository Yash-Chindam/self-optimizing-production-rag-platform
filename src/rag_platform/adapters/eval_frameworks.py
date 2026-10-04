"""RAGAS and DeepEval over an evaluation run (specification section 16).

Section 16 asks for RAGAS, DeepEval, deterministic checks and reviewed evidence together, and
forbids releasing on one LLM-judge score. Everything here is therefore reference-based: the
frameworks score the run against the reviewer-authored expectations in each `EvaluationCase`,
and no metric calls a model. They are a second and third opinion on the same responses the
deterministic `Evaluator` already judged, computed by independent implementations.

- RAGAS scores retrieval: id-based context recall and precision of what was retrieved against
  the evidence a reviewer said a correct answer needs.
- DeepEval runs the regression metrics: one custom `BaseMetric` per release property
  (expected outcome, evidence recall, grounding, authorization, required terms).

`framework_gate` turns both into the same kind of failure list `ReleaseGate` produces.
"""

import os
import statistics
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_platform.evaluation import CaseResult, EvaluationReport
from rag_platform.models import EvaluationCase

# Neither framework may phone home from an evaluation run. Set before they are imported.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from ragas import SingleTurnSample

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall


class MissingResponseError(ValueError):
    """The report was built without responses, so there is nothing for a framework to score."""


def _pairs(
    report: EvaluationReport, cases: Sequence[EvaluationCase]
) -> list[tuple[EvaluationCase, CaseResult]]:
    by_id = {case.case_id: case for case in cases}
    pairs = []
    for result in report.results:
        if result.response is None:
            raise MissingResponseError(f"case {result.case_id} has no recorded response")
        pairs.append((by_id[result.case_id], result))
    return pairs


# RAGAS -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RagasScores:
    context_recall: float
    context_precision: float
    scored_cases: int
    per_case: dict[str, tuple[float, float]]
    """Case id to (recall, precision), for the cases that name required evidence."""


def ragas_scores(report: EvaluationReport, cases: Sequence[EvaluationCase]) -> RagasScores:
    """Id-based context recall and precision over the cases that require evidence."""
    recall_metric, precision_metric = IDBasedContextRecall(), IDBasedContextPrecision()
    per_case: dict[str, tuple[float, float]] = {}
    for case, result in _pairs(report, cases):
        if not case.required_evidence_chunk_ids:
            continue
        assert result.response is not None
        retrieved = list(result.response.trace.retrieved_chunk_ids)
        if not retrieved:
            per_case[case.case_id] = (0.0, 0.0)
            continue
        sample = SingleTurnSample(
            user_input=case.question,
            retrieved_context_ids=retrieved,
            reference_context_ids=sorted(case.required_evidence_chunk_ids),
        )
        per_case[case.case_id] = (
            float(recall_metric.single_turn_score(sample)),
            float(precision_metric.single_turn_score(sample)),
        )
    if not per_case:
        return RagasScores(1.0, 1.0, 0, {})
    return RagasScores(
        context_recall=statistics.fmean(recall for recall, _precision in per_case.values()),
        context_precision=statistics.fmean(precision for _recall, precision in per_case.values()),
        scored_cases=len(per_case),
        per_case=per_case,
    )


# DeepEval ----------------------------------------------------------------------


def to_test_case(case: EvaluationCase, result: CaseResult) -> LLMTestCase:
    """A DeepEval test case. Contexts are chunk ids: chunk text never leaves the platform."""
    if result.response is None:
        raise MissingResponseError(f"case {result.case_id} has no recorded response")
    response = result.response
    return LLMTestCase(
        name=case.case_id,
        input=case.question,
        actual_output=response.answer,
        expected_output=case.expected_status,
        context=sorted(case.required_evidence_chunk_ids),
        retrieval_context=list(response.trace.context_chunk_ids),
        metadata={
            "status": response.status,
            "retrieved_chunk_ids": list(response.trace.retrieved_chunk_ids),
            "unsupported_claims": list(response.trace.unsupported_claims),
            "forbidden_chunk_ids": sorted(case.forbidden_chunk_ids),
            "forbidden_claims": list(case.forbidden_claims),
            "acceptable_answer_terms": sorted(case.acceptable_answer_terms),
        },
    )


class _ReferenceMetric(BaseMetric):  # type: ignore[misc]
    """A deterministic DeepEval metric: scored from the test case alone, no model involved."""

    name = "reference"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.async_mode = False

    def evaluate(self, test_case: LLMTestCase) -> tuple[float, str]:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        self.score, self.reason = self.evaluate(test_case)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return self.name


def _metadata(test_case: LLMTestCase) -> dict[str, Any]:
    return dict(test_case.metadata or {})


class ExpectedOutcomeMetric(_ReferenceMetric):
    """Answered, abstained or asked for clarification, as the reviewer expected."""

    name = "Expected outcome"

    def evaluate(self, test_case: LLMTestCase) -> tuple[float, str]:
        actual = _metadata(test_case)["status"]
        matched = actual == test_case.expected_output
        return float(matched), f"expected {test_case.expected_output}, got {actual}"


class EvidenceRecallMetric(_ReferenceMetric):
    """The share of required evidence that was retrieved."""

    name = "Evidence recall"

    def evaluate(self, test_case: LLMTestCase) -> tuple[float, str]:
        required = set(test_case.context or [])
        if not required:
            return 1.0, "no evidence required"
        found = required & set(_metadata(test_case)["retrieved_chunk_ids"])
        return len(found) / len(required), f"{len(found)} of {len(required)} retrieved"


class GroundednessMetric(_ReferenceMetric):
    """No claim the verifier could not support, and no claim the reviewer forbade."""

    name = "Groundedness"

    def evaluate(self, test_case: LLMTestCase) -> tuple[float, str]:
        metadata = _metadata(test_case)
        answer = (test_case.actual_output or "").lower()
        forbidden = [claim for claim in metadata["forbidden_claims"] if claim.lower() in answer]
        problems = len(metadata["unsupported_claims"]) + len(forbidden)
        return float(problems == 0), f"{problems} unsupported or forbidden claim(s)"


class AuthorizationMetric(_ReferenceMetric):
    """Nothing the caller must not see reached the context."""

    name = "Authorization"

    def evaluate(self, test_case: LLMTestCase) -> tuple[float, str]:
        leaked = set(_metadata(test_case)["forbidden_chunk_ids"]) & set(
            test_case.retrieval_context or []
        )
        return float(not leaked), f"{len(leaked)} forbidden chunk(s) in context"


class RequiredTermsMetric(_ReferenceMetric):
    """The share of reviewer-required terms the answer mentions."""

    name = "Required terms"

    def evaluate(self, test_case: LLMTestCase) -> tuple[float, str]:
        terms = _metadata(test_case)["acceptable_answer_terms"]
        if not terms:
            return 1.0, "no terms required"
        answer = (test_case.actual_output or "").lower()
        present = [term for term in terms if term.lower() in answer]
        return len(present) / len(terms), f"{len(present)} of {len(terms)} present"


def regression_metrics() -> list[_ReferenceMetric]:
    return [
        ExpectedOutcomeMetric(),
        EvidenceRecallMetric(),
        GroundednessMetric(),
        AuthorizationMetric(),
        RequiredTermsMetric(),
    ]


@dataclass(frozen=True, slots=True)
class DeepEvalFailure:
    case_id: str
    metric: str
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class DeepEvalResult:
    case_count: int
    metric_means: dict[str, float]
    failures: tuple[DeepEvalFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def deepeval_regression(
    report: EvaluationReport, cases: Sequence[EvaluationCase]
) -> DeepEvalResult:
    """Run every regression metric on every case."""
    scores: dict[str, list[float]] = {}
    failures: list[DeepEvalFailure] = []
    pairs = _pairs(report, cases)
    for case, result in pairs:
        test_case = to_test_case(case, result)
        for metric in regression_metrics():
            score = metric.measure(test_case)
            scores.setdefault(metric.name, []).append(score)
            if not metric.is_successful():
                failures.append(DeepEvalFailure(case.case_id, metric.name, score, metric.reason))
    return DeepEvalResult(
        case_count=len(pairs),
        metric_means={name: statistics.fmean(values) for name, values in scores.items()},
        failures=tuple(failures),
    )


# Gate --------------------------------------------------------------------------


def framework_gate(
    ragas: RagasScores,
    deepeval: DeepEvalResult,
    *,
    min_context_recall: float = 1.0,
    min_context_precision: float = 0.5,
) -> tuple[str, ...]:
    """Release-blocking findings from the two frameworks, in `ReleaseGate`'s format."""
    found: list[str] = []
    if ragas.context_recall < min_context_recall:
        found.append(
            f"RAGAS context recall {ragas.context_recall:.2f} is below {min_context_recall:.2f}"
        )
    if ragas.context_precision < min_context_precision:
        found.append(
            f"RAGAS context precision {ragas.context_precision:.2f} is below "
            f"{min_context_precision:.2f}"
        )
    found.extend(
        f"DeepEval {failure.metric} failed for {failure.case_id}: {failure.reason}"
        for failure in deepeval.failures
    )
    return tuple(found)

"""RAG evaluation (specification section 16) and failure classification (section 13).

An EvaluationCase pins a question to its authorized access context, the evidence a correct
answer must cite, terms a correct answer must mention, and claims it must never make. Running
the case set against a query service checks retrieval, grounding and authorization correctness
through deterministic comparisons against reviewer-authored expectations, rather than treating an
LLM judge as ground truth, which section 16 says must never be the sole release signal.

A failed case is tagged with the taxonomy category a reviewer would have assigned by hand
(section 13): an unauthorized chunk in context is always `authorization`, a wrong outcome or
missing evidence is `retrieval`, an ungrounded or forbidden claim is `citation`, and a correct,
grounded answer that omits an expected term is `generation`.
"""

import hashlib
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_platform.models import (
    AccessContext,
    EvaluationCase,
    EvaluationSummary,
    FailureCategory,
    PipelineConfig,
    QueryResponse,
)


class ProgramRevisions(Protocol):
    def revisions(self) -> tuple[str, ...]: ...


class Answerer(Protocol):
    """The shape Evaluator needs; QueryService satisfies it without declaring it."""

    @property
    def config(self) -> PipelineConfig: ...

    @property
    def programs(self) -> ProgramRevisions: ...

    def answer(self, question: str, access: AccessContext) -> QueryResponse: ...


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    passed: bool
    status: str
    recall_at_k: float
    reciprocal_rank: float
    grounded: bool
    authorization_violations: tuple[str, ...]
    latency_ms: float
    failure_category: FailureCategory | None
    notes: tuple[str, ...]
    response: QueryResponse | None = None
    """What the service returned, kept so other evaluation frameworks can score the same run."""


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_revision: str
    config_version: str
    program_revisions: tuple[str, ...]
    summary: EvaluationSummary
    results: tuple[CaseResult, ...]

    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(result for result in self.results if not result.passed)


class Evaluator:
    """Runs a versioned evaluation set against one query service (specification section 16)."""

    def __init__(self, service: Answerer, *, dataset_revision: str) -> None:
        self._service = service
        self._dataset_revision = dataset_revision

    def evaluate(self, cases: Sequence[EvaluationCase]) -> EvaluationReport:
        results = tuple(self._run_case(case) for case in cases)
        return EvaluationReport(
            dataset_revision=self._dataset_revision,
            config_version=self._service.config.version,
            program_revisions=self._service.programs.revisions(),
            summary=_summarize(results),
            results=results,
        )

    def _run_case(self, case: EvaluationCase) -> CaseResult:
        access = AccessContext(tenant_id=case.tenant_id, labels=case.access_labels)
        start = time.perf_counter()
        response = self._service.answer(case.question, access)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = response.trace.retrieved_chunk_ids
        recall = _recall(case.required_evidence_chunk_ids, set(retrieved_ids))
        reciprocal_rank = _reciprocal_rank(case.required_evidence_chunk_ids, retrieved_ids)
        leaked_evidence = tuple(
            sorted(case.forbidden_chunk_ids & set(response.trace.context_chunk_ids))
        )
        leaked_claims = tuple(
            claim for claim in case.forbidden_claims if claim.lower() in response.answer.lower()
        )
        missing_terms = tuple(
            term
            for term in case.acceptable_answer_terms
            if term.lower() not in response.answer.lower()
        )
        category = _classify_failure(
            case=case,
            status=response.status,
            recall=recall,
            unsupported_claims=response.trace.unsupported_claims,
            leaked_evidence=leaked_evidence,
            leaked_claims=leaked_claims,
            missing_terms=missing_terms,
        )
        return CaseResult(
            case_id=case.case_id,
            passed=category is None,
            status=response.status,
            recall_at_k=recall,
            reciprocal_rank=reciprocal_rank,
            grounded=not response.trace.unsupported_claims,
            authorization_violations=leaked_evidence,
            latency_ms=latency_ms,
            failure_category=category,
            notes=leaked_claims + missing_terms,
            response=response,
        )


def _classify_failure(
    *,
    case: EvaluationCase,
    status: str,
    recall: float,
    unsupported_claims: Sequence[str],
    leaked_evidence: tuple[str, ...],
    leaked_claims: tuple[str, ...],
    missing_terms: tuple[str, ...],
) -> FailureCategory | None:
    if leaked_evidence:
        return "authorization"
    if status != case.expected_status:
        if case.expected_status == "answered" and case.required_evidence_chunk_ids and recall < 1.0:
            return "retrieval"
        return "generation"
    if status != "answered":
        return None
    if unsupported_claims or leaked_claims:
        return "citation"
    if case.required_evidence_chunk_ids and recall < 1.0:
        return "retrieval"
    if missing_terms:
        return "generation"
    return None


def _recall(required: frozenset[str], retrieved: set[str]) -> float:
    if not required:
        return 1.0
    return len(required & retrieved) / len(required)


def _reciprocal_rank(required: frozenset[str], retrieved_ranked: Sequence[str]) -> float:
    if not required:
        return 1.0
    for rank, chunk_id in enumerate(retrieved_ranked, start=1):
        if chunk_id in required:
            return 1 / rank
    return 0.0


def _summarize(results: Sequence[CaseResult]) -> EvaluationSummary:
    if not results:
        return EvaluationSummary(
            case_count=0,
            passed_count=0,
            mean_recall_at_k=0.0,
            mean_reciprocal_rank=0.0,
            grounded_rate=0.0,
            authorization_violations=0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
        )
    latencies = sorted(result.latency_ms for result in results)
    return EvaluationSummary(
        case_count=len(results),
        passed_count=sum(result.passed for result in results),
        mean_recall_at_k=statistics.fmean(result.recall_at_k for result in results),
        mean_reciprocal_rank=statistics.fmean(result.reciprocal_rank for result in results),
        grounded_rate=statistics.fmean(result.grounded for result in results),
        authorization_violations=sum(len(result.authorization_violations) for result in results),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
    )


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


# Versioned datasets -------------------------------------------------------------


def load_cases(path: Path) -> tuple[EvaluationCase, ...]:
    """Read a JSON Lines evaluation set. Blank lines are allowed; case ids must be unique."""
    cases: list[EvaluationCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(EvaluationCase.model_validate_json(line))
    identifiers = [case.case_id for case in cases]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate evaluation case ids: {', '.join(duplicates)}")
    return tuple(cases)


def dataset_revision(path: Path) -> str:
    """The MD5 of the dataset file: the same content hash DVC records for it in dvc.lock.

    Deriving the revision from the bytes means an evaluation result can never be attributed to
    a dataset it was not run against, and that the revision MLflow shows is the one DVC tracks.
    """
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def reproducible_metrics(report: EvaluationReport) -> dict[str, object]:
    """Summary metrics that depend only on data, configuration and code, never on timing."""
    summary = report.summary
    return {
        "dataset_revision": report.dataset_revision,
        "config_version": report.config_version,
        "case_count": summary.case_count,
        "passed_count": summary.passed_count,
        "pass_rate": round(summary.pass_rate, 6),
        "mean_recall_at_k": round(summary.mean_recall_at_k, 6),
        "mean_reciprocal_rank": round(summary.mean_reciprocal_rank, 6),
        "grounded_rate": round(summary.grounded_rate, 6),
        "authorization_violations": summary.authorization_violations,
    }


def reproducible_results(report: EvaluationReport) -> list[dict[str, object]]:
    return [
        {
            "case_id": result.case_id,
            "passed": result.passed,
            "status": result.status,
            "recall_at_k": round(result.recall_at_k, 6),
            "reciprocal_rank": round(result.reciprocal_rank, 6),
            "grounded": result.grounded,
            "authorization_violations": list(result.authorization_violations),
            "failure_category": result.failure_category,
        }
        for result in report.results
    ]


def write_report(report: EvaluationReport, *, metrics: Path, results: Path) -> None:
    """Write the timing-free report DVC tracks, byte-identical for identical inputs."""
    metrics.parent.mkdir(parents=True, exist_ok=True)
    results.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(reproducible_metrics(report), indent=2, sort_keys=True) + "\n")
    with results.open("w", encoding="utf-8", newline="\n") as handle:
        for row in reproducible_results(report):
            handle.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass(frozen=True, slots=True)
class ReleaseGate:
    """The evaluation thresholds a change must clear before it is released (section 16)."""

    min_pass_rate: float = 1.0
    min_grounded_rate: float = 1.0
    max_authorization_violations: int = 0

    def failures(self, report: EvaluationReport) -> tuple[str, ...]:
        summary = report.summary
        found: list[str] = []
        if summary.case_count == 0:
            found.append("the evaluation set is empty")
        if summary.authorization_violations > self.max_authorization_violations:
            found.append(
                f"{summary.authorization_violations} unauthorized chunk(s) reached context"
            )
        if summary.pass_rate < self.min_pass_rate:
            failed = ", ".join(result.case_id for result in report.failures())
            found.append(
                f"pass rate {summary.pass_rate:.2f} is below {self.min_pass_rate:.2f} "
                f"(failed: {failed})"
            )
        if summary.grounded_rate < self.min_grounded_rate:
            found.append(
                f"grounded rate {summary.grounded_rate:.2f} is below "
                f"{self.min_grounded_rate:.2f}"
            )
        return tuple(found)

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

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
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

"""User and reviewer feedback on answers (specification sections 5 and 14).

Feedback is tied to the trace of the answer it is about, so a reviewer can open exactly the
query that was complained about. It only becomes evaluation material after a named reviewer has
accepted it: raw user feedback is a signal, reviewed feedback is evidence. An accepted item can
be drafted into an `EvaluationCase`, with the reviewer supplying what a correct answer requires.

Free text is redacted on the way in. Feedback is read by reviewers, not by the user who wrote
it, so an identifier typed into a comment must not survive into the log.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from rag_platform.models import EvaluationCase, FailureCategory
from rag_platform.pii import redact_identifiers

FeedbackStatus = Literal["submitted", "accepted", "dismissed"]


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2_000)
    rating: Literal["helpful", "unhelpful"]
    comment: str = Field(default="", max_length=2_000)
    trace_id: str | None = Field(default=None, max_length=64)


class FeedbackReview(BaseModel):
    reviewer: str = Field(min_length=1)
    decision: Literal["accepted", "dismissed"]
    failure_category: FailureCategory | None = None


class Feedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    feedback_id: str
    tenant_id: str
    access_labels: frozenset[str]
    question: str
    rating: Literal["helpful", "unhelpful"]
    comment: str = ""
    trace_id: str | None = None
    status: FeedbackStatus = "submitted"
    reviewer: str | None = None
    failure_category: FailureCategory | None = None

    def to_case(
        self,
        *,
        required_evidence_chunk_ids: frozenset[str] = frozenset(),
        acceptable_answer_terms: frozenset[str] = frozenset(),
        expected_status: Literal[
            "answered", "insufficient_evidence", "clarification_needed"
        ] = "answered",
    ) -> EvaluationCase:
        """Draft an evaluation case from reviewed feedback; the reviewer supplies the truth."""
        if self.status != "accepted" or self.reviewer is None:
            raise FeedbackReviewError("only accepted feedback can become an evaluation case")
        return EvaluationCase(
            case_id=f"case-{self.feedback_id}",
            question=self.question,
            tenant_id=self.tenant_id,
            access_labels=self.access_labels,
            required_evidence_chunk_ids=required_evidence_chunk_ids,
            acceptable_answer_terms=acceptable_answer_terms,
            expected_status=expected_status,
            reviewer=self.reviewer,
            failure_tags=(self.failure_category,) if self.failure_category else (),
        )


class UnknownFeedbackError(LookupError):
    pass


class FeedbackReviewError(RuntimeError):
    pass


def _new_id() -> str:
    return f"fb-{uuid4().hex[:12]}"


@dataclass(slots=True)
class FeedbackLog:
    """Feedback per tenant. A tenant can never read or review another tenant's feedback."""

    redactor: Callable[[str], str] = redact_identifiers
    id_factory: Callable[[], str] = _new_id
    _items: dict[str, Feedback] = field(default_factory=dict)

    def submit(
        self, tenant_id: str, access_labels: frozenset[str], request: FeedbackRequest
    ) -> Feedback:
        feedback = Feedback(
            feedback_id=self.id_factory(),
            tenant_id=tenant_id,
            access_labels=access_labels,
            question=self.redactor(request.question),
            rating=request.rating,
            comment=self.redactor(request.comment),
            trace_id=request.trace_id,
        )
        self._items[feedback.feedback_id] = feedback
        return feedback

    def review(self, tenant_id: str, feedback_id: str, review: FeedbackReview) -> Feedback:
        current = self.get(tenant_id, feedback_id)
        if current.status != "submitted":
            raise FeedbackReviewError(f"feedback {feedback_id} was already {current.status}")
        reviewed = current.model_copy(
            update={
                "status": review.decision,
                "reviewer": review.reviewer,
                "failure_category": review.failure_category,
            }
        )
        self._items[feedback_id] = reviewed
        return reviewed

    def get(self, tenant_id: str, feedback_id: str) -> Feedback:
        found = self._items.get(feedback_id)
        # A feedback id from another tenant is reported exactly like one that does not exist.
        if found is None or found.tenant_id != tenant_id:
            raise UnknownFeedbackError(feedback_id)
        return found

    def items(self, tenant_id: str, status: FeedbackStatus | None = None) -> list[Feedback]:
        return [
            item
            for item in self._items.values()
            if item.tenant_id == tenant_id and (status is None or item.status == status)
        ]

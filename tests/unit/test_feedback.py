import pytest

from rag_platform.feedback import (
    FeedbackLog,
    FeedbackRequest,
    FeedbackReview,
    FeedbackReviewError,
    UnknownFeedbackError,
)

LABELS = frozenset({"public", "employees"})


def log() -> FeedbackLog:
    counter = iter(range(1, 100))
    return FeedbackLog(id_factory=lambda: f"fb-{next(counter):04d}")


def request(**overrides: object) -> FeedbackRequest:
    values: dict[str, object] = {
        "question": "How do I request annual leave?",
        "rating": "unhelpful",
        "comment": "The answer quoted the wrong policy.",
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
    }
    values.update(overrides)
    return FeedbackRequest.model_validate(values)


def test_submitted_feedback_is_tied_to_its_tenant_and_trace() -> None:
    feedback = log().submit("tenant-a", LABELS, request())

    assert feedback.feedback_id == "fb-0001"
    assert feedback.tenant_id == "tenant-a"
    assert feedback.trace_id == "0af7651916cd43dd8448eb211c80319c"
    assert feedback.status == "submitted"
    assert feedback.reviewer is None


def test_identifiers_are_redacted_from_the_question_and_the_comment() -> None:
    feedback = log().submit(
        "tenant-a",
        LABELS,
        request(
            question="Why was jane.doe@example.com refused leave?",
            comment="Call me on +44 20 7946 0958 about this.",
        ),
    )
    assert "jane.doe@example.com" not in feedback.question
    assert "[EMAIL]" in feedback.question
    assert "7946" not in feedback.comment


def test_a_reviewer_accepts_feedback_and_assigns_a_failure_category() -> None:
    feedback_log = log()
    submitted = feedback_log.submit("tenant-a", LABELS, request())

    reviewed = feedback_log.review(
        "tenant-a",
        submitted.feedback_id,
        FeedbackReview(reviewer="r.osei", decision="accepted", failure_category="retrieval"),
    )

    assert reviewed.status == "accepted"
    assert reviewed.reviewer == "r.osei"
    assert reviewed.failure_category == "retrieval"
    assert feedback_log.get("tenant-a", submitted.feedback_id) == reviewed


def test_feedback_cannot_be_reviewed_twice() -> None:
    feedback_log = log()
    submitted = feedback_log.submit("tenant-a", LABELS, request())
    review = FeedbackReview(reviewer="r.osei", decision="dismissed")
    feedback_log.review("tenant-a", submitted.feedback_id, review)

    with pytest.raises(FeedbackReviewError, match="already dismissed"):
        feedback_log.review("tenant-a", submitted.feedback_id, review)


def test_another_tenant_cannot_read_or_review_feedback() -> None:
    feedback_log = log()
    submitted = feedback_log.submit("tenant-a", LABELS, request())

    with pytest.raises(UnknownFeedbackError):
        feedback_log.get("tenant-b", submitted.feedback_id)
    with pytest.raises(UnknownFeedbackError):
        feedback_log.review(
            "tenant-b", submitted.feedback_id, FeedbackReview(reviewer="x", decision="accepted")
        )
    assert feedback_log.items("tenant-b") == []


def test_unknown_feedback_is_reported() -> None:
    with pytest.raises(UnknownFeedbackError):
        log().get("tenant-a", "fb-missing")


def test_listing_filters_by_status() -> None:
    feedback_log = log()
    first = feedback_log.submit("tenant-a", LABELS, request())
    second = feedback_log.submit("tenant-a", LABELS, request(rating="helpful"))
    feedback_log.review(
        "tenant-a", first.feedback_id, FeedbackReview(reviewer="r.osei", decision="accepted")
    )

    assert [item.feedback_id for item in feedback_log.items("tenant-a")] == ["fb-0001", "fb-0002"]
    assert [item.feedback_id for item in feedback_log.items("tenant-a", "submitted")] == [
        second.feedback_id
    ]
    assert [item.feedback_id for item in feedback_log.items("tenant-a", "accepted")] == [
        first.feedback_id
    ]


def test_accepted_feedback_drafts_an_evaluation_case_with_the_reviewers_expectations() -> None:
    feedback_log = log()
    submitted = feedback_log.submit("tenant-a", LABELS, request())
    accepted = feedback_log.review(
        "tenant-a",
        submitted.feedback_id,
        FeedbackReview(reviewer="r.osei", decision="accepted", failure_category="retrieval"),
    )

    case = accepted.to_case(
        required_evidence_chunk_ids=frozenset({"leave"}),
        acceptable_answer_terms=frozenset({"HR portal"}),
    )

    assert case.case_id == "case-fb-0001"
    assert case.question == "How do I request annual leave?"
    assert case.tenant_id == "tenant-a"
    assert case.access_labels == LABELS
    assert case.reviewer == "r.osei"
    assert case.failure_tags == ("retrieval",)
    assert case.required_evidence_chunk_ids == frozenset({"leave"})


def test_unreviewed_or_dismissed_feedback_never_becomes_a_case() -> None:
    feedback_log = log()
    submitted = feedback_log.submit("tenant-a", LABELS, request())
    with pytest.raises(FeedbackReviewError):
        submitted.to_case()

    dismissed = feedback_log.review(
        "tenant-a", submitted.feedback_id, FeedbackReview(reviewer="r.osei", decision="dismissed")
    )
    with pytest.raises(FeedbackReviewError):
        dismissed.to_case()


def test_generated_ids_are_unique() -> None:
    feedback_log = FeedbackLog()
    first = feedback_log.submit("tenant-a", LABELS, request())
    second = feedback_log.submit("tenant-a", LABELS, request())
    assert first.feedback_id != second.feedback_id
    assert first.feedback_id.startswith("fb-")

"""Presidio recognition: the adapter's own logic against a fake analyzer, then the real one.

The real-analyzer tests skip unless Presidio and its spaCy model are installed:

    python -m pip install -e ".[privacy]" && python -m spacy download en_core_web_sm
"""

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import pytest

from rag_platform.adapters.presidio import (
    DEFAULT_SPACY_MODEL,
    PresidioRecognizer,
    PresidioRecognizerFactory,
    build_analyzer,
)
from rag_platform.models import PiiPolicy
from rag_platform.pii import PiiProcessor, RehydrationApproval


@dataclass
class Hit:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass
class FakeAnalyzer:
    hits: list[Hit] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def analyze(self, **kwargs: Any) -> list[Hit]:
        self.requests.append(kwargs)
        return list(self.hits)


def recognizer(hits: list[Hit], kinds: tuple[str, ...]) -> tuple[PresidioRecognizer, FakeAnalyzer]:
    analyzer = FakeAnalyzer(hits=hits)
    return PresidioRecognizer(analyzer=analyzer, kinds=kinds), analyzer  # type: ignore[arg-type]


def test_only_the_kinds_the_policy_names_are_requested() -> None:
    target, analyzer = recognizer([], ("email", "person"))
    target.detect("Contact Jane Doe.")
    assert analyzer.requests[0]["entities"] == ["EMAIL_ADDRESS", "PERSON"]


def test_a_kind_presidio_does_not_know_is_ignored() -> None:
    target, analyzer = recognizer([], ("email", "unknown_kind"))
    target.detect("text")
    assert analyzer.requests[0]["entities"] == ["EMAIL_ADDRESS"]


def test_no_requested_kind_never_calls_the_analyzer() -> None:
    target, analyzer = recognizer([], ("unknown_kind",))
    assert target.detect("text") == []
    assert analyzer.requests == []


def test_blank_text_never_calls_the_analyzer() -> None:
    target, analyzer = recognizer([], ("email",))
    assert target.detect("   ") == []
    assert analyzer.requests == []


def test_entity_types_are_mapped_back_to_policy_kinds_in_text_order() -> None:
    target, _ = recognizer(
        [Hit("PERSON", 20, 28, 0.85), Hit("EMAIL_ADDRESS", 0, 16, 1.0)], ("email", "person")
    )
    assert target.detect("x" * 40) == [("email", 0, 16), ("person", 20, 28)]


def test_the_more_confident_of_two_overlapping_spans_wins() -> None:
    target, _ = recognizer(
        [Hit("PHONE_NUMBER", 0, 12, 0.4), Hit("US_SSN", 0, 11, 0.85)], ("phone", "national_id")
    )
    assert target.detect("536-90-4399 ") == [("national_id", 0, 11)]


def test_the_longer_span_wins_a_confidence_tie() -> None:
    target, _ = recognizer(
        [Hit("PERSON", 0, 4, 0.85), Hit("PERSON", 0, 8, 0.85)], ("person",)
    )
    assert target.detect("Jane Doe") == [("person", 0, 8)]


def test_the_score_threshold_is_passed_to_the_analyzer() -> None:
    analyzer = FakeAnalyzer()
    PresidioRecognizer(analyzer=analyzer, kinds=("email",), score_threshold=0.7).detect("text")  # type: ignore[arg-type]
    assert analyzer.requests[0]["score_threshold"] == 0.7


def test_the_factory_builds_a_recognizer_per_policy_on_one_analyzer() -> None:
    analyzer = FakeAnalyzer()
    factory = PresidioRecognizerFactory(analyzer=analyzer)  # type: ignore[arg-type]
    built = factory(("person",))
    assert built.kinds == ("person",)
    assert built.analyzer is analyzer


def test_the_processor_pseudonymizes_what_the_injected_recognizer_finds() -> None:
    """`PiiProcessor` is unchanged; only the recognizer behind it is swapped."""
    text = "Jane Doe approved the request."
    analyzer = FakeAnalyzer(hits=[Hit("PERSON", 0, 8, 0.85)])
    processor = PiiProcessor(recognizer_factory=PresidioRecognizerFactory(analyzer=analyzer))  # type: ignore[arg-type]

    result = processor.process(text, PiiPolicy(recognizers=("person",)), "tenant-a")

    assert "Jane Doe" not in result.text
    assert result.entities[0].kind == "person"
    assert len(processor.vault) == 1


# The real analyzer -------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer() -> Any:
    pytest.importorskip("presidio_analyzer")
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package(DEFAULT_SPACY_MODEL):
        pytest.skip(f"spaCy model {DEFAULT_SPACY_MODEL} is not installed")
    return build_analyzer()


SAMPLE = "Contact Jane Doe at jane.doe@example.com or call +1 415 555 0134."


def test_presidio_finds_a_person_name_that_no_pattern_can(analyzer: Any) -> None:
    spans = PresidioRecognizer(analyzer=analyzer, kinds=("person",)).detect(SAMPLE)
    assert [(kind, SAMPLE[start:end]) for kind, start, end in spans] == [("person", "Jane Doe")]


def test_presidio_finds_email_and_phone_without_overlapping_spans(analyzer: Any) -> None:
    spans = PresidioRecognizer(analyzer=analyzer, kinds=("email", "phone")).detect(SAMPLE)
    found = {kind: SAMPLE[start:end] for kind, start, end in spans}
    assert found["email"] == "jane.doe@example.com"
    assert found["phone"].endswith("555 0134")
    ordered = sorted(spans, key=lambda span: span[1])
    assert all(left[2] <= right[1] for left, right in pairwise(ordered))


def test_presidio_only_reports_the_kinds_it_was_asked_for(analyzer: Any) -> None:
    spans = PresidioRecognizer(analyzer=analyzer, kinds=("email",)).detect(SAMPLE)
    assert {kind for kind, _start, _end in spans} == {"email"}


def test_ingestion_side_pseudonymization_runs_on_presidio_end_to_end(analyzer: Any) -> None:
    processor = PiiProcessor(recognizer_factory=PresidioRecognizerFactory(analyzer=analyzer))
    policy = PiiPolicy(recognizers=("email", "phone", "person"))

    result = processor.process(SAMPLE, policy, "tenant-a")

    assert "Jane Doe" not in result.text
    assert "jane.doe@example.com" not in result.text
    person = next(entity for entity in result.entities if entity.kind == "person")
    approval = RehydrationApproval(
        workflow="data_subject_request", approver="privacy-officer", justification="DSAR 42"
    )
    assert processor.vault.rehydrate("tenant-a", person.pseudonym, approval) == "Jane Doe"

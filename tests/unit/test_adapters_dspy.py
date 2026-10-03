"""DSPy programs, exercised offline.

Most tests drive the adapters with a stub predictor, because what they assert is the adapter's
own guarantees: invented citations are discarded, a failing model falls back or raises as
designed, an unverifiable answer is never grounded. A few use DSPy's `DummyLM` to prove the real
signature, compilation and artifact path without a model provider.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# numpy first: DSPy leaves it partly initialized, which breaks a later Qdrant client import.
pytest.importorskip("numpy")
dspy = pytest.importorskip("dspy")

from dspy.utils import DummyLM  # noqa: E402

from rag_platform.adapters import dspy_examples  # noqa: E402
from rag_platform.adapters.dspy_programs import (  # noqa: E402
    SIGNATURES,
    ClassifyQuery,
    DspyAnswerSynthesizer,
    DspyClaimVerifier,
    DspyClarificationGenerator,
    DspyQueryClassifier,
    DspyQueryDecomposer,
    DspyQueryRewriter,
    build_program_suite,
    clarification_metric,
    compile_program,
    decomposition_metric,
    format_evidence,
    intent_metric,
    load_program,
    rewrite_metric,
    split_examples,
    state_revision,
    synthesis_metric,
    verification_metric,
)
from rag_platform.context import ContextBuilder, EvidenceItem  # noqa: E402
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig  # noqa: E402
from rag_platform.programs import QueryClassification, SynthesizedAnswer  # noqa: E402
from rag_platform.repository import InMemoryChunkRepository  # noqa: E402
from rag_platform.retrieval import HybridRetriever  # noqa: E402
from rag_platform.service import QueryService  # noqa: E402

ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))
FACTUAL = QueryClassification(intent="factual", content_terms=("annual", "leave"))

LEAVE = DocumentChunk(
    chunk_id="leave",
    tenant_id="tenant-a",
    access_labels=frozenset({"public"}),
    text="Annual leave requests use the HR portal.",
    source_uri="https://example.test/leave",
    source_title="Handbook",
    index_version="iv-1",
)
EVIDENCE = [EvidenceItem(chunk=LEAVE, citation_id="C1", score=1.0, expanded_to_parent=False)]


@dataclass
class Stub:
    """A predictor that returns canned fields, or raises."""

    fields: dict[str, Any] | None = None
    error: Exception | None = None
    calls: int = 0

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(**(self.fields or {}))


BROKEN = RuntimeError("model unavailable")


# Classifier --------------------------------------------------------------------


def test_the_classifier_returns_the_models_intent_with_content_terms() -> None:
    classifier = DspyQueryClassifier(lm=None, predictor=Stub({"intent": "procedural"}))
    result = classifier.classify("How do I request annual leave?")
    assert result.intent == "procedural"
    assert "leave" in result.content_terms


def test_an_intent_outside_the_typed_set_falls_back_to_the_rules() -> None:
    classifier = DspyQueryClassifier(lm=None, predictor=Stub({"intent": "philosophical"}))
    assert classifier.classify("How do I request annual leave?").intent == "procedural"


def test_a_failing_classifier_falls_back_to_the_rules() -> None:
    classifier = DspyQueryClassifier(lm=None, predictor=Stub(error=BROKEN))
    assert classifier.classify("leave").intent == "ambiguous"


# Rewriter, decomposer, clarifier -----------------------------------------------


def test_the_rewriter_returns_the_models_rewrite() -> None:
    rewriter = DspyQueryRewriter(lm=None, predictor=Stub({"rewritten": "annual leave request"}))
    assert rewriter.rewrite("vacation", FACTUAL) == "annual leave request"


@pytest.mark.parametrize("stub", [Stub(error=BROKEN), Stub({"rewritten": "  "})])
def test_a_failing_or_empty_rewrite_falls_back_to_alias_expansion(stub: Stub) -> None:
    classification = QueryClassification(intent="factual", content_terms=("vacation",))
    assert DspyQueryRewriter(lm=None, predictor=stub).rewrite("vacation", classification) == (
        "vacation leave"
    )


def test_the_decomposer_is_bounded_by_the_limit() -> None:
    decomposer = DspyQueryDecomposer(
        lm=None, predictor=Stub({"subqueries": ["a?", "b?", "c?", "d?"]})
    )
    assert decomposer.decompose("a and b and c and d", limit=2) == ["a?", "b?"]


def test_a_limit_of_one_never_calls_the_model() -> None:
    stub = Stub({"subqueries": ["a?", "b?"]})
    assert DspyQueryDecomposer(lm=None, predictor=stub).decompose("a and b", limit=1) == [
        "a and b"
    ]
    assert stub.calls == 0


def test_an_empty_decomposition_keeps_the_original_question() -> None:
    decomposer = DspyQueryDecomposer(lm=None, predictor=Stub({"subqueries": []}))
    assert decomposer.decompose("How do I request leave?", limit=3) == ["How do I request leave?"]


def test_a_failing_decomposer_falls_back_to_conjunction_splitting() -> None:
    decomposer = DspyQueryDecomposer(lm=None, predictor=Stub(error=BROKEN))
    parts = decomposer.decompose(
        "How do I request annual leave and who approves expense claims?", limit=3
    )
    assert len(parts) == 2


def test_the_clarifier_returns_the_models_question() -> None:
    clarifier = DspyClarificationGenerator(
        lm=None, predictor=Stub({"clarification": "Which leave do you mean?"})
    )
    assert clarifier.generate("leave", FACTUAL) == "Which leave do you mean?"


@pytest.mark.parametrize("stub", [Stub(error=BROKEN), Stub({"clarification": ""})])
def test_a_failing_or_empty_clarifier_falls_back_to_the_template(stub: Stub) -> None:
    text = DspyClarificationGenerator(lm=None, predictor=stub).generate("leave", FACTUAL)
    assert "more detail" in text


# Synthesizer -------------------------------------------------------------------


def test_evidence_is_presented_with_its_citation_identifier() -> None:
    assert format_evidence(EVIDENCE) == ["[C1] Annual leave requests use the HR portal."]


def test_the_synthesizer_returns_a_cited_answer() -> None:
    synthesizer = DspyAnswerSynthesizer(
        lm=None, predictor=Stub({"answer": "Use the HR portal.", "citation_ids": ["C1"]})
    )
    result = synthesizer.synthesize("How?", EVIDENCE, sentence_limit=2)
    assert result == SynthesizedAnswer(text="Use the HR portal.", citation_ids=("C1",))


def test_a_citation_the_model_invents_is_discarded() -> None:
    synthesizer = DspyAnswerSynthesizer(
        lm=None, predictor=Stub({"answer": "Use the HR portal.", "citation_ids": ["[C1]", "C9"]})
    )
    assert synthesizer.synthesize("How?", EVIDENCE, sentence_limit=2).citation_ids == ("C1",)


def test_an_answer_with_no_surviving_citation_is_not_an_answer() -> None:
    synthesizer = DspyAnswerSynthesizer(
        lm=None, predictor=Stub({"answer": "Leave is unlimited.", "citation_ids": ["C9"]})
    )
    assert synthesizer.synthesize("How?", EVIDENCE, sentence_limit=2) == SynthesizedAnswer(
        text="", citation_ids=()
    )


def test_a_failing_synthesizer_raises_so_the_circuit_breaker_decides() -> None:
    synthesizer = DspyAnswerSynthesizer(lm=None, predictor=Stub(error=BROKEN))
    with pytest.raises(RuntimeError):
        synthesizer.synthesize("How?", EVIDENCE, sentence_limit=2)


# Verifier ----------------------------------------------------------------------

ANSWER = SynthesizedAnswer(text="Use the HR portal.", citation_ids=("C1",))


def test_an_answer_with_no_unsupported_claim_is_grounded() -> None:
    verifier = DspyClaimVerifier(lm=None, predictor=Stub({"unsupported_claims": []}))
    assert verifier.verify(ANSWER, EVIDENCE).grounded is True


def test_an_unsupported_claim_is_reported_and_blocks_grounding() -> None:
    verifier = DspyClaimVerifier(
        lm=None, predictor=Stub({"unsupported_claims": ["Leave is unlimited."]})
    )
    result = verifier.verify(ANSWER, EVIDENCE)
    assert result.grounded is False
    assert result.unsupported_claims == ("Leave is unlimited.",)


def test_an_empty_answer_is_never_grounded_and_never_reaches_the_model() -> None:
    stub = Stub({"unsupported_claims": []})
    result = DspyClaimVerifier(lm=None, predictor=stub).verify(
        SynthesizedAnswer(text="", citation_ids=()), EVIDENCE
    )
    assert result.grounded is False
    assert stub.calls == 0


def test_an_answer_citing_nothing_in_context_is_never_grounded() -> None:
    stub = Stub({"unsupported_claims": []})
    result = DspyClaimVerifier(lm=None, predictor=stub).verify(
        SynthesizedAnswer(text="Use the portal.", citation_ids=("C9",)), EVIDENCE
    )
    assert result.grounded is False
    assert stub.calls == 0


def test_the_verifier_only_sees_the_evidence_the_answer_cites() -> None:
    captured: dict[str, Any] = {}

    def predictor(**inputs: Any) -> SimpleNamespace:
        captured.update(inputs)
        return SimpleNamespace(unsupported_claims=[])

    other = EvidenceItem(
        chunk=LEAVE.model_copy(update={"chunk_id": "other", "text": "Unrelated passage."}),
        citation_id="C2",
        score=0.5,
        expanded_to_parent=False,
    )
    DspyClaimVerifier(lm=None, predictor=predictor).verify(ANSWER, [*EVIDENCE, other])
    assert captured["evidence"] == ["[C1] Annual leave requests use the HR portal."]


def test_a_failing_verifier_raises_rather_than_confirming_grounding() -> None:
    verifier = DspyClaimVerifier(lm=None, predictor=Stub(error=BROKEN))
    with pytest.raises(RuntimeError):
        verifier.verify(ANSWER, EVIDENCE)


# Metrics -----------------------------------------------------------------------


def example(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


def test_the_metrics_accept_correct_predictions_and_reject_wrong_ones() -> None:
    assert intent_metric(example(intent="factual"), example(intent="factual"))
    assert not intent_metric(example(intent="factual"), example(intent="ambiguous"))
    assert rewrite_metric(example(expected_terms=["leave"]), example(rewritten="annual leave"))
    assert not rewrite_metric(example(expected_terms=["leave"]), example(rewritten="vacation"))
    assert decomposition_metric(example(expected_count=2), example(subqueries=["a", "b"]))
    assert not decomposition_metric(example(expected_count=2), example(subqueries=["a"]))
    assert verification_metric(example(grounded=True), example(unsupported_claims=[]))
    assert not verification_metric(example(grounded=True), example(unsupported_claims=["x"]))
    assert clarification_metric(example(), example(clarification="Which leave do you mean?"))
    assert not clarification_metric(example(), example(clarification="Tell me more."))


def test_the_synthesis_metric_requires_citations_terms_and_the_sentence_budget() -> None:
    expected = example(expected_citation_ids=["C1"], expected_terms=["portal"], max_sentences=1)
    good = example(answer="Use the HR portal.", citation_ids=["C1"])
    assert synthesis_metric(expected, good)
    assert not synthesis_metric(expected, example(answer="Use the HR portal.", citation_ids=["C2"]))
    assert not synthesis_metric(expected, example(answer="Ask HR.", citation_ids=["C1"]))
    assert not synthesis_metric(
        expected, example(answer="Use the HR portal. It is fast.", citation_ids=["C1"])
    )


# Splits, compilation and artifacts ---------------------------------------------


def test_the_split_is_deterministic_and_disjoint() -> None:
    examples = list(range(9))
    dataset = split_examples(examples)
    assert dataset.heldout == (2, 5, 8)
    assert dataset.train == (0, 1, 3, 4, 6, 7)
    assert split_examples(examples) == dataset


def test_a_split_that_would_hold_everything_out_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        split_examples([1, 2, 3], heldout_every=1)


@pytest.mark.parametrize("name", sorted(SIGNATURES))
def test_every_program_has_curated_examples_covering_its_signature(name: str) -> None:
    examples = getattr(dspy_examples, f"{name}_examples")()
    dataset = split_examples(examples)
    assert len(dataset.train) >= 4
    assert len(dataset.heldout) >= 2
    signature = SIGNATURES[name]
    for item in examples:
        assert set(item.inputs().keys()) == set(signature.input_fields)
        assert set(signature.output_fields) <= set(item.keys())


def classifier_lm() -> Any:
    """Answers every classification from the curated labels, like a perfect model would."""
    labels = {item.question: item.intent for item in dspy_examples.classifier_examples()}

    class LabelLM(DummyLM):  # type: ignore[misc]
        def __call__(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
            content = (messages or [{}])[-1].get("content", "")
            text = content if isinstance(content, str) else str(content)
            # Short questions such as "it" occur inside longer ones, so the longest match wins.
            matches = [question for question in labels if question in text]
            intent = labels[max(matches, key=len)] if matches else "factual"
            self.answers = iter([{"intent": intent}])
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

    return LabelLM([{"intent": "factual"}])


def test_compilation_bootstraps_demonstrations_and_scores_both_splits() -> None:
    compiled = compile_program(
        "classifier",
        ClassifyQuery,
        dspy_examples.classifier_examples(),
        intent_metric,
        classifier_lm(),
        max_demos=3,
    )
    assert compiled.train_examples == 8
    assert compiled.heldout_examples == 4
    assert len(compiled.predictor.demos) >= 1
    assert compiled.train_score == 1.0
    assert compiled.heldout_score == 1.0
    assert compiled.revision.startswith("classifier-dspy-")


def test_the_revision_is_derived_from_the_compiled_state() -> None:
    bare = dspy.Predict(ClassifyQuery)
    compiled = compile_program(
        "classifier",
        ClassifyQuery,
        dspy_examples.classifier_examples(),
        intent_metric,
        classifier_lm(),
    )
    assert state_revision("classifier", bare) == state_revision(
        "classifier", dspy.Predict(ClassifyQuery)
    )
    assert compiled.revision != state_revision("classifier", bare)


def test_a_compiled_artifact_round_trips_with_its_revision(tmp_path: Path) -> None:
    compiled = compile_program(
        "classifier",
        ClassifyQuery,
        dspy_examples.classifier_examples(),
        intent_metric,
        classifier_lm(),
    )
    path = compiled.save(tmp_path)
    loaded = load_program("classifier", ClassifyQuery, tmp_path)

    assert path.name == "classifier.json"
    assert loaded is not None
    predictor, revision = loaded
    assert revision == compiled.revision
    assert state_revision("classifier", predictor) == compiled.revision
    assert len(predictor.demos) == len(compiled.predictor.demos)


def test_a_missing_artifact_loads_as_nothing(tmp_path: Path) -> None:
    assert load_program("classifier", ClassifyQuery, tmp_path) is None


def test_the_suite_revision_changes_when_a_compiled_artifact_is_used(tmp_path: Path) -> None:
    uncompiled = build_program_suite(None)
    compile_program(
        "classifier",
        ClassifyQuery,
        dspy_examples.classifier_examples(),
        intent_metric,
        classifier_lm(),
    ).save(tmp_path)
    compiled = build_program_suite(None, artifacts=tmp_path)

    assert uncompiled.classifier.revision == "classifier-dspy-uncompiled"
    assert compiled.classifier.revision.startswith("classifier-dspy-")
    assert compiled.classifier.revision != uncompiled.classifier.revision
    assert compiled.revision != uncompiled.revision
    assert compiled.verifier.revision == "verifier-dspy-uncompiled"


# The whole workflow on DSPy programs -------------------------------------------


def workflow_lm(answers: Sequence[dict[str, Any]]) -> Any:
    return DummyLM(list(answers))


def service(lm: Any) -> QueryService:
    config = PipelineConfig()
    return QueryService(
        HybridRetriever(InMemoryChunkRepository([LEAVE]), config),
        config,
        ContextBuilder(config),
        build_program_suite(lm),
    )


def test_the_query_workflow_answers_through_real_dspy_signatures() -> None:
    lm = workflow_lm(
        [
            {"intent": "procedural"},
            {"rewritten": "How do I request annual leave?"},
            {"subqueries": ["How do I request annual leave?"]},
            {"answer": "Requests are made in the HR portal.", "citation_ids": ["C1"]},
            {"unsupported_claims": []},
        ]
    )
    response = service(lm).answer("How do I request annual leave?", ACCESS)

    assert response.status == "answered"
    assert response.answer == "Requests are made in the HR portal."
    assert [citation.chunk_id for citation in response.citations] == ["leave"]
    assert response.trace.intent == "procedural"
    assert any(
        revision.startswith("programs-dspy-") for revision in response.trace.program_revisions
    )


def test_an_ungrounded_generation_is_repaired_then_abstains() -> None:
    lm = workflow_lm(
        [
            {"intent": "factual"},
            {"rewritten": "annual leave"},
            {"subqueries": ["annual leave"]},
            {"answer": "Annual leave is unlimited.", "citation_ids": ["C1"]},
            {"unsupported_claims": ["Annual leave is unlimited."]},
            {"answer": "Annual leave is unlimited.", "citation_ids": ["C1"]},
            {"unsupported_claims": ["Annual leave is unlimited."]},
        ]
    )
    response = service(lm).answer("What is the annual leave allowance?", ACCESS)

    assert response.status == "insufficient_evidence"
    assert response.trace.repair_attempts == 1
    assert response.trace.unsupported_claims == ["Annual leave is unlimited."]

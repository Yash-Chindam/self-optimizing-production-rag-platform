from rag_platform.context import EvidenceItem
from rag_platform.models import DocumentChunk
from rag_platform.programs import (
    AliasQueryRewriter,
    ConjunctionQueryDecomposer,
    EntailmentClaimVerifier,
    ExtractiveAnswerSynthesizer,
    ProgramSuite,
    QueryClassification,
    RuleQueryClassifier,
    SynthesizedAnswer,
    TemplateClarificationGenerator,
    split_sentences,
)

LEAVE_TEXT = (
    "Annual leave requests use the HR portal. Managers approve within two working days."
)


def evidence(citation_id: str, text: str, identifier: str = "chunk") -> EvidenceItem:
    return EvidenceItem(
        chunk=DocumentChunk(
            chunk_id=identifier,
            tenant_id="tenant-a",
            access_labels=frozenset({"public"}),
            text=text,
            source_uri="https://example.test/doc",
            source_title="Doc",
            index_version="index-v1",
        ),
        citation_id=citation_id,
        score=1.0,
        expanded_to_parent=False,
    )


def test_classifier_separates_intents() -> None:
    classifier = RuleQueryClassifier()
    assert classifier.classify("How do I submit annual leave?").intent == "procedural"
    assert classifier.classify("Difference between leave and sabbatical").intent == "comparison"
    assert classifier.classify("Annual leave allowance").intent == "factual"
    assert classifier.classify("leave").intent == "ambiguous"


def test_classifier_reports_the_content_terms_it_used() -> None:
    assert RuleQueryClassifier().classify("How do I request leave?").content_terms == (
        "request",
        "leave",
    )


def test_rewriter_adds_indexed_vocabulary_for_synonyms() -> None:
    classification = RuleQueryClassifier().classify("How do I book vacation?")
    rewritten = AliasQueryRewriter().rewrite("How do I book vacation?", classification)
    assert rewritten.endswith("leave")


def test_rewriter_leaves_a_question_alone_when_no_synonym_applies() -> None:
    question = "Annual leave allowance"
    classification = RuleQueryClassifier().classify(question)
    assert AliasQueryRewriter().rewrite(question, classification) == question


def test_decomposer_splits_multi_part_questions() -> None:
    parts = ConjunctionQueryDecomposer().decompose(
        "How do I request leave and who approves expenses?", limit=3
    )
    assert parts == ["How do I request leave", "who approves expenses"]


def test_decomposer_keeps_single_questions_whole() -> None:
    decomposer = ConjunctionQueryDecomposer()
    assert decomposer.decompose("Annual leave rules", limit=3) == ["Annual leave rules"]
    assert decomposer.decompose("leave and pay", limit=3) == ["leave and pay"]
    assert decomposer.decompose("a and b", limit=1) == ["a and b"]


def test_decomposer_respects_its_limit() -> None:
    parts = ConjunctionQueryDecomposer().decompose(
        "leave rules and expense rules and travel rules", limit=2
    )
    assert len(parts) == 2


def test_synthesizer_selects_sentences_from_the_evidence() -> None:
    answer = ExtractiveAnswerSynthesizer().synthesize(
        "How do I request annual leave?", [evidence("C1", LEAVE_TEXT)], sentence_limit=1
    )
    assert answer.text == "Annual leave requests use the HR portal."
    assert answer.citation_ids == ("C1",)


def test_synthesizer_cites_every_source_it_quoted() -> None:
    answer = ExtractiveAnswerSynthesizer().synthesize(
        "leave approval",
        [
            evidence("C1", "Annual leave requests use the HR portal.", "a"),
            evidence("C2", "Leave approval is granted by a manager.", "b"),
        ],
        sentence_limit=2,
    )
    assert set(answer.citation_ids) == {"C1", "C2"}


def test_synthesizer_returns_nothing_when_no_sentence_matches() -> None:
    answer = ExtractiveAnswerSynthesizer().synthesize(
        "lunar eclipse", [evidence("C1", LEAVE_TEXT)], sentence_limit=2
    )
    assert answer == SynthesizedAnswer(text="", citation_ids=())


def test_verifier_accepts_answers_taken_from_cited_evidence() -> None:
    answer = ExtractiveAnswerSynthesizer().synthesize(
        "annual leave", [evidence("C1", LEAVE_TEXT)], sentence_limit=2
    )
    result = EntailmentClaimVerifier().verify(answer, [evidence("C1", LEAVE_TEXT)])
    assert result.grounded
    assert result.unsupported_claims == ()


def test_verifier_rejects_claims_that_are_not_in_the_cited_chunk() -> None:
    invented = SynthesizedAnswer(
        text="Annual leave is unlimited.", citation_ids=("C1",)
    )
    result = EntailmentClaimVerifier().verify(invented, [evidence("C1", LEAVE_TEXT)])
    assert not result.grounded
    assert result.unsupported_claims == ("Annual leave is unlimited.",)


def test_verifier_rejects_uncited_and_empty_answers() -> None:
    verifier = EntailmentClaimVerifier()
    uncited = SynthesizedAnswer(text="Annual leave is unlimited.", citation_ids=())
    assert not verifier.verify(uncited, [evidence("C1", LEAVE_TEXT)]).grounded
    assert verifier.verify(SynthesizedAnswer(text="", citation_ids=()), []).grounded is False


def test_clarifier_asks_about_the_terms_it_recognized() -> None:
    generator = TemplateClarificationGenerator()
    classification = RuleQueryClassifier().classify("leave")
    assert "leave" in generator.generate("leave", classification)

    empty = QueryClassification(intent="ambiguous", content_terms=())
    assert "Which topic" in generator.generate("???", empty)


def test_suite_reports_every_revision_for_provenance() -> None:
    revisions = ProgramSuite().revisions()
    assert revisions[0] == "programs-v1"
    assert "classifier-v1" in revisions
    assert "verifier-v1" in revisions


def test_sentence_splitting_ignores_blank_fragments() -> None:
    assert split_sentences("One.  Two!   ") == ["One.", "Two!"]

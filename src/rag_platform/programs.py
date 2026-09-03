"""Typed language programs (specification section 10).

Each program has a typed signature, a revision and a deterministic default implementation. A
DSPy-compiled module replaces an implementation without changing the signature the workflow
depends on, and the suite revision is what evaluation and promotion record.

The default synthesizer is extractive: it selects sentences from the assembled context rather
than writing new text, so every claim is entailed by the chunk it cites. A generative
implementation swaps in behind the same signature, and the claim verifier is what keeps it
honest.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from rag_platform.context import EvidenceItem
from rag_platform.repository import SEMANTIC_ALIASES, tokenize

Intent = Literal["factual", "procedural", "comparison", "ambiguous"]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PROCEDURAL_MARKERS = frozenset({"how", "steps", "process", "procedure", "submit", "request"})
COMPARISON_MARKERS = frozenset({"compare", "difference", "differences", "versus", "vs"})
CONJUNCTION_SPLIT = re.compile(r"\s+and\s+|\s*;\s*|\s*\?\s*")


@dataclass(frozen=True, slots=True)
class QueryClassification:
    intent: Intent
    content_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SynthesizedAnswer:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    grounded: bool
    unsupported_claims: tuple[str, ...]


class QueryClassifier(Protocol):
    revision: str

    def classify(self, question: str) -> QueryClassification: ...


class QueryRewriter(Protocol):
    revision: str

    def rewrite(self, question: str, classification: QueryClassification) -> str: ...


class QueryDecomposer(Protocol):
    revision: str

    def decompose(self, question: str, *, limit: int) -> list[str]: ...


class AnswerSynthesizer(Protocol):
    revision: str

    def synthesize(
        self, question: str, evidence: Sequence[EvidenceItem], *, sentence_limit: int
    ) -> SynthesizedAnswer: ...


class ClaimVerifier(Protocol):
    revision: str

    def verify(
        self, answer: SynthesizedAnswer, evidence: Sequence[EvidenceItem]
    ) -> VerificationResult: ...


class ClarificationGenerator(Protocol):
    revision: str

    def generate(self, question: str, classification: QueryClassification) -> str: ...


@dataclass(slots=True)
class RuleQueryClassifier:
    revision: str = "classifier-v1"

    def classify(self, question: str) -> QueryClassification:
        terms = tuple(dict.fromkeys(tokenize(question)))
        lowered = set(question.lower().split())
        if len(terms) < 2:
            return QueryClassification(intent="ambiguous", content_terms=terms)
        if lowered & COMPARISON_MARKERS:
            return QueryClassification(intent="comparison", content_terms=terms)
        if lowered & PROCEDURAL_MARKERS:
            return QueryClassification(intent="procedural", content_terms=terms)
        return QueryClassification(intent="factual", content_terms=terms)


@dataclass(slots=True)
class AliasQueryRewriter:
    """Adds the indexed vocabulary for terms the asker used a synonym for."""

    revision: str = "rewriter-v1"

    def rewrite(self, question: str, classification: QueryClassification) -> str:
        present = set(classification.content_terms)
        additions = [
            SEMANTIC_ALIASES[term]
            for term in classification.content_terms
            if term in SEMANTIC_ALIASES and SEMANTIC_ALIASES[term] not in present
        ]
        if not additions:
            return question
        return f"{question} {' '.join(dict.fromkeys(additions))}"


@dataclass(slots=True)
class ConjunctionQueryDecomposer:
    revision: str = "decomposer-v1"

    def decompose(self, question: str, *, limit: int) -> list[str]:
        if limit <= 1:
            return [question]
        parts = [part.strip() for part in CONJUNCTION_SPLIT.split(question) if part.strip()]
        answerable = [part for part in parts if len(tokenize(part)) >= 2]
        if len(answerable) < 2:
            return [question]
        return answerable[:limit]


@dataclass(slots=True)
class ExtractiveAnswerSynthesizer:
    revision: str = "synthesizer-extractive-v1"

    def synthesize(
        self, question: str, evidence: Sequence[EvidenceItem], *, sentence_limit: int
    ) -> SynthesizedAnswer:
        query_terms = set(tokenize(question, semantic=True))
        candidates: list[tuple[float, int, int, str, str]] = []
        for rank, item in enumerate(evidence):
            for position, sentence in enumerate(split_sentences(item.chunk.text)):
                overlap = query_terms & set(tokenize(sentence, semantic=True))
                if overlap:
                    candidates.append(
                        (-len(overlap) / len(query_terms), rank, position, sentence,
                         item.citation_id)
                    )
        if not candidates:
            return SynthesizedAnswer(text="", citation_ids=())

        chosen = sorted(candidates)[:sentence_limit]
        ordered = sorted(chosen, key=lambda item: (item[1], item[2]))
        return SynthesizedAnswer(
            text=" ".join(sentence for *_head, sentence, _citation in ordered),
            citation_ids=tuple(dict.fromkeys(citation for *_head, citation in ordered)),
        )


@dataclass(slots=True)
class EntailmentClaimVerifier:
    """Every sentence of the answer must appear in a cited chunk."""

    revision: str = "verifier-v1"

    def verify(
        self, answer: SynthesizedAnswer, evidence: Sequence[EvidenceItem]
    ) -> VerificationResult:
        if not answer.text.strip():
            return VerificationResult(grounded=False, unsupported_claims=())
        if not answer.citation_ids:
            return VerificationResult(
                grounded=False, unsupported_claims=(answer.text.strip(),)
            )
        cited = {
            item.citation_id: normalize(item.chunk.text)
            for item in evidence
            if item.citation_id in answer.citation_ids
        }
        unsupported = tuple(
            sentence
            for sentence in split_sentences(answer.text)
            if not any(normalize(sentence) in text for text in cited.values())
        )
        return VerificationResult(grounded=not unsupported, unsupported_claims=unsupported)


@dataclass(slots=True)
class TemplateClarificationGenerator:
    revision: str = "clarifier-v1"

    def generate(self, question: str, classification: QueryClassification) -> str:
        if classification.content_terms:
            subject = " and ".join(classification.content_terms)
            return (
                f"I need more detail before I can answer. Which aspect of {subject} do you "
                "mean, and which document or team does it concern?"
            )
        return (
            "I need more detail before I can answer. Which topic, document or team does your "
            "question concern?"
        )


@dataclass(frozen=True, slots=True)
class ProgramSuite:
    """The compiled program set a pipeline configuration points at."""

    revision: str = "programs-v1"
    classifier: QueryClassifier = field(default_factory=RuleQueryClassifier)
    rewriter: QueryRewriter = field(default_factory=AliasQueryRewriter)
    decomposer: QueryDecomposer = field(default_factory=ConjunctionQueryDecomposer)
    synthesizer: AnswerSynthesizer = field(default_factory=ExtractiveAnswerSynthesizer)
    verifier: ClaimVerifier = field(default_factory=EntailmentClaimVerifier)
    clarifier: ClarificationGenerator = field(default_factory=TemplateClarificationGenerator)

    def revisions(self) -> tuple[str, ...]:
        return (
            self.revision,
            self.classifier.revision,
            self.rewriter.revision,
            self.decomposer.revision,
            self.synthesizer.revision,
            self.verifier.revision,
            self.clarifier.revision,
        )


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT.split(text) if sentence.strip()]


def normalize(text: str) -> str:
    return " ".join(text.split())

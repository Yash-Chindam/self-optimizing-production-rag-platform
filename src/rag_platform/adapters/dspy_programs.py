"""DSPy language programs behind the program protocols (specification section 10).

Each of the six candidate modules has what the specification requires of it: a typed signature,
curated examples, an explicit metric, a training and a held-out split, and a versioned compiled
artifact. A program's revision is derived from its compiled state, so two different compilations
can never report the same revision, and the revision an `AnswerTrace` records identifies exactly
the prompt and demonstrations that produced the answer.

What DSPy does *not* decide is unchanged: access policy, indexes and promotion stay outside it.
The adapters also keep three guarantees the deterministic programs gave for free:

- A citation the model invents is discarded; only identifiers present in the context survive.
- Classification, rewriting, decomposition and clarification fall back to the deterministic
  program when the model call fails, because nothing downstream can recover from their absence.
- Synthesis and verification do not fall back. They raise, and the workflow's circuit breakers
  turn that into abstention — a verifier that cannot run never confirms grounding.

Requires the `llm` extra.
"""

import hashlib
import importlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

# DSPy leaves numpy registered but only partly initialized, and a later `import numpy.typing`
# (the Qdrant client does one) then fails on a circular import. Loading numpy fully first makes
# the DSPy programs and the store adapters usable in one process, in either import order.
importlib.import_module("numpy")

import dspy  # noqa: E402

from rag_platform.context import EvidenceItem  # noqa: E402
from rag_platform.programs import (  # noqa: E402
    AliasQueryRewriter,
    ConjunctionQueryDecomposer,
    Intent,
    ProgramSuite,
    QueryClassification,
    QueryClassifier,
    QueryDecomposer,
    QueryRewriter,
    RuleQueryClassifier,
    SynthesizedAnswer,
    TemplateClarificationGenerator,
    VerificationResult,
    split_sentences,
)
from rag_platform.repository import tokenize  # noqa: E402

INTENTS: tuple[str, ...] = get_args(Intent)
Metric = Callable[[Any, Any, Any], bool]


# Typed signatures --------------------------------------------------------------


class ClassifyQuery(dspy.Signature):  # type: ignore[misc]
    """Classify a question asked of an enterprise knowledge base.

    `ambiguous` means the question cannot be answered without asking what it refers to.
    """

    question: str = dspy.InputField()
    intent: Literal["factual", "procedural", "comparison", "ambiguous"] = dspy.OutputField()


class RewriteQuery(dspy.Signature):  # type: ignore[misc]
    """Rewrite a question into the vocabulary a document index is likely to use.

    Keep the meaning. Expand abbreviations and informal synonyms. Do not answer the question.
    """

    question: str = dspy.InputField()
    intent: str = dspy.InputField()
    rewritten: str = dspy.OutputField()


class DecomposeQuery(dspy.Signature):  # type: ignore[misc]
    """Split a question into the independent sub-questions it contains.

    Return the question unchanged, as a single item, when it asks only one thing.
    """

    question: str = dspy.InputField()
    max_subqueries: int = dspy.InputField()
    subqueries: list[str] = dspy.OutputField()


class SynthesizeAnswer(dspy.Signature):  # type: ignore[misc]
    """Answer the question using only the evidence, and cite every passage relied on.

    Each evidence passage starts with its citation identifier in brackets. Make no claim the
    evidence does not support. If the evidence does not answer the question, return an empty
    answer and no citations.
    """

    question: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    max_sentences: int = dspy.InputField()
    answer: str = dspy.OutputField()
    citation_ids: list[str] = dspy.OutputField()


class VerifyClaims(dspy.Signature):  # type: ignore[misc]
    """List every sentence of the answer that the cited evidence does not entail.

    Return an empty list only when each sentence is fully supported by the evidence.
    """

    answer: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    unsupported_claims: list[str] = dspy.OutputField()


class GenerateClarification(dspy.Signature):  # type: ignore[misc]
    """Ask the one question that would make an ambiguous request answerable."""

    question: str = dspy.InputField()
    clarification: str = dspy.OutputField()


# Protocol adapters -------------------------------------------------------------


def _predict(lm: Any, predictor: Any, **inputs: Any) -> Any:
    with dspy.context(lm=lm):
        return predictor(**inputs)


def format_evidence(evidence: Sequence[EvidenceItem]) -> list[str]:
    return [f"[{item.citation_id}] {item.chunk.text}" for item in evidence]


@dataclass(slots=True)
class DspyQueryClassifier:
    lm: Any
    predictor: Any = field(default_factory=lambda: dspy.Predict(ClassifyQuery))
    revision: str = "classifier-dspy-uncompiled"
    fallback: QueryClassifier = field(default_factory=RuleQueryClassifier)

    def classify(self, question: str) -> QueryClassification:
        try:
            intent = str(_predict(self.lm, self.predictor, question=question).intent).strip()
        except Exception:
            return self.fallback.classify(question)
        if intent not in INTENTS:
            return self.fallback.classify(question)
        return QueryClassification(
            intent=intent,  # type: ignore[arg-type]
            content_terms=tuple(dict.fromkeys(tokenize(question))),
        )


@dataclass(slots=True)
class DspyQueryRewriter:
    lm: Any
    predictor: Any = field(default_factory=lambda: dspy.Predict(RewriteQuery))
    revision: str = "rewriter-dspy-uncompiled"
    fallback: QueryRewriter = field(default_factory=AliasQueryRewriter)

    def rewrite(self, question: str, classification: QueryClassification) -> str:
        try:
            rewritten = str(
                _predict(
                    self.lm, self.predictor, question=question, intent=classification.intent
                ).rewritten
            ).strip()
        except Exception:
            return self.fallback.rewrite(question, classification)
        return rewritten or self.fallback.rewrite(question, classification)


@dataclass(slots=True)
class DspyQueryDecomposer:
    lm: Any
    predictor: Any = field(default_factory=lambda: dspy.Predict(DecomposeQuery))
    revision: str = "decomposer-dspy-uncompiled"
    fallback: QueryDecomposer = field(default_factory=ConjunctionQueryDecomposer)

    def decompose(self, question: str, *, limit: int) -> list[str]:
        if limit <= 1:
            return [question]
        try:
            raw = _predict(
                self.lm, self.predictor, question=question, max_subqueries=limit
            ).subqueries
        except Exception:
            return self.fallback.decompose(question, limit=limit)
        parts = [str(part).strip() for part in raw or [] if str(part).strip()]
        return parts[:limit] or [question]


@dataclass(slots=True)
class DspyAnswerSynthesizer:
    lm: Any
    predictor: Any = field(default_factory=lambda: dspy.Predict(SynthesizeAnswer))
    revision: str = "synthesizer-dspy-uncompiled"

    def synthesize(
        self, question: str, evidence: Sequence[EvidenceItem], *, sentence_limit: int
    ) -> SynthesizedAnswer:
        prediction = _predict(
            self.lm,
            self.predictor,
            question=question,
            evidence=format_evidence(evidence),
            max_sentences=sentence_limit,
        )
        text = str(prediction.answer or "").strip()
        known = {item.citation_id for item in evidence}
        cited = tuple(
            dict.fromkeys(
                identifier
                for raw in prediction.citation_ids or []
                if (identifier := str(raw).strip("[] ")) in known
            )
        )
        if not text or not cited:
            # An answer with no surviving citation is an uncited claim, so it is not an answer.
            return SynthesizedAnswer(text="", citation_ids=())
        return SynthesizedAnswer(text=text, citation_ids=cited)


@dataclass(slots=True)
class DspyClaimVerifier:
    lm: Any
    predictor: Any = field(default_factory=lambda: dspy.Predict(VerifyClaims))
    revision: str = "verifier-dspy-uncompiled"

    def verify(
        self, answer: SynthesizedAnswer, evidence: Sequence[EvidenceItem]
    ) -> VerificationResult:
        if not answer.text.strip():
            return VerificationResult(grounded=False, unsupported_claims=())
        cited = [item for item in evidence if item.citation_id in answer.citation_ids]
        if not cited:
            return VerificationResult(grounded=False, unsupported_claims=(answer.text.strip(),))
        raw = _predict(
            self.lm, self.predictor, answer=answer.text, evidence=format_evidence(cited)
        ).unsupported_claims
        unsupported = tuple(str(claim).strip() for claim in raw or [] if str(claim).strip())
        return VerificationResult(grounded=not unsupported, unsupported_claims=unsupported)


@dataclass(slots=True)
class DspyClarificationGenerator:
    lm: Any
    predictor: Any = field(default_factory=lambda: dspy.Predict(GenerateClarification))
    revision: str = "clarifier-dspy-uncompiled"
    fallback: TemplateClarificationGenerator = field(
        default_factory=TemplateClarificationGenerator
    )

    def generate(self, question: str, classification: QueryClassification) -> str:
        try:
            text = str(_predict(self.lm, self.predictor, question=question).clarification).strip()
        except Exception:
            return self.fallback.generate(question, classification)
        return text or self.fallback.generate(question, classification)


# Explicit metrics --------------------------------------------------------------


def intent_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    return bool(str(prediction.intent).strip() == example.intent)


def rewrite_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    """The rewrite must add every expected index term and must not drop the question."""
    rewritten = set(tokenize(str(prediction.rewritten)))
    return set(example.expected_terms) <= rewritten


def decomposition_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    return bool(len(prediction.subqueries or []) == example.expected_count)


def synthesis_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    """Cites exactly the expected passages, mentions the expected terms, and stays in budget."""
    cited = {str(identifier).strip("[] ") for identifier in prediction.citation_ids or []}
    answer = str(prediction.answer or "")
    return (
        cited == set(example.expected_citation_ids)
        and all(term.lower() in answer.lower() for term in example.expected_terms)
        and len(split_sentences(answer)) <= example.max_sentences
    )


def verification_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    return bool(prediction.unsupported_claims) == (not example.grounded)


def clarification_metric(example: Any, prediction: Any, trace: Any = None) -> bool:
    text = str(prediction.clarification or "").strip()
    return text.endswith("?") and len(text) <= 300


# Splits, compilation and versioned artifacts -----------------------------------


@dataclass(frozen=True, slots=True)
class ProgramDataset:
    train: tuple[Any, ...]
    heldout: tuple[Any, ...]


def split_examples(examples: Sequence[Any], *, heldout_every: int = 3) -> ProgramDataset:
    """A deterministic split: every `heldout_every`-th example is held out of training."""
    if heldout_every < 2:
        raise ValueError("heldout_every must be at least 2 so some examples remain for training")
    train = tuple(item for index, item in enumerate(examples) if (index + 1) % heldout_every)
    heldout = tuple(
        item for index, item in enumerate(examples) if not (index + 1) % heldout_every
    )
    return ProgramDataset(train=train, heldout=heldout)


def score(predictor: Any, examples: Sequence[Any], metric: Metric, lm: Any) -> float:
    if not examples:
        return 0.0
    passed = 0
    for example in examples:
        try:
            prediction = _predict(lm, predictor, **example.inputs())
        except Exception:
            continue
        passed += bool(metric(example, prediction, None))
    return passed / len(examples)


def state_revision(name: str, predictor: Any) -> str:
    state = json.dumps(_state_without_lm(predictor), sort_keys=True, default=str)
    return f"{name}-dspy-{hashlib.sha256(state.encode('utf-8')).hexdigest()[:10]}"


def _state_without_lm(predictor: Any) -> dict[str, Any]:
    state = dict(predictor.dump_state())
    state.pop("lm", None)
    return state


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    name: str
    predictor: Any
    revision: str
    train_examples: int
    heldout_examples: int
    train_score: float
    heldout_score: float

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "revision": self.revision,
                    "train_examples": self.train_examples,
                    "heldout_examples": self.heldout_examples,
                    "train_score": self.train_score,
                    "heldout_score": self.heldout_score,
                    "state": _state_without_lm(self.predictor),
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        return path


def compile_program(
    name: str,
    signature: type[Any],
    examples: Sequence[Any],
    metric: Metric,
    lm: Any,
    *,
    max_demos: int = 4,
    heldout_every: int = 3,
) -> CompiledProgram:
    """Bootstrap demonstrations on the training split and score on the held-out split."""
    dataset = split_examples(examples, heldout_every=heldout_every)
    with dspy.context(lm=lm):
        predictor = dspy.BootstrapFewShot(
            metric=metric, max_bootstrapped_demos=max_demos, max_labeled_demos=max_demos
        ).compile(dspy.Predict(signature), trainset=list(dataset.train))
    return CompiledProgram(
        name=name,
        predictor=predictor,
        revision=state_revision(name, predictor),
        train_examples=len(dataset.train),
        heldout_examples=len(dataset.heldout),
        train_score=score(predictor, dataset.train, metric, lm),
        heldout_score=score(predictor, dataset.heldout, metric, lm),
    )


def load_program(name: str, signature: type[Any], directory: Path) -> tuple[Any, str] | None:
    """The compiled predictor and its recorded revision, or None when no artifact exists."""
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    predictor = dspy.Predict(signature)
    # The artifact records the program, not the model it ran on: which model serves a program is
    # deployment configuration, and it is supplied when the suite is built.
    predictor.load_state({**artifact["state"], "lm": None})
    return predictor, str(artifact["revision"])


SIGNATURES: dict[str, type[Any]] = {
    "classifier": ClassifyQuery,
    "rewriter": RewriteQuery,
    "decomposer": DecomposeQuery,
    "synthesizer": SynthesizeAnswer,
    "verifier": VerifyClaims,
    "clarifier": GenerateClarification,
}


def build_program_suite(lm: Any, *, artifacts: Path | None = None) -> ProgramSuite:
    """A `ProgramSuite` on DSPy, using compiled artifacts where they exist."""
    loaded: dict[str, tuple[Any, str]] = {}
    for name, signature in SIGNATURES.items():
        artifact = load_program(name, signature, artifacts) if artifacts is not None else None
        loaded[name] = artifact or (dspy.Predict(signature), f"{name}-dspy-uncompiled")

    def part(name: str) -> dict[str, Any]:
        predictor, revision = loaded[name]
        return {"lm": lm, "predictor": predictor, "revision": revision}

    revisions = "|".join(revision for _predictor, revision in loaded.values())
    return ProgramSuite(
        revision=f"programs-dspy-{hashlib.sha256(revisions.encode('utf-8')).hexdigest()[:10]}",
        classifier=DspyQueryClassifier(**part("classifier")),
        rewriter=DspyQueryRewriter(**part("rewriter")),
        decomposer=DspyQueryDecomposer(**part("decomposer")),
        synthesizer=DspyAnswerSynthesizer(**part("synthesizer")),
        verifier=DspyClaimVerifier(**part("verifier")),
        clarifier=DspyClarificationGenerator(**part("clarifier")),
    )

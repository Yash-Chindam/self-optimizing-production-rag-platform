"""Sensitive-data recognition with Microsoft Presidio (specification sections 6 and 15).

`PresidioRecognizer` is a `PiiRecognizer`, so it replaces the regex stand-in inside
`PiiProcessor` without touching pseudonymization, the vault or the rehydration rules. It runs
in-process — "inside the trusted boundary" — and nothing leaves the machine.

What it adds over the regex recognizers is named-entity recognition: a person's name has no
pattern, so only a model finds it. A `PiiPolicy` opts into a kind by naming it, exactly as
before; a kind Presidio is not asked for is never analyzed.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from presidio_analyzer import AnalyzerEngine

KIND_TO_ENTITY: dict[str, str] = {
    "email": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER",
    "national_id": "US_SSN",
    "person": "PERSON",
    "credit_card": "CREDIT_CARD",
    "iban": "IBAN_CODE",
    "ip_address": "IP_ADDRESS",
    "location": "LOCATION",
}
ENTITY_TO_KIND = {entity: kind for kind, entity in KIND_TO_ENTITY.items()}

DEFAULT_SPACY_MODEL = "en_core_web_sm"


def build_analyzer(model_name: str = DEFAULT_SPACY_MODEL, language: str = "en") -> "AnalyzerEngine":
    """An analyzer on a named spaCy model, rather than Presidio's large default download."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": model_name}],
        }
    )
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=[language])


@dataclass(slots=True)
class PresidioRecognizer:
    analyzer: "AnalyzerEngine"
    kinds: tuple[str, ...]
    language: str = "en"
    score_threshold: float = 0.4

    def detect(self, text: str) -> list[tuple[str, int, int]]:
        entities = [KIND_TO_ENTITY[kind] for kind in self.kinds if kind in KIND_TO_ENTITY]
        if not entities or not text.strip():
            return []
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=entities,
            score_threshold=self.score_threshold,
        )
        return _resolve_overlaps(results)


@dataclass(slots=True)
class PresidioRecognizerFactory:
    """What `PiiProcessor.recognizer_factory` takes: one analyzer, a recognizer per policy."""

    analyzer: "AnalyzerEngine"
    language: str = "en"
    score_threshold: float = 0.4

    def __call__(self, kinds: tuple[str, ...]) -> PresidioRecognizer:
        return PresidioRecognizer(
            analyzer=self.analyzer,
            kinds=kinds,
            language=self.language,
            score_threshold=self.score_threshold,
        )


def _resolve_overlaps(results: list[Any]) -> list[tuple[str, int, int]]:
    """Keep the most confident span where two overlap; a longer span wins a tie."""
    ranked = sorted(
        results, key=lambda item: (-float(item.score), -(item.end - item.start), item.start)
    )
    kept: list[Any] = []
    for candidate in ranked:
        if all(candidate.end <= other.start or candidate.start >= other.end for other in kept):
            kept.append(candidate)
    return sorted(
        ((ENTITY_TO_KIND[item.entity_type], int(item.start), int(item.end)) for item in kept),
        key=lambda span: span[1],
    )

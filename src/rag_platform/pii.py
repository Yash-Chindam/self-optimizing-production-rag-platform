"""Sensitive-data processing inside the trusted boundary.

The recognizers here are deterministic stand-ins with the same interface a Presidio analyzer
adapter would implement. Pseudonym mappings are held in a vault that is deliberately separate
from the index stores, and rehydration requires an explicitly approved workflow.
"""

import re
from dataclasses import dataclass, field
from hashlib import blake2s
from typing import Literal, Protocol

from rag_platform.models import PiiPolicy

RECOGNIZER_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){9,13}\d(?!\d)"),
    "national_id": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
}


@dataclass(frozen=True, slots=True)
class PiiEntity:
    kind: str
    pseudonym: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PiiResult:
    text: str
    entities: tuple[PiiEntity, ...]


@dataclass(frozen=True, slots=True)
class RehydrationApproval:
    """Evidence that a rehydration request was approved by a named workflow."""

    workflow: Literal["legal_hold", "data_subject_request", "incident_review"]
    approver: str
    justification: str


class PiiRecognizer(Protocol):
    def detect(self, text: str) -> list[tuple[str, int, int]]: ...


class RegexRecognizer:
    def __init__(self, kinds: tuple[str, ...]) -> None:
        self._kinds = tuple(kind for kind in kinds if kind in RECOGNIZER_PATTERNS)

    def detect(self, text: str) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        for kind in self._kinds:
            for match in RECOGNIZER_PATTERNS[kind].finditer(text):
                spans.append((kind, match.start(), match.end()))
        spans.sort(key=lambda span: (span[1], -span[2]))
        return _drop_overlaps(spans)


def _drop_overlaps(spans: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    kept: list[tuple[str, int, int]] = []
    boundary = -1
    for kind, start, end in spans:
        if start >= boundary:
            kept.append((kind, start, end))
            boundary = end
    return kept


class PseudonymVault:
    """Pseudonym-to-original mappings, stored apart from every retrieval index."""

    def __init__(self) -> None:
        self._mappings: dict[tuple[str, str], str] = {}

    def register(self, tenant_id: str, pseudonym: str, original: str) -> None:
        self._mappings[(tenant_id, pseudonym)] = original

    def rehydrate(self, tenant_id: str, pseudonym: str, approval: RehydrationApproval) -> str:
        if not approval.approver.strip() or not approval.justification.strip():
            raise PermissionError("rehydration requires a named approver and a justification")
        try:
            return self._mappings[(tenant_id, pseudonym)]
        except KeyError as error:
            raise KeyError(f"no pseudonym mapping for {pseudonym!r}") from error

    def __len__(self) -> int:
        return len(self._mappings)


@dataclass(slots=True)
class PiiProcessor:
    vault: PseudonymVault = field(default_factory=PseudonymVault)

    def process(self, text: str, policy: PiiPolicy, tenant_id: str) -> PiiResult:
        if policy.mode == "off":
            return PiiResult(text=text, entities=())
        spans = RegexRecognizer(policy.recognizers).detect(text)
        if not spans:
            return PiiResult(text=text, entities=())

        entities: list[PiiEntity] = []
        rewritten: list[str] = []
        cursor = 0
        for kind, start, end in spans:
            original = text[start:end]
            pseudonym = _pseudonym(tenant_id, kind, original)
            entities.append(PiiEntity(kind=kind, pseudonym=pseudonym, start=start, end=end))
            if policy.mode == "pseudonymize":
                self.vault.register(tenant_id, pseudonym, original)
                rewritten.append(text[cursor:start])
                rewritten.append(pseudonym)
                cursor = end
        if policy.mode == "detect":
            return PiiResult(text=text, entities=tuple(entities))
        rewritten.append(text[cursor:])
        return PiiResult(text="".join(rewritten), entities=tuple(entities))


def _pseudonym(tenant_id: str, kind: str, value: str) -> str:
    digest = blake2s(f"{tenant_id}:{kind}:{value}".encode(), digest_size=4).hexdigest()
    return f"[{kind.upper()}_{digest}]"

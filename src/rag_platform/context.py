"""Context construction (specification section 8).

Deduplicate overlapping chunks, expand parent context only when it is needed, keep evidence
diverse, enforce a character budget, and attach citation identifiers before generation.
Authorization is rechecked here even though retrieval already filtered: context assembly is the
last point before evidence reaches a generator, and the recheck is cheap.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.repository import ScoredChunk, is_authorized, tokenize

ParentLookup = Callable[[str, str], DocumentChunk | None]


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    chunk: DocumentChunk
    citation_id: str
    score: float
    expanded_to_parent: bool


@dataclass(frozen=True, slots=True)
class ContextBundle:
    items: tuple[EvidenceItem, ...]
    used_characters: int
    dropped_chunk_ids: tuple[str, ...]
    policy_notes: tuple[str, ...]


class ContextBuilder:
    def __init__(
        self, config: PipelineConfig, parent_lookup: ParentLookup | None = None
    ) -> None:
        self._config = config
        self._parent_lookup = parent_lookup

    def build(self, ranked: Sequence[ScoredChunk], access: AccessContext) -> ContextBundle:
        dropped: list[str] = []
        notes: list[str] = []

        authorized = self._recheck_authorization(ranked, access, dropped, notes)
        deduplicated = self._drop_near_duplicates(authorized, dropped, notes)
        diverse = self._limit_per_source(deduplicated, dropped, notes)
        return self._fill_budget(diverse, access, dropped, notes)

    def _recheck_authorization(
        self,
        ranked: Iterable[ScoredChunk],
        access: AccessContext,
        dropped: list[str],
        notes: list[str],
    ) -> list[ScoredChunk]:
        kept: list[ScoredChunk] = []
        for item in ranked:
            if is_authorized(item.chunk, access):
                kept.append(item)
                continue
            dropped.append(item.chunk.chunk_id)
            notes.append(f"authorization_recheck_removed:{item.chunk.chunk_id}")
        return kept

    def _drop_near_duplicates(
        self, ranked: Sequence[ScoredChunk], dropped: list[str], notes: list[str]
    ) -> list[ScoredChunk]:
        kept: list[ScoredChunk] = []
        signatures: list[frozenset[str]] = []
        for item in ranked:
            signature = frozenset(tokenize(item.chunk.text))
            if any(
                _jaccard(signature, existing) >= self._config.near_duplicate_threshold
                for existing in signatures
            ):
                dropped.append(item.chunk.chunk_id)
                notes.append(f"near_duplicate_removed:{item.chunk.chunk_id}")
                continue
            signatures.append(signature)
            kept.append(item)
        return kept

    def _limit_per_source(
        self, ranked: Sequence[ScoredChunk], dropped: list[str], notes: list[str]
    ) -> list[ScoredChunk]:
        counts: dict[str, int] = {}
        kept: list[ScoredChunk] = []
        for item in ranked:
            source = item.chunk.source_version_id
            if counts.get(source, 0) >= self._config.max_chunks_per_source:
                dropped.append(item.chunk.chunk_id)
                notes.append(f"evidence_diversity_removed:{item.chunk.chunk_id}")
                continue
            counts[source] = counts.get(source, 0) + 1
            kept.append(item)
        return kept

    def _fill_budget(
        self,
        ranked: Sequence[ScoredChunk],
        access: AccessContext,
        dropped: list[str],
        notes: list[str],
    ) -> ContextBundle:
        items: list[EvidenceItem] = []
        selected_ids: set[str] = set()
        used = 0
        for item in ranked:
            chunk, expanded = self._maybe_expand(item.chunk, access, self._remaining(used))
            if chunk.chunk_id in selected_ids:
                # Two fragments of one section must not enter the context twice.
                dropped.append(item.chunk.chunk_id)
                notes.append(f"parent_already_in_context:{item.chunk.chunk_id}")
                continue
            size = len(chunk.text)
            if used + size > self._config.context_character_budget:
                dropped.append(item.chunk.chunk_id)
                notes.append(f"context_budget_removed:{item.chunk.chunk_id}")
                continue
            used += size
            selected_ids.add(chunk.chunk_id)
            items.append(
                EvidenceItem(
                    chunk=chunk,
                    citation_id=f"C{len(items) + 1}",
                    score=item.score,
                    expanded_to_parent=expanded,
                )
            )
            if expanded:
                notes.append(f"parent_context_expanded:{chunk.chunk_id}")
        return ContextBundle(
            items=tuple(items),
            used_characters=used,
            dropped_chunk_ids=tuple(dropped),
            policy_notes=tuple(notes),
        )

    def _remaining(self, used: int) -> int:
        return self._config.context_character_budget - used

    def _maybe_expand(
        self, chunk: DocumentChunk, access: AccessContext, remaining: int
    ) -> tuple[DocumentChunk, bool]:
        """Replace a fragment with its parent section when the parent still fits."""
        if (
            not self._config.expand_parent_context
            or self._parent_lookup is None
            or chunk.parent_chunk_id is None
        ):
            return chunk, False
        parent = self._parent_lookup(access.tenant_id, chunk.parent_chunk_id)
        if parent is None or not is_authorized(parent, access):
            return chunk, False
        if len(parent.text) > remaining:
            return chunk, False
        return parent, True


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

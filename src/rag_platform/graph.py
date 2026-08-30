"""Graph retrieval over an entity and provenance graph.

Graph retrieval complements dense and lexical retrieval instead of replacing it: it starts from
chunks the other strategies already found and follows entity relationships to evidence that
shares no vocabulary with the question. Every edge records the chunk it was extracted from, so
an expanded result can always be traced back to its provenance.
"""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from rag_platform.catalog import IndexCatalog
from rag_platform.models import AccessContext, DocumentChunk
from rag_platform.repository import is_authorized

ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][\w-]*)(?:\s+(?:[A-Z][\w-]*|of|for|and))*\b")
RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("reports_to", re.compile(r"(.+?)\s+reports? to\s+(.+)")),
    ("depends_on", re.compile(r"(.+?)\s+depends? on\s+(.+)")),
    ("owned_by", re.compile(r"(.+?)\s+is owned by\s+(.+)")),
    ("approved_by", re.compile(r"(.+?)\s+is approved by\s+(.+)")),
    ("governed_by", re.compile(r"(.+?)\s+is governed by\s+(.+)")),
    ("derived_from", re.compile(r"(.+?)\s+is derived from\s+(.+)")),
)
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+")
ENTITY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "every",
        "for",
        "if",
        "in",
        "it",
        "of",
        "the",
        "this",
        "when",
        "where",
        "which",
        "who",
    }
)


@dataclass(frozen=True, slots=True)
class Relationship:
    subject: str
    relation: str
    object: str
    evidence_chunk_id: str


ARTICLES = ("the ", "a ", "an ")


def normalize_entity(text: str) -> str:
    cleaned = " ".join(text.split()).strip(" .,:;").lower()
    for article in ARTICLES:
        if cleaned.startswith(article):
            return cleaned[len(article) :]
    return cleaned


def extract_entities(text: str) -> set[str]:
    """Capitalized phrases and acronyms.

    A single capitalized word that merely opens a sentence is not an entity, so sentence
    position is part of the rule rather than a hand-maintained word list.
    """
    entities: set[str] = set()
    for sentence in SENTENCE_PATTERN.findall(text):
        stripped = sentence.lstrip()
        offset = len(sentence) - len(stripped)
        for match in ENTITY_PATTERN.finditer(sentence):
            candidate = normalize_entity(match.group(0))
            if not candidate or candidate in ENTITY_STOP_WORDS:
                continue
            words = match.group(0).split()
            is_acronym = len(words) == 1 and words[0].isupper() and len(words[0]) > 1
            if len(words) == 1 and not is_acronym and match.start() == offset:
                continue
            if len(candidate) < (2 if is_acronym else 3):
                continue
            entities.add(candidate)
    return entities


def extract_relationships(chunk: DocumentChunk) -> list[Relationship]:
    relationships: list[Relationship] = []
    for sentence in SENTENCE_PATTERN.findall(chunk.text):
        for relation, pattern in RELATION_PATTERNS:
            match = pattern.search(sentence)
            if match is None:
                continue
            subject = _leading_entity(match.group(1))
            target = _leading_entity(match.group(2))
            if subject and target:
                relationships.append(
                    Relationship(
                        subject=subject,
                        relation=relation,
                        object=target,
                        evidence_chunk_id=chunk.chunk_id,
                    )
                )
    return relationships


def _leading_entity(fragment: str) -> str:
    """Reduce a relation side to the entity key used for mentions."""
    candidates = extract_entities(fragment)
    if not candidates:
        return ""
    return max(candidates, key=len)


@dataclass(slots=True)
class KnowledgeGraph:
    """Entity mentions, typed relationships and the chunk that evidences each edge."""

    revision: str = "graph-extractor-v1"
    _mentions: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _relationships: list[Relationship] = field(default_factory=list)

    def index(self, chunks: Iterable[DocumentChunk]) -> None:
        for chunk in chunks:
            for entity in extract_entities(chunk.text):
                self._mentions[entity].add(chunk.chunk_id)
            for relationship in extract_relationships(chunk):
                self._relationships.append(relationship)
                self._edges[relationship.subject].add(relationship.object)
                self._edges[relationship.object].add(relationship.subject)

    def relationships(self) -> tuple[Relationship, ...]:
        return tuple(self._relationships)

    def entities_of(self, chunk: DocumentChunk) -> set[str]:
        return extract_entities(chunk.text)

    def related(self, entities: Iterable[str], depth: int) -> set[str]:
        """Entities reachable from the given ones in one to `depth` relationship hops."""
        reached: set[str] = set()
        frontier = set(entities)
        for _hop in range(depth):
            expanded = {
                neighbour for entity in frontier for neighbour in self._edges.get(entity, set())
            }
            new = expanded - reached
            if not new:
                break
            reached |= new
            frontier = new
        return reached

    def chunk_ids_for(self, entities: Iterable[str]) -> set[str]:
        return {
            chunk_id for entity in entities for chunk_id in self._mentions.get(entity, set())
        }


class GraphExpander(Protocol):
    def expand(
        self,
        seeds: Iterable[DocumentChunk],
        access: AccessContext,
        *,
        depth: int,
        limit: int,
    ) -> list[DocumentChunk]: ...


class GraphRetriever:
    """Expands a seed result set along entity relationships within the access boundary."""

    def __init__(self, graph: KnowledgeGraph, chunks: Iterable[DocumentChunk]) -> None:
        self._graph = graph
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def expand(
        self,
        seeds: Iterable[DocumentChunk],
        access: AccessContext,
        *,
        depth: int,
        limit: int,
    ) -> list[DocumentChunk]:
        seed_chunks = list(seeds)
        if depth <= 0 or limit <= 0 or not seed_chunks:
            return []
        seed_ids = {chunk.chunk_id for chunk in seed_chunks}
        seed_entities = {
            entity for chunk in seed_chunks for entity in self._graph.entities_of(chunk)
        }
        # Only entities reached through relationship edges expand the candidate set; sharing
        # a mention with a seed chunk is not on its own a reason to retrieve.
        related = self._graph.related(seed_entities, depth)
        candidate_ids = self._graph.chunk_ids_for(related) - seed_ids

        expanded = [
            chunk
            for chunk_id in sorted(candidate_ids)
            if (chunk := self._chunks.get(chunk_id)) is not None
            and chunk.retrievable
            and is_authorized(chunk, access)
        ]
        return expanded[:limit]


class CatalogGraphRetriever:
    """Graph retrieval over the tenant's active index version.

    The graph is rebuilt when the active index version changes, so activation and rollback move
    the graph and the vector and lexical indexes together.
    """

    def __init__(self, catalog: IndexCatalog, revision: str = "graph-extractor-v1") -> None:
        self._catalog = catalog
        self._revision = revision
        self._cache: dict[str, tuple[str, GraphRetriever]] = {}

    def expand(
        self,
        seeds: Iterable[DocumentChunk],
        access: AccessContext,
        *,
        depth: int,
        limit: int,
    ) -> list[DocumentChunk]:
        retriever = self._retriever_for(access.tenant_id)
        if retriever is None:
            return []
        return retriever.expand(seeds, access, depth=depth, limit=limit)

    def _retriever_for(self, tenant_id: str) -> GraphRetriever | None:
        active = self._catalog.active_index_version(tenant_id)
        if active is None:
            return None
        cached = self._cache.get(tenant_id)
        if cached is not None and cached[0] == active.index_version_id:
            return cached[1]
        chunks = self._catalog.active_chunks(tenant_id)
        graph = KnowledgeGraph(revision=self._revision)
        graph.index(chunks)
        retriever = GraphRetriever(graph, chunks)
        self._cache[tenant_id] = (active.index_version_id, retriever)
        return retriever

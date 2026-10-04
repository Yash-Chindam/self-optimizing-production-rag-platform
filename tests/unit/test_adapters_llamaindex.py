"""LlamaIndex ingestion composition.

The chunker is swappable only because its output obeys the built-in chunker's contract, so
these tests check the contract (bounds, heading context, parent links) and then run a whole
ingest and a query on top of it.
"""

import pytest

from rag_platform.catalog import IndexCatalog
from rag_platform.chunking import build_pieces
from rag_platform.context import ContextBuilder
from rag_platform.ingestion import IngestionPipeline
from rag_platform.models import AccessContext, ChunkingConfig, PipelineConfig, SourceRegistration
from rag_platform.repository import CatalogChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService

pytest.importorskip("llama_index.core")

from rag_platform.adapters.llamaindex_ingestion import (
    REVISION,
    LlamaIndexPieceBuilder,
    extract_sections,
)

HANDBOOK = """# Handbook

## Annual leave

Annual leave requests use the HR portal. Managers approve within two working days.

## Expenses

### Travel

Travel expenses need a receipt.
"""

SENTENCES = [
    f"Sentence number {index} explains one travel rule in plain words." for index in range(12)
]
LONG = "# Policy\n\n## Travel\n\n" + " ".join(SENTENCES) + "\n"
SMALL = ChunkingConfig(max_characters=200, overlap_characters=30)

REGISTRATION = SourceRegistration(
    source_id="handbook",
    tenant_id="tenant-a",
    owner="people-operations",
    source_uri="https://example.test/handbook",
    source_title="Handbook",
)


def test_markdown_structure_becomes_section_paths() -> None:
    sections = extract_sections(HANDBOOK, "markdown")
    assert [section.path for section in sections] == [
        ("Handbook", "Annual leave"),
        ("Handbook", "Expenses", "Travel"),
    ]
    assert sections[0].text.startswith("Annual leave requests use the HR portal.")
    assert [section.ordinal for section in sections] == [0, 1]


def test_headings_without_a_body_produce_no_section() -> None:
    assert extract_sections("# Title only\n\n## Empty\n", "markdown") == []


def test_plain_text_is_one_section_without_a_path() -> None:
    [section] = extract_sections("First paragraph.\n\nSecond paragraph.", "text")
    assert section.path == ()
    assert section.text == "First paragraph. Second paragraph."
    assert extract_sections("   \n", "text") == []


def test_a_short_section_is_one_retrievable_piece_with_its_heading_context() -> None:
    pieces = LlamaIndexPieceBuilder()(HANDBOOK, "markdown", ChunkingConfig())

    assert [piece.retrievable for piece in pieces] == [True, True]
    assert pieces[0].text.startswith("Handbook > Annual leave: Annual leave requests")
    assert pieces[1].section_path == ("Handbook", "Expenses", "Travel")
    assert [piece.ordinal for piece in pieces] == [0, 1]


def test_a_long_section_is_split_within_the_character_budget() -> None:
    pieces = LlamaIndexPieceBuilder()(LONG, "markdown", SMALL)
    children = [piece for piece in pieces if piece.retrievable]

    assert len(children) > 1
    assert all(len(piece.text) <= SMALL.max_characters for piece in children)
    assert all(piece.text.startswith("Policy > Travel: ") for piece in children)


def test_windows_break_on_sentence_boundaries() -> None:
    """The reason to prefer this splitter: the built-in one cuts mid-sentence."""
    pieces = LlamaIndexPieceBuilder()(LONG, "markdown", SMALL)
    children = [piece for piece in pieces if piece.retrievable]

    for piece in children:
        body = piece.text.removeprefix("Policy > Travel: ")
        assert body.startswith("Sentence number")
        assert body.endswith("plain words.")

    builtin = [piece for piece in build_pieces(LONG, "markdown", SMALL) if piece.retrievable]
    assert not all(
        piece.text.removeprefix("Policy > Travel: ").startswith("Sentence") for piece in builtin
    )


def test_every_sentence_survives_splitting() -> None:
    pieces = LlamaIndexPieceBuilder()(LONG, "markdown", SMALL)
    joined = " ".join(piece.text for piece in pieces if piece.retrievable)
    assert all(sentence in joined for sentence in SENTENCES)


def test_split_sections_keep_a_non_retrievable_parent() -> None:
    pieces = LlamaIndexPieceBuilder()(LONG, "markdown", SMALL)
    parent, *children = pieces

    assert parent.retrievable is False
    assert parent.parent_ordinal is None
    assert all(sentence in parent.text for sentence in SENTENCES)
    assert all(child.parent_ordinal == parent.ordinal for child in children)


def test_parent_chunks_can_be_turned_off() -> None:
    flat = SMALL.model_copy(update={"parent_child": False})
    pieces = LlamaIndexPieceBuilder()(LONG, "markdown", flat)
    assert all(piece.retrievable and piece.parent_ordinal is None for piece in pieces)


def test_the_token_strategy_ignores_document_structure() -> None:
    config = SMALL.model_copy(update={"strategy": "token"})
    pieces = LlamaIndexPieceBuilder()(LONG, "markdown", config)
    assert all(piece.section_path == () for piece in pieces)
    assert LlamaIndexPieceBuilder()("   ", "text", config) == []


def test_ingestion_and_retrieval_work_unchanged_on_llamaindex_chunks() -> None:
    catalog = IndexCatalog()
    pipeline = IngestionPipeline(
        catalog=catalog,
        chunking=SMALL,
        piece_builder=LlamaIndexPieceBuilder(),
        transformation_revision=REVISION,
    )
    result = pipeline.ingest(REGISTRATION, HANDBOOK + "\n" + LONG.replace("# Policy", "## Policy"))

    assert result.source_version.transformation_revision == "transform-llamaindex-v1"
    assert "retrieval_smoke_passed" in result.report.validation_checks
    chunks = catalog.active_chunks("tenant-a")
    assert any(not chunk.retrievable for chunk in chunks)
    children = [chunk for chunk in chunks if chunk.parent_chunk_id is not None]
    assert children and all(
        catalog.chunk("tenant-a", chunk.parent_chunk_id or "") for chunk in children
    )

    config = PipelineConfig()
    service = QueryService(
        HybridRetriever(CatalogChunkRepository(catalog), config),
        config,
        ContextBuilder(config, catalog.chunk),
    )
    response = service.answer("How do I request annual leave?", AccessContext(tenant_id="tenant-a"))
    assert response.status == "answered"
    assert "HR portal" in response.answer

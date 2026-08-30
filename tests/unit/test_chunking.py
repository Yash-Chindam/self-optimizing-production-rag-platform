import pytest

from rag_platform.chunking import build_pieces, extract_sections, split_windows
from rag_platform.models import ChunkingConfig

HANDBOOK = """# Handbook

Introductory sentence.

## Leave

### Annual leave

Requests use the HR portal.

## Expenses

Claims are monthly.
"""


def test_markdown_sections_carry_their_heading_path() -> None:
    sections = extract_sections(HANDBOOK, "markdown")
    assert [section.path for section in sections] == [
        ("Handbook",),
        ("Handbook", "Leave", "Annual leave"),
        ("Handbook", "Expenses"),
    ]
    assert sections[1].text == "Requests use the HR portal."


def test_plain_text_sections_split_on_blank_lines() -> None:
    sections = extract_sections("First para.\n\nSecond\npara.\n", "text")
    assert [section.text for section in sections] == ["First para.", "Second para."]


def test_windows_respect_the_budget_and_overlap() -> None:
    text = " ".join(f"word{index}" for index in range(60))
    windows = split_windows(text, 100, 30)
    assert len(windows) > 1
    assert all(len(window) <= 100 for window in windows)
    assert windows[1].split()[0] in windows[0].split()


def test_windows_of_empty_text_are_empty() -> None:
    assert split_windows("   ", 100, 10) == []


def test_structural_chunks_keep_heading_context_within_the_budget() -> None:
    config = ChunkingConfig(max_characters=120, overlap_characters=20)
    pieces = build_pieces(HANDBOOK, "markdown", config)
    retrievable = [piece for piece in pieces if piece.retrievable]
    assert all(len(piece.text) <= config.max_characters for piece in retrievable)
    assert retrievable[1].text.startswith("Handbook > Leave > Annual leave: ")


def test_oversized_sections_produce_a_parent_and_children() -> None:
    long_section = "# Title\n\n" + ("sentence about leave. " * 60)
    pieces = build_pieces(
        long_section, "markdown", ChunkingConfig(max_characters=200, overlap_characters=40)
    )
    parents = [piece for piece in pieces if not piece.retrievable]
    children = [piece for piece in pieces if piece.retrievable]
    assert len(parents) == 1
    assert len(children) > 1
    assert {piece.parent_ordinal for piece in children} == {parents[0].ordinal}


def test_parent_child_can_be_disabled() -> None:
    long_section = "# Title\n\n" + ("sentence about leave. " * 60)
    config = ChunkingConfig(max_characters=200, overlap_characters=40, parent_child=False)
    pieces = build_pieces(long_section, "markdown", config)
    assert all(piece.retrievable for piece in pieces)
    assert all(piece.parent_ordinal is None for piece in pieces)


def test_token_strategy_ignores_document_structure() -> None:
    config = ChunkingConfig(strategy="token", max_characters=120, overlap_characters=20)
    pieces = build_pieces(HANDBOOK, "markdown", config)
    assert all(piece.section_path == () for piece in pieces)
    assert "# Handbook" in pieces[0].text


def test_overlap_must_be_smaller_than_the_window() -> None:
    with pytest.raises(ValueError, match="overlap_characters"):
        ChunkingConfig(max_characters=100, overlap_characters=100)


def test_blank_content_produces_no_pieces() -> None:
    assert build_pieces("   \n\n  ", "text", ChunkingConfig()) == []
    assert build_pieces("   ", "markdown", ChunkingConfig(strategy="token")) == []

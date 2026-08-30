"""Structure extraction and chunk generation.

Structural chunking keeps heading context with the evidence it introduces, which is the
mitigation for the "evidence divided into unusable fragments" failure category. Token chunking
is retained because it is part of the evaluated candidate configuration space.
"""

import re
from dataclasses import dataclass

from rag_platform.models import ChunkingConfig

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MINIMUM_WINDOW_CHARACTERS = 40


@dataclass(frozen=True, slots=True)
class Section:
    path: tuple[str, ...]
    text: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class ChunkPiece:
    text: str
    section_path: tuple[str, ...]
    ordinal: int
    parent_ordinal: int | None
    retrievable: bool


def extract_sections(text: str, document_type: str) -> list[Section]:
    if document_type == "markdown":
        return _markdown_sections(text)
    return _paragraph_sections(text)


def _markdown_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(line for line in buffer if line.strip()).strip()
        buffer.clear()
        if body:
            sections.append(Section(path=tuple(stack), text=body, ordinal=len(sections)))

    for line in text.splitlines():
        heading = HEADING_PATTERN.match(line)
        if heading is None:
            buffer.append(line)
            continue
        flush()
        level = len(heading.group(1))
        del stack[level - 1 :]
        stack.append(heading.group(2))
    flush()
    return sections


def _paragraph_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    for block in re.split(r"\n\s*\n", text):
        body = " ".join(block.split())
        if body:
            sections.append(Section(path=(), text=body, ordinal=len(sections)))
    return sections


def build_pieces(text: str, document_type: str, config: ChunkingConfig) -> list[ChunkPiece]:
    if config.strategy == "token":
        sections = [Section(path=(), text=" ".join(text.split()), ordinal=0)]
    else:
        sections = extract_sections(text, document_type)

    pieces: list[ChunkPiece] = []
    for section in sections:
        # The heading prefix is part of the stored chunk, so it is charged against the budget.
        prefix_length = len(_heading_prefix(section))
        budget = max(MINIMUM_WINDOW_CHARACTERS, config.max_characters - prefix_length)
        overlap = min(config.overlap_characters, budget - 1)
        windows = split_windows(section.text, budget, overlap)
        if not windows:
            continue
        if len(windows) == 1:
            pieces.append(
                ChunkPiece(
                    text=_with_heading(section, windows[0]),
                    section_path=section.path,
                    ordinal=len(pieces),
                    parent_ordinal=None,
                    retrievable=True,
                )
            )
            continue

        parent_ordinal: int | None = None
        if config.parent_child:
            parent_ordinal = len(pieces)
            pieces.append(
                ChunkPiece(
                    text=_with_heading(section, section.text),
                    section_path=section.path,
                    ordinal=parent_ordinal,
                    parent_ordinal=None,
                    retrievable=False,
                )
            )
        for window in windows:
            pieces.append(
                ChunkPiece(
                    text=_with_heading(section, window),
                    section_path=section.path,
                    ordinal=len(pieces),
                    parent_ordinal=parent_ordinal,
                    retrievable=True,
                )
            )
    return pieces


def _heading_prefix(section: Section) -> str:
    if not section.path:
        return ""
    return f"{' > '.join(section.path)}: "


def _with_heading(section: Section, body: str) -> str:
    return f"{_heading_prefix(section)}{body}"


def split_windows(text: str, max_characters: int, overlap_characters: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    windows: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and length + addition > max_characters:
            windows.append(" ".join(current))
            current, length = _overlap_tail(current, overlap_characters)
            addition = len(word) + (1 if current else 0)
        current.append(word)
        length += addition
    if current:
        windows.append(" ".join(current))
    return windows


def _overlap_tail(words: list[str], overlap_characters: int) -> tuple[list[str], int]:
    carry: list[str] = []
    length = 0
    for word in reversed(words):
        addition = len(word) + (1 if carry else 0)
        if length + addition > overlap_characters:
            break
        carry.insert(0, word)
        length += addition
    return carry, length

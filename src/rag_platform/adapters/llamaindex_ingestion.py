"""LlamaIndex ingestion composition (specification sections 5 and 7).

LlamaIndex supplies the parser and the transformations: `MarkdownNodeParser` extracts the
document structure and `SentenceSplitter` cuts a long section on sentence boundaries. The
result is returned as the same `ChunkPiece` list the built-in chunker produces, so everything
downstream of chunking (identifiers, deduplication, parent links, validation, staged
activation) is unchanged and an index built with either chunker obeys the same contract.

Sizes are in characters, like `ChunkingConfig`: the splitter is given a character tokenizer, so
`max_characters` is an exact bound and nothing has to be downloaded to count tokens.
"""

from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline as LlamaIngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from rag_platform.chunking import MINIMUM_WINDOW_CHARACTERS, ChunkPiece, Section
from rag_platform.models import ChunkingConfig

REVISION = "transform-llamaindex-v1"


def _characters(text: str) -> list[str]:
    return list(text)


def extract_sections(text: str, document_type: str) -> list[Section]:
    """Sections from LlamaIndex's parsers, in document order, without empty bodies."""
    if document_type != "markdown":
        body = " ".join(text.split())
        return [Section(path=(), text=body, ordinal=0)] if body else []

    pipeline = LlamaIngestionPipeline(transformations=[MarkdownNodeParser()])
    nodes = pipeline.run(documents=[Document(text=text)])
    sections: list[Section] = []
    for node in nodes:
        heading, body = _split_heading(node.get_content())
        if not body:
            continue
        ancestors = tuple(
            part for part in str(node.metadata.get("header_path", "/")).split("/") if part
        )
        path = (*ancestors, heading) if heading else ancestors
        sections.append(Section(path=path, text=body, ordinal=len(sections)))
    return sections


def _split_heading(content: str) -> tuple[str, str]:
    lines = content.strip().splitlines()
    if lines and lines[0].startswith("#"):
        heading = lines[0].lstrip("#").strip()
        body = " ".join(" ".join(lines[1:]).split())
        return heading, body
    return "", " ".join(content.split())


@dataclass(frozen=True, slots=True)
class LlamaIndexPieceBuilder:
    """A `PieceBuilder` for `IngestionPipeline`, composed from LlamaIndex components."""

    revision: str = REVISION

    def __call__(self, text: str, document_type: str, config: ChunkingConfig) -> list[ChunkPiece]:
        if config.strategy == "token":
            body = " ".join(text.split())
            sections = [Section(path=(), text=body, ordinal=0)] if body else []
        else:
            sections = extract_sections(text, document_type)

        pieces: list[ChunkPiece] = []
        for section in sections:
            prefix = f"{' > '.join(section.path)}: " if section.path else ""
            budget = max(MINIMUM_WINDOW_CHARACTERS, config.max_characters - len(prefix))
            windows = self._windows(
                section.text, budget, min(config.overlap_characters, budget - 1)
            )
            if len(windows) == 1:
                pieces.append(
                    ChunkPiece(
                        text=f"{prefix}{windows[0]}",
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
                        text=f"{prefix}{section.text}",
                        section_path=section.path,
                        ordinal=parent_ordinal,
                        parent_ordinal=None,
                        retrievable=False,
                    )
                )
            for window in windows:
                pieces.append(
                    ChunkPiece(
                        text=f"{prefix}{window}",
                        section_path=section.path,
                        ordinal=len(pieces),
                        parent_ordinal=parent_ordinal,
                        retrievable=True,
                    )
                )
        return pieces

    @staticmethod
    def _windows(text: str, budget: int, overlap: int) -> list[str]:
        splitter = SentenceSplitter(chunk_size=budget, chunk_overlap=overlap, tokenizer=_characters)
        return [window.strip() for window in splitter.split_text(text) if window.strip()]

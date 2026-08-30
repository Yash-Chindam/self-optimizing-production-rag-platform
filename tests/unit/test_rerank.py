from rag_platform.models import DocumentChunk
from rag_platform.rerank import LexicalCrossEncoder


def chunk(identifier: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text=text,
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="index-v1",
    )


NEAR = chunk("near", "Annual leave requests use the HR portal.")
FAR = chunk("far", "Annual budget planning. " + ("filler text here. " * 20) + "Leave now.")
PARTIAL = chunk("partial", "Annual planning happens in December.")


def test_full_coverage_outranks_partial_coverage() -> None:
    ranked = LexicalCrossEncoder().rerank("annual leave", [PARTIAL, NEAR], limit=5)
    assert [item.chunk.chunk_id for item in ranked] == ["near", "partial"]


def test_proximity_breaks_ties_between_equally_covered_chunks() -> None:
    ranked = LexicalCrossEncoder().rerank("annual leave", [FAR, NEAR], limit=5)
    assert [item.chunk.chunk_id for item in ranked] == ["near", "far"]
    assert ranked[0].score > ranked[1].score


def test_chunks_without_any_query_term_are_dropped() -> None:
    unrelated = chunk("unrelated", "Catering guidance for the Dublin office.")
    assert LexicalCrossEncoder().rerank("annual leave", [unrelated], limit=5) == []


def test_an_empty_query_scores_nothing() -> None:
    assert LexicalCrossEncoder().rerank("!!!", [NEAR], limit=5) == []
    assert LexicalCrossEncoder().score((), NEAR) == 0.0


def test_single_term_queries_take_the_full_proximity_credit() -> None:
    ranked = LexicalCrossEncoder().rerank("leave", [NEAR], limit=5)
    assert ranked[0].score == 1.25


def test_the_limit_bounds_the_returned_candidates() -> None:
    ranked = LexicalCrossEncoder().rerank("annual leave", [NEAR, FAR, PARTIAL], limit=2)
    assert len(ranked) == 2


def test_the_revision_is_reported_for_provenance() -> None:
    assert LexicalCrossEncoder().revision == "reranker-lexical-v1"
    assert LexicalCrossEncoder(revision="reranker-v2").revision == "reranker-v2"

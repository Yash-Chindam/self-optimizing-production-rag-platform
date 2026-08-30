from rag_platform.models import AccessContext, DocumentChunk
from rag_platform.repository import InMemoryChunkRepository, is_authorized, tokenize


def test_tokenize_normalizes_semantic_aliases() -> None:
    assert tokenize("Vacation requests!", semantic=True) == ["leave", "submit"]
    assert tokenize("Vacation requests!") == ["vacation", "requests"]


def test_authorization_requires_tenant_and_every_non_public_label(
    chunks: list[DocumentChunk], employee_access: AccessContext
) -> None:
    assert is_authorized(chunks[0], employee_access)
    assert not is_authorized(chunks[1], employee_access)
    assert not is_authorized(chunks[2], employee_access)
    finance_manager = AccessContext(
        tenant_id="tenant-a", labels=frozenset({"public", "finance", "management"})
    )
    assert is_authorized(chunks[1], finance_manager)


def test_lexical_search_filters_before_ranking(
    repository: InMemoryChunkRepository, employee_access: AccessContext
) -> None:
    results = repository.lexical_search("annual leave", employee_access, limit=10)
    assert [result.chunk.chunk_id for result in results] == ["public-leave"]


def test_semantic_search_matches_alias_and_respects_limit(
    repository: InMemoryChunkRepository, employee_access: AccessContext
) -> None:
    results = repository.semantic_search("vacation", employee_access, limit=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "public-leave"
    assert results[0].score > 0


def test_search_with_no_tokens_or_matches_returns_empty(
    repository: InMemoryChunkRepository, employee_access: AccessContext
) -> None:
    assert repository.lexical_search("!!!", employee_access, limit=5) == []
    assert repository.semantic_search("!!!", employee_access, limit=5) == []
    assert repository.lexical_search("astronomy", employee_access, limit=5) == []


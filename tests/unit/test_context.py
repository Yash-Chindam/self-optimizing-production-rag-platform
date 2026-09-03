from rag_platform.context import ContextBuilder
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.repository import ScoredChunk

EMPLOYEE = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))


def chunk(
    identifier: str,
    text: str,
    *,
    labels: frozenset[str] | None = None,
    source_version_id: str = "sv-1",
    parent_chunk_id: str | None = None,
    tenant_id: str = "tenant-a",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id=tenant_id,
        access_labels=labels or frozenset({"public"}),
        text=text,
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="index-v1",
        source_version_id=source_version_id,
        parent_chunk_id=parent_chunk_id,
    )


def scored(*chunks: DocumentChunk) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=item, score=float(len(chunks) - index))
        for index, item in enumerate(chunks)
    ]


def test_citation_ids_are_assigned_in_rank_order() -> None:
    bundle = ContextBuilder(PipelineConfig()).build(
        scored(
            chunk("a", "Annual leave uses the HR portal.", source_version_id="sv-1"),
            chunk("b", "Expense claims are monthly.", source_version_id="sv-2"),
        ),
        EMPLOYEE,
    )
    assert [item.citation_id for item in bundle.items] == ["C1", "C2"]
    assert bundle.used_characters == sum(len(item.chunk.text) for item in bundle.items)


def test_authorization_is_rechecked_before_context_assembly() -> None:
    restricted = chunk("restricted", "Confidential forecast.", labels=frozenset({"finance"}))
    other_tenant = chunk("foreign", "Other tenant leave policy.", tenant_id="tenant-b")
    bundle = ContextBuilder(PipelineConfig()).build(
        scored(chunk("ok", "Annual leave uses the HR portal."), restricted, other_tenant),
        EMPLOYEE,
    )
    assert [item.chunk.chunk_id for item in bundle.items] == ["ok"]
    assert set(bundle.dropped_chunk_ids) == {"restricted", "foreign"}
    assert "authorization_recheck_removed:restricted" in bundle.policy_notes


def test_near_duplicate_chunks_are_removed() -> None:
    text = "Annual leave requests must be submitted in the HR portal."
    bundle = ContextBuilder(PipelineConfig()).build(
        scored(chunk("a", text), chunk("b", text + " "), chunk("c", "Expenses are monthly.")),
        EMPLOYEE,
    )
    assert [item.chunk.chunk_id for item in bundle.items] == ["a", "c"]
    assert "near_duplicate_removed:b" in bundle.policy_notes


def test_a_lower_threshold_keeps_only_clearly_distinct_evidence() -> None:
    config = PipelineConfig(near_duplicate_threshold=0.3)
    bundle = ContextBuilder(config).build(
        scored(
            chunk("a", "Annual leave requests use the HR portal."),
            chunk("b", "Annual leave requests need manager approval."),
        ),
        EMPLOYEE,
    )
    assert [item.chunk.chunk_id for item in bundle.items] == ["a"]


def test_evidence_diversity_caps_chunks_per_source() -> None:
    config = PipelineConfig(max_chunks_per_source=1)
    bundle = ContextBuilder(config).build(
        scored(
            chunk("a", "Leave requests use the HR portal.", source_version_id="sv-1"),
            chunk("b", "Managers approve within two days.", source_version_id="sv-1"),
            chunk("c", "Expense claims are monthly.", source_version_id="sv-2"),
        ),
        EMPLOYEE,
    )
    assert [item.chunk.chunk_id for item in bundle.items] == ["a", "c"]
    assert "evidence_diversity_removed:b" in bundle.policy_notes


def test_parent_context_is_expanded_when_it_fits() -> None:
    parent = chunk("parent", "Leave section. " * 5)
    child = chunk("child", "Leave section.", parent_chunk_id="parent")
    builder = ContextBuilder(PipelineConfig(), lambda _tenant, _id: parent)

    bundle = builder.build(scored(child), EMPLOYEE)
    assert bundle.items[0].chunk.chunk_id == "parent"
    assert bundle.items[0].expanded_to_parent is True
    assert "parent_context_expanded:parent" in bundle.policy_notes


def test_parent_context_is_left_alone_when_it_does_not_fit() -> None:
    parent = chunk("parent", "Leave section. " * 100)
    child = chunk("child", "Leave section.", parent_chunk_id="parent")
    builder = ContextBuilder(PipelineConfig(context_character_budget=200), lambda *_: parent)

    bundle = builder.build(scored(child), EMPLOYEE)
    assert bundle.items[0].chunk.chunk_id == "child"
    assert bundle.items[0].expanded_to_parent is False


def test_two_fragments_of_one_section_expand_to_a_single_parent() -> None:
    parent = chunk("parent", "Leave section covering requests and approvals.")
    builder = ContextBuilder(PipelineConfig(), lambda *_: parent)
    bundle = builder.build(
        scored(
            chunk("child-1", "Leave requests.", parent_chunk_id="parent"),
            chunk("child-2", "Approval rules.", parent_chunk_id="parent"),
        ),
        EMPLOYEE,
    )
    assert [item.chunk.chunk_id for item in bundle.items] == ["parent"]
    assert "parent_already_in_context:child-2" in bundle.policy_notes


def test_parent_expansion_can_be_disabled() -> None:
    parent = chunk("parent", "Leave section. " * 5)
    child = chunk("child", "Leave section.", parent_chunk_id="parent")
    builder = ContextBuilder(
        PipelineConfig(expand_parent_context=False), lambda *_: parent
    )
    assert builder.build(scored(child), EMPLOYEE).items[0].chunk.chunk_id == "child"


def test_an_unauthorized_or_missing_parent_is_never_substituted() -> None:
    restricted_parent = chunk("parent", "Secret section.", labels=frozenset({"finance"}))
    child = chunk("child", "Leave section.", parent_chunk_id="parent")

    restricted = ContextBuilder(PipelineConfig(), lambda *_: restricted_parent)
    assert restricted.build(scored(child), EMPLOYEE).items[0].chunk.chunk_id == "child"

    missing = ContextBuilder(PipelineConfig(), lambda *_: None)
    assert missing.build(scored(child), EMPLOYEE).items[0].chunk.chunk_id == "child"


def test_the_budget_drops_evidence_that_does_not_fit() -> None:
    config = PipelineConfig(context_character_budget=200, max_chunks_per_source=5)
    bundle = ContextBuilder(config).build(
        scored(
            chunk("a", "Leave requests use the HR portal.", source_version_id="sv-1"),
            chunk("b", "x" * 400, source_version_id="sv-2"),
            chunk("c", "Expense claims are monthly.", source_version_id="sv-3"),
        ),
        EMPLOYEE,
    )
    assert [item.chunk.chunk_id for item in bundle.items] == ["a", "c"]
    assert "context_budget_removed:b" in bundle.policy_notes


def test_an_empty_candidate_set_produces_an_empty_bundle() -> None:
    bundle = ContextBuilder(PipelineConfig()).build([], EMPLOYEE)
    assert bundle.items == ()
    assert bundle.used_characters == 0

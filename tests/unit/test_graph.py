from rag_platform.catalog import IndexCatalog
from rag_platform.graph import (
    CatalogGraphRetriever,
    GraphRetriever,
    KnowledgeGraph,
    extract_entities,
    extract_relationships,
    normalize_entity,
)
from rag_platform.ingestion import IngestionPipeline
from rag_platform.models import AccessContext, DocumentChunk, SourceRegistration

ACCESS = AccessContext(tenant_id="tenant-a", labels=frozenset({"public"}))


def chunk(identifier: str, text: str, *, labels: frozenset[str] | None = None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier,
        tenant_id="tenant-a",
        access_labels=labels or frozenset({"public"}),
        text=text,
        source_uri=f"https://example.test/{identifier}",
        source_title=identifier,
        index_version="index-v1",
    )


CORPUS = [
    chunk("payroll", "The Payroll Service depends on Identity Platform."),
    chunk("identity", "Identity Platform requires hardware keys for administrators."),
    chunk("security", "Identity Platform is owned by the Security Team."),
    chunk("catering", "Catering guidance for the Dublin Office."),
    chunk("restricted", "Identity Platform incident notes.", labels=frozenset({"security"})),
]


def test_sentence_initial_words_are_not_entities() -> None:
    entities = extract_entities("Annual leave requests use the HR portal.")
    assert "annual" not in entities
    assert "hr" in entities


def test_multi_word_names_and_acronyms_are_entities() -> None:
    entities = extract_entities("The Payroll Service reports to ACME Finance.")
    assert "payroll service" in entities
    assert "acme finance" in entities


def test_leading_articles_are_normalized_away() -> None:
    assert normalize_entity("  The Payroll Service. ") == "payroll service"
    assert normalize_entity("An Example") == "example"


def test_relationships_record_their_evidence_chunk() -> None:
    relationships = extract_relationships(CORPUS[0])
    assert len(relationships) == 1
    assert relationships[0].subject == "payroll service"
    assert relationships[0].relation == "depends_on"
    assert relationships[0].object == "identity platform"
    assert relationships[0].evidence_chunk_id == "payroll"


def test_graph_indexes_every_supported_relation_type() -> None:
    graph = KnowledgeGraph()
    graph.index(
        [
            chunk("a", "Alpha Team reports to Beta Group."),
            chunk("b", "Alpha Service is approved by Risk Board."),
            chunk("c", "Alpha Table is derived from Source Table."),
            chunk("d", "Alpha Process is governed by Data Policy."),
        ]
    )
    assert {relationship.relation for relationship in graph.relationships()} == {
        "reports_to",
        "approved_by",
        "derived_from",
        "governed_by",
    }


def test_traversal_follows_relationship_edges_only() -> None:
    graph = KnowledgeGraph()
    graph.index(CORPUS)
    one_hop = graph.related({"payroll service"}, 1)
    two_hops = graph.related({"payroll service"}, 2)

    assert "identity platform" in one_hop
    assert "security team" not in one_hop
    assert "security team" in two_hops
    assert "dublin office" not in two_hops


def test_traversal_stops_when_no_new_entities_are_reachable() -> None:
    graph = KnowledgeGraph()
    graph.index(CORPUS)
    assert graph.related({"payroll service"}, 4) == graph.related({"payroll service"}, 3)
    assert graph.related({"dublin office"}, 3) == set()


def test_expansion_excludes_seeds_and_unauthorized_chunks() -> None:
    graph = KnowledgeGraph()
    graph.index(CORPUS)
    retriever = GraphRetriever(graph, CORPUS)
    expanded = retriever.expand([CORPUS[0]], ACCESS, depth=1, limit=10)

    identifiers = [item.chunk_id for item in expanded]
    assert "payroll" not in identifiers
    assert "identity" in identifiers
    assert "restricted" not in identifiers


def test_expansion_respects_its_candidate_limit() -> None:
    graph = KnowledgeGraph()
    graph.index(CORPUS)
    retriever = GraphRetriever(graph, CORPUS)
    assert len(retriever.expand([CORPUS[0]], ACCESS, depth=2, limit=1)) == 1
    assert retriever.expand([CORPUS[0]], ACCESS, depth=2, limit=0) == []
    assert retriever.expand([], ACCESS, depth=2, limit=5) == []


def test_catalog_graph_follows_index_activation_and_rollback() -> None:
    catalog = IndexCatalog()
    pipeline = IngestionPipeline(catalog=catalog)
    registration = SourceRegistration(
        source_id="platform-map",
        tenant_id="tenant-a",
        owner="platform",
        source_uri="https://example.test/platform",
        source_title="Platform map",
    )
    pipeline.ingest(
        registration,
        "# Platform\n\n## Payroll\n\nThe Payroll Service depends on Identity Platform.\n",
    )
    pipeline.ingest(
        SourceRegistration(
            source_id="identity-notes",
            tenant_id="tenant-a",
            owner="platform",
            source_uri="https://example.test/identity",
            source_title="Identity notes",
        ),
        "# Identity\n\n## Keys\n\nIdentity Platform requires hardware keys.\n",
    )

    graph_retriever = CatalogGraphRetriever(catalog)
    seeds = [
        item for item in catalog.active_chunks("tenant-a") if "Payroll Service" in item.text
    ]
    expanded = graph_retriever.expand(seeds, ACCESS, depth=1, limit=5)
    assert any("hardware keys" in item.text for item in expanded)

    catalog.rollback("tenant-a")
    after_rollback = graph_retriever.expand(seeds, ACCESS, depth=1, limit=5)
    assert after_rollback == []


def test_catalog_graph_without_an_active_index_returns_nothing() -> None:
    retriever = CatalogGraphRetriever(IndexCatalog())
    assert retriever.expand(CORPUS, ACCESS, depth=1, limit=5) == []

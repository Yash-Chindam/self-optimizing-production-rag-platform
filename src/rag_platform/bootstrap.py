"""Local platform assembly.

The demo corpus is ingested through the real ingestion pipeline rather than injected as
pre-built chunks, so the local application exercises source versioning, chunking, sensitive-data
processing and staged index activation on every start.
"""

from dataclasses import dataclass

from rag_platform.catalog import IndexCatalog
from rag_platform.context import ContextBuilder
from rag_platform.evaluation import Answerer
from rag_platform.feedback import FeedbackLog
from rag_platform.graph import CatalogGraphRetriever
from rag_platform.ingestion import IngestionPipeline
from rag_platform.models import PipelineConfig, SourceRegistration
from rag_platform.optimization import OptimizationContext
from rag_platform.repository import CatalogChunkRepository
from rag_platform.rerank import LexicalCrossEncoder
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService
from rag_platform.workflow import WorkflowObserver

ACME_HANDBOOK = """# ACME Employee Handbook

## Annual leave

Annual leave requests must be submitted in the HR portal at least five working days before the
first day of leave. Managers approve or decline a request within two working days.

## Expenses

Expense claims are submitted monthly and are approved by the owner of the cost centre that
funds the claim.
"""

ACME_FORECAST = """# ACME Finance Forecast

## Revenue outlook

The confidential finance forecast projects twelve percent revenue growth for the next
financial year.
"""

GLOBEX_HANDBOOK = """# Globex Employee Handbook

## Annual leave

Globex employees submit annual leave requests by email to their line manager.
"""

DEMO_SOURCES: tuple[tuple[SourceRegistration, str], ...] = (
    (
        SourceRegistration(
            source_id="acme-handbook",
            tenant_id="tenant-acme",
            owner="people-operations",
            access_labels=frozenset({"public"}),
            source_uri="https://knowledge.example/acme/handbook",
            source_title="ACME Employee Handbook",
        ),
        ACME_HANDBOOK,
    ),
    (
        SourceRegistration(
            source_id="acme-forecast",
            tenant_id="tenant-acme",
            owner="finance",
            access_labels=frozenset({"finance"}),
            source_uri="https://knowledge.example/acme/finance/forecast",
            source_title="ACME Finance Forecast",
        ),
        ACME_FORECAST,
    ),
    (
        SourceRegistration(
            source_id="globex-handbook",
            tenant_id="tenant-globex",
            owner="people-operations",
            access_labels=frozenset({"public"}),
            source_uri="https://knowledge.example/globex/handbook",
            source_title="Globex Employee Handbook",
        ),
        GLOBEX_HANDBOOK,
    ),
)


@dataclass(frozen=True, slots=True)
class Platform:
    config: PipelineConfig
    catalog: IndexCatalog
    ingestion: IngestionPipeline
    query_service: Answerer
    """A `QueryService`, or one wrapped by the answer cache."""
    feedback: FeedbackLog
    optimization: OptimizationContext
    """What the offline evaluation and optimization commands assemble candidate services from."""


def build_platform(
    *,
    seed_demo_sources: bool = True,
    config: PipelineConfig | None = None,
    observer: WorkflowObserver | None = None,
) -> Platform:
    config = config or PipelineConfig()
    catalog = IndexCatalog()
    ingestion = IngestionPipeline(catalog=catalog)
    if seed_demo_sources:
        for registration, content in DEMO_SOURCES:
            ingestion.ingest(registration, content)
    repository = CatalogChunkRepository(catalog)
    graph_retriever = CatalogGraphRetriever(catalog, config.graph_extractor_revision)
    reranker = LexicalCrossEncoder(revision=config.reranker_revision)
    retriever = HybridRetriever(
        repository, config, graph_retriever=graph_retriever, reranker=reranker
    )
    query_service = QueryService(
        retriever, config, ContextBuilder(config, catalog.chunk), observer=observer
    )
    return Platform(
        config=config,
        catalog=catalog,
        ingestion=ingestion,
        query_service=query_service,
        feedback=FeedbackLog(),
        optimization=OptimizationContext(
            repository=repository,
            programs=query_service.programs,
            graph_retriever=graph_retriever,
            reranker=reranker,
            parent_lookup=catalog.chunk,
        ),
    )


def build_query_service() -> Answerer:
    return build_platform().query_service

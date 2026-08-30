"""Local platform assembly.

The demo corpus is ingested through the real ingestion pipeline rather than injected as
pre-built chunks, so the local application exercises source versioning, chunking, sensitive-data
processing and staged index activation on every start.
"""

from dataclasses import dataclass

from rag_platform.catalog import IndexCatalog
from rag_platform.ingestion import IngestionPipeline
from rag_platform.models import PipelineConfig, SourceRegistration
from rag_platform.repository import CatalogChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService

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
    query_service: QueryService


def build_platform(*, seed_demo_sources: bool = True) -> Platform:
    config = PipelineConfig()
    catalog = IndexCatalog()
    ingestion = IngestionPipeline(catalog=catalog)
    if seed_demo_sources:
        for registration, content in DEMO_SOURCES:
            ingestion.ingest(registration, content)
    repository = CatalogChunkRepository(catalog)
    query_service = QueryService(HybridRetriever(repository, config), config)
    return Platform(
        config=config, catalog=catalog, ingestion=ingestion, query_service=query_service
    )


def build_query_service() -> QueryService:
    return build_platform().query_service

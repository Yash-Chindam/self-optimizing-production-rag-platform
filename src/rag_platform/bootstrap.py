from rag_platform.models import DocumentChunk, PipelineConfig
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService


def build_query_service() -> QueryService:
    config = PipelineConfig()
    chunks = [
        DocumentChunk(
            chunk_id="acme-handbook-leave",
            tenant_id="tenant-acme",
            access_labels=frozenset({"public"}),
            text=(
                "Annual leave requests must be submitted in the HR portal at least five "
                "working days before the first day of leave."
            ),
            source_uri="https://knowledge.example/acme/handbook#annual-leave",
            source_title="ACME Employee Handbook",
            index_version="demo-index-v1",
        ),
        DocumentChunk(
            chunk_id="acme-finance-forecast",
            tenant_id="tenant-acme",
            access_labels=frozenset({"finance"}),
            text="The confidential finance forecast projects twelve percent revenue growth.",
            source_uri="https://knowledge.example/acme/finance/forecast",
            source_title="ACME Finance Forecast",
            index_version="demo-index-v1",
        ),
        DocumentChunk(
            chunk_id="globex-handbook-leave",
            tenant_id="tenant-globex",
            access_labels=frozenset({"public"}),
            text="Globex employees submit annual leave requests by email to their manager.",
            source_uri="https://knowledge.example/globex/handbook#annual-leave",
            source_title="Globex Employee Handbook",
            index_version="demo-index-v1",
        ),
    ]
    repository = InMemoryChunkRepository(chunks)
    return QueryService(HybridRetriever(repository, config), config)


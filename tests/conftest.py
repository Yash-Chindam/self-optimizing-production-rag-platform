import pytest

from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig
from rag_platform.repository import InMemoryChunkRepository


@pytest.fixture
def chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="public-leave",
            tenant_id="tenant-a",
            access_labels=frozenset({"public"}),
            text="Annual leave requests use the HR portal.",
            source_uri="https://example.test/leave",
            source_title="Leave policy",
            index_version="index-v1",
        ),
        DocumentChunk(
            chunk_id="finance-plan",
            tenant_id="tenant-a",
            access_labels=frozenset({"finance", "management"}),
            text="The confidential revenue plan targets twenty percent growth.",
            source_uri="https://example.test/finance",
            source_title="Finance plan",
            index_version="index-v1",
        ),
        DocumentChunk(
            chunk_id="other-tenant-leave",
            tenant_id="tenant-b",
            access_labels=frozenset({"public"}),
            text="Annual leave is approved by a department director.",
            source_uri="https://example.test/other-leave",
            source_title="Other tenant policy",
            index_version="index-v2",
        ),
    ]


@pytest.fixture
def repository(chunks: list[DocumentChunk]) -> InMemoryChunkRepository:
    return InMemoryChunkRepository(chunks)


@pytest.fixture
def employee_access() -> AccessContext:
    return AccessContext(tenant_id="tenant-a", labels=frozenset({"public", "employees"}))


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig(dense_top_k=5, sparse_top_k=5, final_top_k=5)


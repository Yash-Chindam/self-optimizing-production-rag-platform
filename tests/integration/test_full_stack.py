"""The platform assembled from the environment, running on every real service at once.

Skipped unless the whole stack from docker-compose.yml is configured (deploy/local.env). This is
the end-to-end proof that the running application, not just each adapter on its own, ingests
into and answers from PostgreSQL, Qdrant, OpenSearch, Neo4j, MinIO, Redis, Kafka and the
collector, and that a restarted process picks the same index back up.
"""

import dataclasses
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from rag_platform.api import create_app
from rag_platform.models import AccessContext
from rag_platform.runtime import build_runtime
from rag_platform.settings import Settings

REQUIRED = (
    "RAG_QDRANT_URL",
    "RAG_OPENSEARCH_URL",
    "RAG_NEO4J_URL",
    "RAG_POSTGRES_DSN",
    "RAG_MINIO_ENDPOINT",
    "RAG_REDIS_URL",
    "RAG_KAFKA_BOOTSTRAP_SERVERS",
    "RAG_OTLP_ENDPOINT",
)
CONFIGURED = all(os.environ.get(name) for name in REQUIRED)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not CONFIGURED, reason="the full service stack is not configured"),
]

ACME = AccessContext(tenant_id="tenant-acme", labels=frozenset({"public"}))
FINANCE = AccessContext(tenant_id="tenant-acme", labels=frozenset({"public", "finance"}))
GLOBEX = AccessContext(tenant_id="tenant-globex", labels=frozenset({"public"}))
LEAVE = "How do I request annual leave?"
FORECAST = "What revenue growth does the finance forecast project?"


def settings(*, seed: bool) -> Settings:
    base = Settings.from_env(os.environ)
    return dataclasses.replace(
        base, seed_demo_sources=seed, kafka_topic_prefix=f"rag-ci-{uuid.uuid4().hex[:8]}."
    )


def test_the_assembled_platform_ingests_into_and_answers_from_the_real_stores() -> None:
    runtime = build_runtime(settings(seed=True))
    try:
        platform = runtime.platform
        active = platform.catalog.active_index_version("tenant-acme")
        assert active is not None
        assert type(platform.catalog).__name__ == "PostgresIndexCatalog"
        assert len(platform.ingestion.mirrors) == 3

        answered = platform.query_service.answer(LEAVE, ACME)
        assert answered.status == "answered"
        assert "HR portal" in answered.answer
        assert answered.trace.index_version == active.index_version_id

        # Authorization holds on the real stores: same question, different labels.
        assert platform.query_service.answer(FORECAST, ACME).status == "insufficient_evidence"
        permitted = platform.query_service.answer(FORECAST, FINANCE)
        assert permitted.status == "answered"
        assert "twelve percent" in permitted.answer

        # Tenant isolation holds on the real stores.
        globex = platform.query_service.answer(LEAVE, GLOBEX)
        assert "line manager" in globex.answer
        assert "HR portal" not in globex.answer

        assert platform.query_service.answer(LEAVE, ACME).trace.served_from_cache is True
    finally:
        runtime.close()


def test_a_restarted_process_serves_the_same_index_without_reingesting() -> None:
    first = build_runtime(settings(seed=True))
    active = first.platform.catalog.active_index_version("tenant-acme")
    first.close()
    assert active is not None

    restarted = build_runtime(settings(seed=False))
    try:
        platform = restarted.platform
        assert platform.catalog.active_index_version("tenant-acme") == active
        response = platform.query_service.answer(FORECAST, FINANCE)
        assert response.status == "answered"
        assert response.trace.index_version == active.index_version_id
    finally:
        restarted.close()


def test_the_api_starts_from_the_environment_and_reports_its_dependencies() -> None:
    build_runtime(settings(seed=True)).close()

    with TestClient(create_app()) as client:
        ready = client.get("/readyz")
        answer = client.post(
            "/v1/query", headers={"X-Tenant-ID": "tenant-acme"}, json={"question": LEAVE}
        )

    assert ready.status_code == 200
    assert {"qdrant", "opensearch", "neo4j", "postgres", "minio", "redis", "kafka", "otlp"} <= set(
        ready.json()["dependencies"]
    )
    assert answer.json()["status"] == "answered"


def test_replicas_stay_in_step_through_postgres() -> None:
    from rag_platform.feedback import FeedbackRequest, FeedbackReview
    from rag_platform.models import SourceRegistration

    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    access = AccessContext(tenant_id=tenant, labels=frozenset({"public"}))
    base = dataclasses.replace(settings(seed=False), catalog_sync_seconds=0)
    first, second = build_runtime(base), build_runtime(base)
    try:
        assert second.platform.query_service.answer(LEAVE, access).status == "insufficient_evidence"

        # Ingest through one replica; the other must serve it without restarting.
        first.platform.ingestion.ingest(
            SourceRegistration(
                source_id="handbook",
                tenant_id=tenant,
                owner="people-operations",
                source_uri="https://example.test/handbook",
                source_title="Handbook",
            ),
            "# Handbook\n\n## Annual leave\n\nAnnual leave requests use the staff portal.\n",
        )
        question = "Where are annual leave requests made?"
        answer = second.platform.query_service.answer(question, access)
        assert answer.status == "answered"
        assert "staff portal" in answer.answer

        # Feedback submitted to one replica is reviewed on the other.
        submitted = first.platform.feedback.submit(
            tenant, access.labels, FeedbackRequest(question=question, rating="helpful")
        )
        reviewed = second.platform.feedback.review(
            tenant, submitted.feedback_id, FeedbackReview(reviewer="r.osei", decision="accepted")
        )
        assert first.platform.feedback.get(tenant, submitted.feedback_id) == reviewed
    finally:
        first.close()
        second.close()

import pytest
from fastapi.testclient import TestClient

from rag_platform.api import create_app


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.mark.integration
def test_health_exposes_active_config_version(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "config_version": "pipeline-v1"}


@pytest.mark.integration
def test_index_serves_browser_client(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Answers that show their evidence" in response.text


@pytest.mark.integration
def test_query_answers_with_authorized_evidence(client: TestClient) -> None:
    response = client.post(
        "/v1/query",
        headers={"X-Tenant-ID": "tenant-acme", "X-Access-Labels": "employees"},
        json={"question": "How do I request vacation?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["citations"][0]["chunk_id"] == "acme-handbook-leave"
    assert "five working days" in payload["answer"]


@pytest.mark.integration
def test_query_does_not_return_other_tenant_or_restricted_evidence(client: TestClient) -> None:
    response = client.post(
        "/v1/query",
        headers={"X-Tenant-ID": "tenant-acme", "X-Access-Labels": "employees"},
        json={"question": "What does the confidential finance forecast say?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "insufficient_evidence"
    assert payload["trace"]["retrieved_chunk_ids"] == []


@pytest.mark.integration
def test_query_requires_tenant_header(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "annual leave"})
    assert response.status_code == 422


@pytest.mark.integration
def test_query_rejects_blank_tenant_header(client: TestClient) -> None:
    response = client.post(
        "/v1/query", headers={"X-Tenant-ID": "   "}, json={"question": "annual leave"}
    )
    assert response.status_code == 400


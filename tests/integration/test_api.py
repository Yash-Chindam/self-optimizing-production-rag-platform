from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_platform.api import create_app

STEWARD = {"X-Tenant-ID": "tenant-acme", "X-Access-Labels": "employees,data-steward"}
EMPLOYEE = {"X-Tenant-ID": "tenant-acme", "X-Access-Labels": "employees"}

NEW_SOURCE = {
    "source_id": "acme-security",
    "owner": "security",
    "source_uri": "https://knowledge.example/acme/security",
    "source_title": "ACME Security Standard",
    "content": (
        "# ACME Security Standard\n\n## Laptop encryption\n\n"
        "Every ACME laptop must use full disk encryption before it leaves the office.\n"
    ),
}


@pytest.fixture
def client() -> Iterator[TestClient]:
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
        "/v1/query", headers=EMPLOYEE, json={"question": "How do I request vacation?"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["citations"][0]["source_title"] == "ACME Employee Handbook"
    assert "five working days" in payload["answer"]
    assert payload["trace"]["index_version"].startswith("iv-tenant-acme-")


@pytest.mark.integration
def test_query_does_not_return_other_tenant_or_restricted_evidence(client: TestClient) -> None:
    response = client.post(
        "/v1/query",
        headers=EMPLOYEE,
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


@pytest.mark.integration
def test_ingested_source_becomes_answerable_under_a_new_index_version(
    client: TestClient,
) -> None:
    before = client.get("/v1/index-versions/active", headers=EMPLOYEE).json()

    created = client.post("/v1/sources", headers=STEWARD, json=NEW_SOURCE)
    assert created.status_code == 201
    report = created.json()["report"]
    assert report["reused_existing_source_version"] is False
    assert "tenant_isolation_verified" in report["validation_checks"]

    after = client.get("/v1/index-versions/active", headers=EMPLOYEE).json()
    assert after["index_version_id"] != before["index_version_id"]
    assert len(after["source_version_ids"]) == len(before["source_version_ids"]) + 1

    answer = client.post(
        "/v1/query", headers=EMPLOYEE, json={"question": "Is laptop disk encryption required?"}
    ).json()
    assert answer["status"] == "answered"
    assert "full disk encryption" in answer["answer"]


@pytest.mark.integration
def test_ingestion_requires_the_data_steward_label(client: TestClient) -> None:
    response = client.post("/v1/sources", headers=EMPLOYEE, json=NEW_SOURCE)
    assert response.status_code == 403
    assert "data-steward" in response.json()["detail"]


@pytest.mark.integration
def test_rollback_restores_the_previous_index_version(client: TestClient) -> None:
    before = client.get("/v1/index-versions/active", headers=EMPLOYEE).json()
    client.post("/v1/sources", headers=STEWARD, json=NEW_SOURCE)

    restored = client.post("/v1/index-versions/rollback", headers=STEWARD)
    assert restored.status_code == 200
    assert restored.json()["index_version_id"] == before["index_version_id"]

    answer = client.post(
        "/v1/query", headers=EMPLOYEE, json={"question": "Is laptop disk encryption required?"}
    ).json()
    assert answer["status"] == "insufficient_evidence"


@pytest.mark.integration
def test_rollback_without_history_is_a_conflict(client: TestClient) -> None:
    response = client.post(
        "/v1/index-versions/rollback",
        headers={"X-Tenant-ID": "tenant-empty", "X-Access-Labels": "data-steward"},
    )
    assert response.status_code == 409


@pytest.mark.integration
def test_unknown_tenant_has_no_active_index_version(client: TestClient) -> None:
    response = client.get("/v1/index-versions/active", headers={"X-Tenant-ID": "tenant-none"})
    assert response.status_code == 404


@pytest.mark.integration
def test_invalid_source_content_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/sources", headers=STEWARD, json={**NEW_SOURCE, "content": "   \n\n   "}
    )
    assert response.status_code == 422


@pytest.mark.integration
def test_answer_cites_only_the_evidence_it_quoted(client: TestClient) -> None:
    payload = client.post(
        "/v1/query", headers=EMPLOYEE, json={"question": "How do I request annual leave?"}
    ).json()

    assert payload["status"] == "answered"
    assert payload["citations"]
    cited_chunks = {citation["chunk_id"] for citation in payload["citations"]}
    assert cited_chunks.issubset(set(payload["trace"]["context_chunk_ids"]))
    assert payload["trace"]["intent"] == "procedural"
    assert payload["trace"]["workflow_path"][0] == "classify"
    assert payload["trace"]["repair_attempts"] == 0
    assert payload["trace"]["unsupported_claims"] == []


@pytest.mark.integration
def test_an_ambiguous_question_returns_a_clarification(client: TestClient) -> None:
    payload = client.post("/v1/query", headers=EMPLOYEE, json={"question": "leave"}).json()

    assert payload["status"] == "clarification_needed"
    assert payload["citations"] == []
    assert payload["trace"]["workflow_path"] == ["classify", "clarify"]


FEEDBACK = {
    "question": "How do I request annual leave?",
    "rating": "unhelpful",
    "comment": "Write to jane.doe@example.com, the answer missed the notice period.",
}


@pytest.mark.integration
def test_a_query_reports_its_latency_model_route_and_token_estimates(client: TestClient) -> None:
    response = client.post(
        "/v1/query", headers=EMPLOYEE, json={"question": "How do I request annual leave?"}
    )
    trace = response.json()["trace"]
    assert trace["latency_ms"] > 0
    assert trace["model_route"] == "synthesizer-extractive-v1"
    assert trace["prompt_tokens"] > 0 and trace["completion_tokens"] > 0
    assert trace["trace_id"] is None


@pytest.mark.integration
def test_any_caller_can_submit_feedback_and_identifiers_are_redacted(client: TestClient) -> None:
    response = client.post("/v1/feedback", headers=EMPLOYEE, json=FEEDBACK)

    assert response.status_code == 201
    body = response.json()
    assert body["tenant_id"] == "tenant-acme"
    assert body["status"] == "submitted"
    assert "jane.doe@example.com" not in body["comment"]


@pytest.mark.integration
def test_only_a_steward_can_list_or_review_feedback(client: TestClient) -> None:
    feedback_id = client.post("/v1/feedback", headers=EMPLOYEE, json=FEEDBACK).json()[
        "feedback_id"
    ]
    review = {"reviewer": "r.osei", "decision": "accepted", "failure_category": "retrieval"}

    assert client.get("/v1/feedback", headers=EMPLOYEE).status_code == 403
    assert (
        client.post(f"/v1/feedback/{feedback_id}/review", headers=EMPLOYEE, json=review).status_code
        == 403
    )


@pytest.mark.integration
def test_a_steward_reviews_feedback_once(client: TestClient) -> None:
    feedback_id = client.post("/v1/feedback", headers=EMPLOYEE, json=FEEDBACK).json()[
        "feedback_id"
    ]
    review = {"reviewer": "r.osei", "decision": "accepted", "failure_category": "retrieval"}

    pending = client.get("/v1/feedback", headers=STEWARD, params={"status": "submitted"})
    assert [item["feedback_id"] for item in pending.json()] == [feedback_id]

    reviewed = client.post(f"/v1/feedback/{feedback_id}/review", headers=STEWARD, json=review)
    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "accepted"
    assert reviewed.json()["failure_category"] == "retrieval"

    again = client.post(f"/v1/feedback/{feedback_id}/review", headers=STEWARD, json=review)
    assert again.status_code == 409
    assert client.get("/v1/feedback", headers=STEWARD, params={"status": "submitted"}).json() == []


@pytest.mark.integration
def test_a_steward_of_another_tenant_cannot_review_feedback(client: TestClient) -> None:
    feedback_id = client.post("/v1/feedback", headers=EMPLOYEE, json=FEEDBACK).json()[
        "feedback_id"
    ]
    other = {"X-Tenant-ID": "tenant-globex", "X-Access-Labels": "data-steward"}
    review = {"reviewer": "m.lind", "decision": "dismissed"}

    response = client.post(f"/v1/feedback/{feedback_id}/review", headers=other, json=review)
    assert response.status_code == 404
    assert client.get("/v1/feedback", headers=other).json() == []

"""Event delivery through a real Kafka broker (docker-compose.yml).

Skipped unless RAG_KAFKA_BOOTSTRAP_SERVERS is set.
"""

import os
import uuid

import pytest

from rag_platform.catalog import IndexCatalog
from rag_platform.events import PlatformEvent
from rag_platform.ingestion import IngestionPipeline
from rag_platform.models import SourceRegistration

BOOTSTRAP = os.environ.get("RAG_KAFKA_BOOTSTRAP_SERVERS")

HANDBOOK = "# Handbook\n\n## Annual leave\n\nAnnual leave requests use the HR portal.\n"


def consume(prefix: str, topics: list[str], expected: int) -> list[PlatformEvent]:
    from confluent_kafka import Consumer

    from rag_platform.adapters.kafka_events import decode

    assert BOOTSTRAP is not None
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"rag-ci-{uuid.uuid4().hex[:8]}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([f"{prefix}{topic}" for topic in topics])
    events: list[PlatformEvent] = []
    try:
        for _attempt in range(60):
            message = consumer.poll(1.0)
            if message is None or message.error():
                continue
            events.append(decode(message.value()))
            if len(events) >= expected:
                break
    finally:
        consumer.close()
    return events


@pytest.mark.integration
@pytest.mark.skipif(not BOOTSTRAP, reason="RAG_KAFKA_BOOTSTRAP_SERVERS not set")
def test_ingestion_events_are_delivered_through_kafka() -> None:
    from rag_platform.adapters.kafka_events import KafkaEventPublisher, producer_for

    assert BOOTSTRAP is not None
    # A prefix per run keeps reruns against a long-lived broker independent.
    prefix = f"rag-ci-{uuid.uuid4().hex[:8]}."
    publisher = KafkaEventPublisher(producer_for(BOOTSTRAP), prefix=prefix)
    tenant = f"tenant-{uuid.uuid4().hex[:8]}"
    registration = SourceRegistration(
        source_id="handbook",
        tenant_id=tenant,
        owner="people-operations",
        source_uri="https://example.test/handbook",
        source_title="Handbook",
    )

    result = IngestionPipeline(catalog=IndexCatalog(), events=publisher).ingest(
        registration, HANDBOOK
    )
    assert "event_published:source.changed" in result.report.validation_checks
    assert "event_published:ingestion.completed" in result.report.validation_checks

    events = consume(prefix, ["source-changed", "ingestion-completed"], expected=2)
    by_type = {event.event_type: event for event in events}
    assert by_type["source.changed"].tenant_id == tenant
    assert by_type["source.changed"].subject == "handbook"
    assert by_type["ingestion.completed"].subject == result.index_version.index_version_id
    assert "HR portal" not in " ".join(event.model_dump_json() for event in events)


@pytest.mark.integration
@pytest.mark.skipif(not BOOTSTRAP, reason="RAG_KAFKA_BOOTSTRAP_SERVERS not set")
def test_an_unreachable_broker_degrades_ingestion_instead_of_failing_it() -> None:
    from rag_platform.adapters.kafka_events import KafkaEventPublisher, producer_for

    publisher = KafkaEventPublisher(producer_for("127.0.0.1:1"), flush_timeout_seconds=2.0)
    catalog = IndexCatalog()
    registration = SourceRegistration(
        source_id="handbook",
        tenant_id="tenant-a",
        owner="people-operations",
        source_uri="https://example.test/handbook",
        source_title="Handbook",
    )

    result = IngestionPipeline(catalog=catalog, events=publisher).ingest(registration, HANDBOOK)

    assert catalog.active_index_version("tenant-a") == result.index_version
    assert any(
        check.startswith("event_publish_failed:") for check in result.report.validation_checks
    )

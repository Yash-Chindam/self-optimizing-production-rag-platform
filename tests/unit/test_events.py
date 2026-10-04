"""Platform events, the Kafka transport (against a fake producer) and ingestion's use of both."""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_platform.adapters.kafka_events import (
    EventDeliveryError,
    KafkaEventPublisher,
    decode,
    encode,
    topic_for,
)
from rag_platform.catalog import IndexCatalog
from rag_platform.events import (
    InMemoryEventPublisher,
    PlatformEvent,
    evaluation_completed,
    ingestion_completed,
    publish_safely,
    source_changed,
)
from rag_platform.ingestion import IngestionPipeline
from rag_platform.models import SourceRegistration

HANDBOOK = (
    "# Handbook\n\n## Annual leave\n\n"
    "Annual leave requests use the HR portal. Contact jane.doe@example.com for help.\n"
)
REGISTRATION = SourceRegistration(
    source_id="handbook",
    tenant_id="tenant-a",
    owner="people-operations",
    source_uri="https://example.test/handbook",
    source_title="Handbook",
)


class BrokenPublisher:
    def publish(self, event: PlatformEvent) -> None:
        raise ConnectionError("broker unreachable")


# Core --------------------------------------------------------------------------


def test_without_a_publisher_nothing_is_published_or_recorded() -> None:
    assert publish_safely(None, source_changed("tenant-a", "handbook", "sv-1", "abc")) is None


def test_a_published_event_is_acknowledged_by_type() -> None:
    publisher = InMemoryEventPublisher()
    note = publish_safely(publisher, ingestion_completed("tenant-a", "iv-1", "sv-1", 3))

    assert note == "event_published:ingestion.completed"
    assert publisher.events[0].subject == "iv-1"
    assert publisher.events[0].attributes == {"source_version_id": "sv-1", "chunk_count": "3"}


def test_a_broker_failure_is_reported_not_raised() -> None:
    note = publish_safely(BrokenPublisher(), source_changed("tenant-a", "handbook", "sv-1", "abc"))
    assert note == "event_publish_failed:source.changed:ConnectionError"


def test_an_evaluation_event_summarizes_the_run() -> None:
    event = evaluation_completed(
        "59deebbdcfb2", "pipeline-v1", passed=9, total=10, authorization_violations=0
    )
    assert event.event_type == "evaluation.completed"
    assert event.tenant_id == "platform"
    assert event.attributes["passed"] == "9"


# Ingestion ---------------------------------------------------------------------


def test_ingestion_announces_the_source_version_and_the_activated_index() -> None:
    publisher = InMemoryEventPublisher()
    result = IngestionPipeline(catalog=IndexCatalog(), events=publisher).ingest(
        REGISTRATION, HANDBOOK
    )

    changed, completed = publisher.events
    assert changed.event_type == "source.changed"
    assert changed.subject == "handbook"
    assert changed.attributes["source_version_id"] == result.source_version.source_version_id
    assert completed.event_type == "ingestion.completed"
    assert completed.subject == result.index_version.index_version_id
    assert "event_published:ingestion.completed" in result.report.validation_checks


def test_events_never_carry_document_text() -> None:
    publisher = InMemoryEventPublisher()
    IngestionPipeline(catalog=IndexCatalog(), events=publisher).ingest(REGISTRATION, HANDBOOK)

    serialized = " ".join(event.model_dump_json() for event in publisher.events)
    assert "HR portal" not in serialized
    assert "jane.doe@example.com" not in serialized


def test_a_reingest_of_identical_content_announces_nothing() -> None:
    publisher = InMemoryEventPublisher()
    pipeline = IngestionPipeline(catalog=IndexCatalog(), events=publisher)
    pipeline.ingest(REGISTRATION, HANDBOOK)
    pipeline.ingest(REGISTRATION, HANDBOOK)
    assert len(publisher.events) == 2


def test_a_broker_failure_does_not_fail_or_undo_the_ingest() -> None:
    catalog = IndexCatalog()
    result = IngestionPipeline(catalog=catalog, events=BrokenPublisher()).ingest(
        REGISTRATION, HANDBOOK
    )

    assert catalog.active_index_version("tenant-a") == result.index_version
    assert (
        "event_publish_failed:ingestion.completed:ConnectionError"
        in result.report.validation_checks
    )


def test_ingestion_without_a_publisher_records_no_event_checks() -> None:
    result = IngestionPipeline(catalog=IndexCatalog()).ingest(REGISTRATION, HANDBOOK)
    assert not any(check.startswith("event_") for check in result.report.validation_checks)


# Kafka -------------------------------------------------------------------------


@dataclass
class FakeProducer:
    delivery_error: str | None = None
    undelivered: int = 0
    produced: list[dict[str, Any]] = field(default_factory=list)
    flushed: list[float] = field(default_factory=list)

    def produce(self, topic: str, *, key: bytes, value: bytes, on_delivery: Any) -> None:
        self.produced.append({"topic": topic, "key": key, "value": value})
        self._callback = on_delivery

    def flush(self, timeout: float) -> int:
        self.flushed.append(timeout)
        self._callback(self.delivery_error, None)
        return self.undelivered


def kafka(**producer_options: Any) -> tuple[KafkaEventPublisher, FakeProducer]:
    producer = FakeProducer(**producer_options)
    return KafkaEventPublisher(producer=producer), producer  # type: ignore[arg-type]


def test_each_event_type_has_its_own_topic() -> None:
    assert topic_for("source.changed") == "rag.source-changed"
    assert topic_for("ingestion.completed") == "rag.ingestion-completed"
    assert topic_for("evaluation.completed", prefix="prod.rag.") == "prod.rag.evaluation-completed"


def test_an_event_round_trips_through_the_wire_format() -> None:
    event = ingestion_completed("tenant-a", "iv-1", "sv-1", 3)
    assert decode(encode(event)) == event
    assert json.loads(encode(event))["event_type"] == "ingestion.completed"


def test_a_published_event_is_keyed_by_tenant_and_flushed() -> None:
    publisher, producer = kafka()
    event = ingestion_completed("tenant-a", "iv-1", "sv-1", 3)
    publisher.publish(event)

    [message] = producer.produced
    assert message["topic"] == "rag.ingestion-completed"
    assert message["key"] == b"tenant-a"
    assert decode(message["value"]) == event
    assert producer.flushed == [10.0]


def test_a_rejected_delivery_is_an_error() -> None:
    publisher, _producer = kafka(delivery_error="topic authorization failed")
    with pytest.raises(EventDeliveryError, match="topic authorization failed"):
        publisher.publish(source_changed("tenant-a", "handbook", "sv-1", "abc"))


def test_an_event_still_queued_after_the_flush_timeout_is_an_error() -> None:
    publisher, _producer = kafka(undelivered=1)
    with pytest.raises(EventDeliveryError, match="not delivered"):
        publisher.publish(source_changed("tenant-a", "handbook", "sv-1", "abc"))


def test_a_kafka_failure_during_ingestion_is_recorded_not_raised() -> None:
    publisher, _producer = kafka(delivery_error="broker down")
    result = IngestionPipeline(catalog=IndexCatalog(), events=publisher).ingest(
        REGISTRATION, HANDBOOK
    )
    assert (
        "event_publish_failed:source.changed:EventDeliveryError" in result.report.validation_checks
    )


def test_the_producer_is_configured_for_idempotent_acknowledged_delivery() -> None:
    pytest.importorskip("confluent_kafka")
    from rag_platform.adapters.kafka_events import producer_for

    producer = producer_for("127.0.0.1:1")
    assert producer.flush(0) == 0

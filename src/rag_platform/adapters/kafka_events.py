"""Kafka transport for platform events (specification sections 6 and 18, optional).

One topic per event type, keyed by tenant so one tenant's events stay ordered on a partition.
The producer is flushed on every publish: these are low-volume control-plane events, and an
event that was accepted but never delivered would be worse than a slow publish.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_platform.events import PlatformEvent

if TYPE_CHECKING:
    from confluent_kafka import Producer

TOPICS: dict[str, str] = {
    "source.changed": "source-changed",
    "ingestion.completed": "ingestion-completed",
    "evaluation.completed": "evaluation-completed",
}


class EventDeliveryError(RuntimeError):
    pass


def topic_for(event_type: str, prefix: str = "rag.") -> str:
    return f"{prefix}{TOPICS[event_type]}"


def encode(event: PlatformEvent) -> bytes:
    return json.dumps(event.model_dump(mode="json"), sort_keys=True).encode("utf-8")


def decode(value: bytes) -> PlatformEvent:
    return PlatformEvent.model_validate_json(value)


@dataclass(slots=True)
class KafkaEventPublisher:
    producer: "Producer"
    prefix: str = "rag."
    flush_timeout_seconds: float = 10.0

    def publish(self, event: PlatformEvent) -> None:
        failures: list[str] = []

        def delivered(error: Any, _message: Any) -> None:
            if error is not None:
                failures.append(str(error))

        self.producer.produce(
            topic_for(event.event_type, self.prefix),
            key=event.tenant_id.encode("utf-8"),
            value=encode(event),
            on_delivery=delivered,
        )
        undelivered = self.producer.flush(self.flush_timeout_seconds)
        if failures:
            raise EventDeliveryError(failures[0])
        if undelivered:
            raise EventDeliveryError(f"{undelivered} event(s) not delivered before the timeout")


def producer_for(bootstrap_servers: str) -> "Producer":
    from confluent_kafka import Producer

    return Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,
            "acks": "all",
        }
    )

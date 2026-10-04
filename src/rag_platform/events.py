"""Platform events (specification section 6: source-change, ingestion and evaluation events).

Events are notifications, not the system of record: the catalog, the registry and the evaluation
reports stay authoritative. A publisher is therefore always best effort. A broker that is down
must never fail an ingest or an evaluation, so `publish_safely` reports the failure to its
caller instead of raising, and the caller records it.

Events carry identifiers and counts only. No chunk text, no question and no answer is ever put
on a topic, because a topic is readable by consumers that hold no tenant's access labels.
"""

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["source.changed", "ingestion.completed", "evaluation.completed"]


class PlatformEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_type: EventType
    tenant_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    """What the event is about: a source id, an index version id or a dataset revision."""
    attributes: dict[str, str] = Field(default_factory=dict)


class EventPublisher(Protocol):
    def publish(self, event: PlatformEvent) -> None: ...


@dataclass(slots=True)
class InMemoryEventPublisher:
    events: list[PlatformEvent] = field(default_factory=list)

    def publish(self, event: PlatformEvent) -> None:
        self.events.append(event)


def publish_safely(publisher: EventPublisher | None, event: PlatformEvent) -> str | None:
    """Publish if there is a publisher. Returns a failure note instead of raising."""
    if publisher is None:
        return None
    try:
        publisher.publish(event)
    except Exception as error:
        return f"event_publish_failed:{event.event_type}:{type(error).__name__}"
    return f"event_published:{event.event_type}"


def source_changed(
    tenant_id: str, source_id: str, source_version_id: str, content_hash: str
) -> PlatformEvent:
    return PlatformEvent(
        event_type="source.changed",
        tenant_id=tenant_id,
        subject=source_id,
        attributes={"source_version_id": source_version_id, "content_hash": content_hash},
    )


def ingestion_completed(
    tenant_id: str, index_version_id: str, source_version_id: str, chunk_count: int
) -> PlatformEvent:
    return PlatformEvent(
        event_type="ingestion.completed",
        tenant_id=tenant_id,
        subject=index_version_id,
        attributes={"source_version_id": source_version_id, "chunk_count": str(chunk_count)},
    )


def evaluation_completed(
    dataset_revision: str,
    config_version: str,
    *,
    passed: int,
    total: int,
    authorization_violations: int,
    tenant_id: str = "platform",
) -> PlatformEvent:
    return PlatformEvent(
        event_type="evaluation.completed",
        tenant_id=tenant_id,
        subject=dataset_revision,
        attributes={
            "config_version": config_version,
            "passed": str(passed),
            "total": str(total),
            "authorization_violations": str(authorization_violations),
        },
    )

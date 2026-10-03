from rag_platform.adapters.payload import from_payload, required_labels, to_payload
from rag_platform.models import DocumentChunk


def chunk(labels: frozenset[str]) -> DocumentChunk:
    return DocumentChunk(
        chunk_id="leave-0001",
        tenant_id="tenant-a",
        access_labels=labels,
        text="Annual leave requests use the HR portal.",
        source_uri="https://example.test/leave",
        source_title="Handbook",
        index_version="iv-tenant-a-abc123",
        source_version_id="sv-handbook-0001",
        parent_chunk_id="leave-0000",
        section_path=("Handbook", "Annual leave"),
        ordinal=1,
    )


def test_a_chunk_round_trips_through_the_payload_codec() -> None:
    original = chunk(frozenset({"public", "employees"}))
    assert from_payload(to_payload(original)) == original


def test_required_labels_exclude_public() -> None:
    payload = to_payload(chunk(frozenset({"public", "finance"})))
    assert payload["required_labels"] == ["finance"]
    assert payload["access_labels"] == ["finance", "public"]


def test_a_public_only_chunk_requires_no_label() -> None:
    assert required_labels(chunk(frozenset({"public"}))) == []
    assert to_payload(chunk(frozenset({"public"})))["required_labels"] == []


def test_every_required_label_is_recorded_for_a_multi_label_chunk() -> None:
    payload = to_payload(chunk(frozenset({"finance", "management"})))
    assert payload["required_labels"] == ["finance", "management"]


def test_the_payload_is_json_safe() -> None:
    payload = to_payload(chunk(frozenset({"public", "employees"})))
    assert isinstance(payload["access_labels"], list)
    assert isinstance(payload["section_path"], list)
    assert payload["retrievable"] is True

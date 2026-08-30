import pytest

from rag_platform.models import PiiPolicy
from rag_platform.pii import (
    PiiProcessor,
    PseudonymVault,
    RegexRecognizer,
    RehydrationApproval,
)

SENSITIVE = "Contact ada@example.com or +1 415 555 0132. Her identifier is 123-45-6789."
APPROVAL = RehydrationApproval(
    workflow="data_subject_request", approver="privacy-office", justification="erasure request"
)


def test_pseudonymization_replaces_every_recognized_entity() -> None:
    processor = PiiProcessor()
    result = processor.process(SENSITIVE, PiiPolicy(), "tenant-a")
    assert "ada@example.com" not in result.text
    assert "123-45-6789" not in result.text
    assert {entity.kind for entity in result.entities} == {"email", "phone", "national_id"}


def test_pseudonyms_are_stable_per_tenant_and_differ_across_tenants() -> None:
    processor = PiiProcessor()
    first = processor.process(SENSITIVE, PiiPolicy(), "tenant-a").text
    again = processor.process(SENSITIVE, PiiPolicy(), "tenant-a").text
    other = processor.process(SENSITIVE, PiiPolicy(), "tenant-b").text
    assert first == again
    assert first != other


def test_detect_mode_reports_entities_without_rewriting_or_storing_mappings() -> None:
    processor = PiiProcessor()
    result = processor.process(SENSITIVE, PiiPolicy(mode="detect"), "tenant-a")
    assert result.text == SENSITIVE
    assert result.entities
    assert len(processor.vault) == 0


def test_off_mode_is_a_no_operation() -> None:
    result = PiiProcessor().process(SENSITIVE, PiiPolicy(mode="off"), "tenant-a")
    assert result == type(result)(text=SENSITIVE, entities=())


def test_text_without_sensitive_data_is_untouched() -> None:
    result = PiiProcessor().process("Annual leave uses the HR portal.", PiiPolicy(), "tenant-a")
    assert result.entities == ()


def test_rehydration_requires_an_approved_workflow() -> None:
    processor = PiiProcessor()
    result = processor.process(SENSITIVE, PiiPolicy(), "tenant-a")
    email = next(entity for entity in result.entities if entity.kind == "email")

    assert processor.vault.rehydrate("tenant-a", email.pseudonym, APPROVAL) == "ada@example.com"
    with pytest.raises(PermissionError):
        processor.vault.rehydrate(
            "tenant-a",
            email.pseudonym,
            RehydrationApproval(workflow="legal_hold", approver=" ", justification="none"),
        )


def test_rehydration_is_scoped_to_the_owning_tenant() -> None:
    processor = PiiProcessor()
    result = processor.process(SENSITIVE, PiiPolicy(), "tenant-a")
    with pytest.raises(KeyError):
        processor.vault.rehydrate("tenant-b", result.entities[0].pseudonym, APPROVAL)


def test_unknown_recognizers_are_ignored() -> None:
    assert RegexRecognizer(("unsupported",)).detect(SENSITIVE) == []


def test_overlapping_spans_keep_only_the_widest_entity() -> None:
    # The phone pattern also matches the local part of this address.
    spans = RegexRecognizer(("email", "phone")).detect("write to 4155550132@example.com today")
    assert [kind for kind, _start, _end in spans] == ["email"]


def test_vault_reports_stored_mapping_count() -> None:
    vault = PseudonymVault()
    vault.register("tenant-a", "[EMAIL_1]", "ada@example.com")
    assert len(vault) == 1

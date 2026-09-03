import pytest

from rag_platform.config_registry import (
    ConfigPromotionError,
    PipelineConfigRegistry,
    UnknownConfigVersionError,
)
from rag_platform.models import PipelineConfig

INITIAL = PipelineConfig(version="pipeline-v1")
CANDIDATE = PipelineConfig(version="pipeline-v2", dense_top_k=10)


def test_a_new_registry_is_active_on_its_initial_config() -> None:
    registry = PipelineConfigRegistry(INITIAL)
    assert registry.active() == INITIAL
    assert registry.versions() == (INITIAL,)


def test_promoting_a_registered_version_makes_it_active() -> None:
    registry = PipelineConfigRegistry(INITIAL)
    registry.register(CANDIDATE)

    promoted = registry.promote("pipeline-v2")

    assert promoted == CANDIDATE
    assert registry.active() == CANDIDATE


def test_promoting_an_unregistered_version_fails() -> None:
    registry = PipelineConfigRegistry(INITIAL)
    with pytest.raises(UnknownConfigVersionError):
        registry.promote("pipeline-v2")


def test_rollback_restores_the_previously_active_version() -> None:
    registry = PipelineConfigRegistry(INITIAL)
    registry.register(CANDIDATE)
    registry.promote("pipeline-v2")

    restored = registry.rollback()

    assert restored == INITIAL
    assert registry.active() == INITIAL


def test_rollback_without_history_fails() -> None:
    registry = PipelineConfigRegistry(INITIAL)
    with pytest.raises(ConfigPromotionError):
        registry.rollback()


def test_promoting_the_already_active_version_does_not_grow_the_history() -> None:
    registry = PipelineConfigRegistry(INITIAL)
    registry.promote("pipeline-v1")
    with pytest.raises(ConfigPromotionError):
        registry.rollback()


def test_rollback_restores_the_immediately_previous_version_not_the_original() -> None:
    """One action steps back one hop; it is not a full undo of every promotion in the run."""
    third = PipelineConfig(version="pipeline-v3", dense_top_k=6)
    registry = PipelineConfigRegistry(INITIAL)
    registry.register(CANDIDATE)
    registry.register(third)
    registry.promote("pipeline-v2")
    registry.promote("pipeline-v3")

    restored = registry.rollback()

    assert restored == CANDIDATE
    assert registry.active() == CANDIDATE

"""Pipeline configuration registry (specification section 17).

Mirrors `IndexCatalog`'s staged-activation and rollback pattern, but for `PipelineConfig`:
promoting a candidate changes which configuration answers queries, and rollback restores the
previous one in one action, without rerunning optimization. Registration is separate from
promotion so every evaluated candidate can be kept for audit even when it is never promoted.
"""

from rag_platform.models import PipelineConfig


class UnknownConfigVersionError(LookupError):
    pass


class ConfigPromotionError(RuntimeError):
    pass


class PipelineConfigRegistry:
    def __init__(self, initial: PipelineConfig) -> None:
        self._versions: dict[str, PipelineConfig] = {initial.version: initial}
        self._active_version = initial.version
        self._history: list[str] = []

    def register(self, config: PipelineConfig) -> PipelineConfig:
        self._versions[config.version] = config
        return config

    def promote(self, version: str) -> PipelineConfig:
        candidate = self._require(version)
        if version != self._active_version:
            self._history.append(self._active_version)
        self._active_version = version
        return candidate

    def rollback(self) -> PipelineConfig:
        if not self._history:
            raise ConfigPromotionError("no previous pipeline configuration to roll back to")
        previous = self._history.pop()
        self._active_version = previous
        return self._versions[previous]

    def active(self) -> PipelineConfig:
        return self._versions[self._active_version]

    def versions(self) -> tuple[PipelineConfig, ...]:
        return tuple(self._versions.values())

    def _require(self, version: str) -> PipelineConfig:
        try:
            return self._versions[version]
        except KeyError as error:
            raise UnknownConfigVersionError(version) from error

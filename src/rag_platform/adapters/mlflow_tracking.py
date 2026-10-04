"""MLflow experiment governance (specification sections 5, 12 and 14).

Two responsibilities, both behind the plain `MlflowClient`:

- `MlflowRunLogger` records every evaluation and every optimization candidate as an MLflow run:
  the configuration as parameters, the summary as metrics, and the dataset revision, program
  revisions, constraint results and promotion disposition as tags. That is the `CandidateRun`
  evidence section 14 asks for, kept where a reviewer can compare candidates side by side.
- `MlflowConfigRegistry` is the durable form of `PipelineConfigRegistry`. Each registered
  `PipelineConfig` is a model version; the `champion` alias is the active configuration, and
  promotion and rollback move that alias. A process that restarts reads the active
  configuration back from MLflow instead of falling back to defaults.

Nothing here decides anything: the optimization loop still applies the constraints. MLflow
records what was decided and makes the decision reversible.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_platform.config_registry import ConfigPromotionError, UnknownConfigVersionError
from rag_platform.evaluation import EvaluationReport
from rag_platform.models import CandidateRun, EvaluationSummary, PipelineConfig

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

CHAMPION = "champion"
CONFIG_ARTIFACT = "pipeline_config.json"
_HISTORY_TAG = "rag.promotion_history"
_CONFIG_TAG = "rag.pipeline_config"
_VERSION_TAG = "rag.config_version"


def summary_metrics(summary: EvaluationSummary) -> dict[str, float]:
    return {
        "case_count": float(summary.case_count),
        "passed_count": float(summary.passed_count),
        "pass_rate": summary.pass_rate,
        "mean_recall_at_k": summary.mean_recall_at_k,
        "mean_reciprocal_rank": summary.mean_reciprocal_rank,
        "grounded_rate": summary.grounded_rate,
        "authorization_violations": float(summary.authorization_violations),
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
    }


def config_params(config: PipelineConfig) -> dict[str, str]:
    return {name: str(value) for name, value in config.model_dump().items()}


@dataclass(slots=True)
class MlflowRunLogger:
    client: "MlflowClient"
    experiment: str = "rag-platform"
    artifact_location: str | None = None
    """Where a newly created experiment keeps artifacts; None leaves it to the server."""

    def log_evaluation(self, report: EvaluationReport, config: PipelineConfig) -> str:
        """One run per evaluation of one configuration. Returns the MLflow run id."""
        failures = report.failures()
        tags = {
            "rag.kind": "evaluation",
            "rag.dataset_revision": report.dataset_revision,
            "rag.program_revisions": ",".join(report.program_revisions),
            "rag.failed_case_ids": ",".join(result.case_id for result in failures),
            "rag.failure_categories": ",".join(
                sorted({result.failure_category or "none" for result in failures})
            ),
        }
        return self._log(
            name=f"evaluation-{config.version}",
            config=config,
            summary=report.summary,
            tags=tags,
        )

    def log_candidate(self, run: CandidateRun) -> str:
        """One run per optimization candidate. Returns the MLflow run id."""
        tags = {
            "rag.kind": "candidate",
            "rag.run_id": run.run_id,
            "rag.baseline_config_version": run.baseline_config_version,
            "rag.config_diff": "; ".join(run.config_diff),
            "rag.dataset_revision": run.dataset_revision,
            "rag.program_revisions": ",".join(run.program_revisions),
            "rag.promotion": run.promotion,
            "rag.constraint_violations": json.dumps(
                [violation.model_dump() for violation in run.constraint_violations]
            ),
        }
        return self._log(name=run.run_id, config=run.config, summary=run.summary, tags=tags)

    def log_optimization(self, runs: Sequence[CandidateRun]) -> dict[str, str]:
        """Candidate run id to MLflow run id, for every candidate including the rejected."""
        return {run.run_id: self.log_candidate(run) for run in runs}

    def record_promotion(self, mlflow_run_id: str, run: CandidateRun) -> None:
        """Update the disposition after a canary; the metrics it was approved on are kept."""
        self.client.set_tag(mlflow_run_id, "rag.promotion", run.promotion)

    def _log(
        self,
        *,
        name: str,
        config: PipelineConfig,
        summary: EvaluationSummary,
        tags: dict[str, str],
    ) -> str:
        created = self.client.create_run(
            self._experiment_id(),
            run_name=name,
            tags={**tags, _VERSION_TAG: config.version},
        )
        run_id = str(created.info.run_id)
        for key, value in config_params(config).items():
            self.client.log_param(run_id, key, value)
        for key, metric in summary_metrics(summary).items():
            self.client.log_metric(run_id, key, metric)
        self.client.log_dict(run_id, config.model_dump(mode="json"), CONFIG_ARTIFACT)
        self.client.set_terminated(run_id)
        return run_id

    def _experiment_id(self) -> str:
        existing = self.client.get_experiment_by_name(self.experiment)
        if existing is not None:
            return str(existing.experiment_id)
        return str(
            self.client.create_experiment(self.experiment, artifact_location=self.artifact_location)
        )


@dataclass(slots=True)
class MlflowConfigRegistry:
    """`PipelineConfigRegistry` semantics, persisted in the MLflow model registry."""

    client: "MlflowClient"
    name: str = "rag-pipeline-config"

    def ensure(self, initial: PipelineConfig) -> PipelineConfig:
        """Create the registry on first use and return whichever configuration is active."""
        self._ensure_model()
        active = self._champion()
        if active is not None:
            return _config_of(active)
        self.register(initial)
        self._set_champion(initial.version)
        return initial

    def register(
        self, config: PipelineConfig, *, mlflow_run_id: str | None = None
    ) -> PipelineConfig:
        """Keep a configuration for audit. Registering the same version twice is a no-op."""
        self._ensure_model()
        if self._find(config.version) is None:
            source = (
                f"runs:/{mlflow_run_id}/{CONFIG_ARTIFACT}"
                if mlflow_run_id is not None
                else f"rag-pipeline-config://{config.version}"
            )
            self.client.create_model_version(
                self.name,
                source=source,
                run_id=mlflow_run_id,
                tags={_VERSION_TAG: config.version, _CONFIG_TAG: config.model_dump_json()},
            )
        return config

    def promote(self, version: str) -> PipelineConfig:
        target = self._require(version)
        active = self._champion()
        history = self._history()
        if active is not None and active.tags[_VERSION_TAG] != version:
            history.append(active.tags[_VERSION_TAG])
            self._write_history(history)
        self.client.set_registered_model_alias(self.name, CHAMPION, target.version)
        return _config_of(target)

    def rollback(self) -> PipelineConfig:
        history = self._history()
        if not history:
            raise ConfigPromotionError("no previous pipeline configuration to roll back to")
        previous = self._require(history.pop())
        self.client.set_registered_model_alias(self.name, CHAMPION, previous.version)
        self._write_history(history)
        return _config_of(previous)

    def active(self) -> PipelineConfig:
        active = self._champion()
        if active is None:
            raise ConfigPromotionError("no pipeline configuration has been promoted")
        return _config_of(active)

    def versions(self) -> tuple[PipelineConfig, ...]:
        found = self.client.search_model_versions(f"name='{self.name}'")
        ordered = sorted(found, key=lambda item: int(item.version))
        return tuple(_config_of(item) for item in ordered)

    # Internals ---------------------------------------------------------------
    def _ensure_model(self) -> None:
        from mlflow.exceptions import MlflowException

        try:
            self.client.get_registered_model(self.name)
        except MlflowException:
            self.client.create_registered_model(self.name)

    def _champion(self) -> Any | None:
        from mlflow.exceptions import MlflowException

        try:
            return self.client.get_model_version_by_alias(self.name, CHAMPION)
        except MlflowException:
            return None

    def _set_champion(self, version: str) -> None:
        self.client.set_registered_model_alias(self.name, CHAMPION, self._require(version).version)

    def _find(self, version: str) -> Any | None:
        for item in self.client.search_model_versions(f"name='{self.name}'"):
            if item.tags.get(_VERSION_TAG) == version:
                return item
        return None

    def _require(self, version: str) -> Any:
        found = self._find(version)
        if found is None:
            raise UnknownConfigVersionError(version)
        return found

    def _history(self) -> list[str]:
        tags = self.client.get_registered_model(self.name).tags
        return [str(item) for item in json.loads(tags.get(_HISTORY_TAG, "[]"))]

    def _write_history(self, history: list[str]) -> None:
        self.client.set_registered_model_tag(self.name, _HISTORY_TAG, json.dumps(history))


def _config_of(model_version: Any) -> PipelineConfig:
    return PipelineConfig.model_validate_json(model_version.tags[_CONFIG_TAG])


def client_for(tracking_uri: str) -> "MlflowClient":
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

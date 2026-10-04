"""Runtime configuration, read from the environment (specification section 18).

Every dependency is optional and selected by its own variable. With nothing set the platform
runs entirely in-process, which is what the demo and the unit suite use; setting a variable
swaps that one boundary to its production store and leaves the rest alone.

Settings are parsed once, at start-up, into an immutable object. A value that cannot be parsed
is a start-up error naming the variable, never a silent fallback to a default.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class SettingsError(ValueError):
    """An environment variable holds a value the platform cannot use."""


@dataclass(frozen=True, slots=True)
class Settings:
    # Stores
    qdrant_url: str | None = None
    opensearch_url: str | None = None
    neo4j_url: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    postgres_dsn: str | None = None
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_secure: bool = False
    minio_bucket: str = "rag-originals"
    redis_url: str | None = None
    cache_ttl_seconds: int = 300
    catalog_sync_seconds: int = 5
    """How stale a replica's view of another replica's activation may be."""
    kafka_bootstrap_servers: str | None = None
    kafka_topic_prefix: str = "rag."

    # Models and programs
    embedder: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    pii_recognizer: Literal["regex", "presidio"] = "regex"
    llm_model: str | None = None
    """A DSPy/LiteLLM model name. Unset, the deterministic programs answer."""
    program_artifacts: str | None = None

    # Composition
    workflow_executor: Literal["inprocess", "langgraph"] = "inprocess"
    chunker: Literal["builtin", "llamaindex"] = "builtin"
    seed_demo_sources: bool | None = None
    """None means: seed only when no durable catalog is configured."""

    # Observability and governance
    otlp_endpoint: str | None = None
    trace_sample_ratio: float = 1.0
    trace_capture_text: bool = False
    """Off by default: a deployment has to opt in to exporting question and answer text."""
    service_name: str = "rag-platform"
    mlflow_tracking_uri: str | None = None

    @property
    def seeds_demo_sources(self) -> bool:
        if self.seed_demo_sources is not None:
            return self.seed_demo_sources
        return self.postgres_dsn is None

    def configured_dependencies(self) -> tuple[str, ...]:
        """The external services this configuration talks to, for the readiness report."""
        candidates = {
            "qdrant": self.qdrant_url,
            "opensearch": self.opensearch_url,
            "neo4j": self.neo4j_url,
            "postgres": self.postgres_dsn,
            "minio": self.minio_endpoint,
            "redis": self.redis_url,
            "kafka": self.kafka_bootstrap_servers,
            "otlp": self.otlp_endpoint,
            "mlflow": self.mlflow_tracking_uri,
            "llm": self.llm_model,
        }
        return tuple(name for name, value in candidates.items() if value)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        read = _Reader(env)
        settings = cls(
            qdrant_url=read.text("RAG_QDRANT_URL"),
            opensearch_url=read.text("RAG_OPENSEARCH_URL"),
            neo4j_url=read.text("RAG_NEO4J_URL"),
            neo4j_user=read.text("RAG_NEO4J_USER") or "neo4j",
            neo4j_password=read.text("RAG_NEO4J_PASSWORD"),
            postgres_dsn=read.text("RAG_POSTGRES_DSN"),
            minio_endpoint=read.text("RAG_MINIO_ENDPOINT"),
            minio_access_key=read.text("RAG_MINIO_ACCESS_KEY"),
            minio_secret_key=read.text("RAG_MINIO_SECRET_KEY"),
            minio_secure=read.flag("RAG_MINIO_SECURE", False),
            minio_bucket=read.text("RAG_MINIO_BUCKET") or "rag-originals",
            redis_url=read.text("RAG_REDIS_URL"),
            cache_ttl_seconds=read.integer("RAG_CACHE_TTL_SECONDS", 300, minimum=1),
            catalog_sync_seconds=read.integer("RAG_CATALOG_SYNC_SECONDS", 5, minimum=0),
            kafka_bootstrap_servers=read.text("RAG_KAFKA_BOOTSTRAP_SERVERS"),
            kafka_topic_prefix=read.text("RAG_KAFKA_TOPIC_PREFIX") or "rag.",
            embedder=read.choice("RAG_EMBEDDER", ("hashing", "sentence-transformers"), "hashing"),
            embedding_model=read.text("RAG_EMBEDDING_MODEL")
            or "sentence-transformers/all-MiniLM-L6-v2",
            pii_recognizer=read.choice("RAG_PII_RECOGNIZER", ("regex", "presidio"), "regex"),
            llm_model=read.text("RAG_LLM_MODEL"),
            program_artifacts=read.text("RAG_PROGRAM_ARTIFACTS"),
            workflow_executor=read.choice(
                "RAG_WORKFLOW_EXECUTOR", ("inprocess", "langgraph"), "inprocess"
            ),
            chunker=read.choice("RAG_CHUNKER", ("builtin", "llamaindex"), "builtin"),
            seed_demo_sources=read.optional_flag("RAG_SEED_DEMO_SOURCES"),
            otlp_endpoint=read.text("RAG_OTLP_ENDPOINT"),
            trace_sample_ratio=read.ratio("RAG_TRACE_SAMPLE_RATIO", 1.0),
            trace_capture_text=read.flag("RAG_TRACE_CAPTURE_TEXT", False),
            service_name=read.text("RAG_SERVICE_NAME") or "rag-platform",
            mlflow_tracking_uri=read.text("RAG_MLFLOW_TRACKING_URI"),
        )
        settings._check()
        return settings

    def _check(self) -> None:
        if self.minio_endpoint and not (self.minio_access_key and self.minio_secret_key):
            raise SettingsError(
                "RAG_MINIO_ENDPOINT needs RAG_MINIO_ACCESS_KEY and RAG_MINIO_SECRET_KEY"
            )
        if self.neo4j_url and not self.neo4j_password:
            raise SettingsError("RAG_NEO4J_URL needs RAG_NEO4J_PASSWORD")


@dataclass(frozen=True, slots=True)
class _Reader:
    env: Mapping[str, str]

    def text(self, name: str) -> str | None:
        value = self.env.get(name, "").strip()
        return value or None

    def optional_flag(self, name: str) -> bool | None:
        value = self.text(name)
        if value is None:
            return None
        lowered = value.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise SettingsError(f"{name} must be true or false, not {value!r}")

    def flag(self, name: str, default: bool) -> bool:
        value = self.optional_flag(name)
        return default if value is None else value

    def integer(self, name: str, default: int, *, minimum: int) -> int:
        value = self.text(name)
        if value is None:
            return default
        try:
            parsed = int(value)
        except ValueError as error:
            raise SettingsError(f"{name} must be an integer, not {value!r}") from error
        if parsed < minimum:
            raise SettingsError(f"{name} must be at least {minimum}")
        return parsed

    def ratio(self, name: str, default: float) -> float:
        value = self.text(name)
        if value is None:
            return default
        try:
            parsed = float(value)
        except ValueError as error:
            raise SettingsError(f"{name} must be a number, not {value!r}") from error
        if not 0.0 <= parsed <= 1.0:
            raise SettingsError(f"{name} must be between 0 and 1")
        return parsed

    def choice(self, name: str, allowed: tuple[str, ...], default: str) -> Any:
        value = self.text(name)
        if value is None:
            return default
        if value not in allowed:
            raise SettingsError(f"{name} must be one of {', '.join(allowed)}, not {value!r}")
        return value

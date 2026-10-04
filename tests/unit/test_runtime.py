"""Settings parsing and the assembly of the platform from them.

The assembly is tested with fake clients that record what they were asked: the point is that
each setting swaps exactly its own boundary, and that reads follow the active index version.
tests/integration/test_full_stack.py runs the same assembly against the real services.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_platform.api import create_app
from rag_platform.bootstrap import build_platform
from rag_platform.catalog import IndexCatalog
from rag_platform.models import AccessContext
from rag_platform.runtime import ClientFactory, build_platform_from_settings, build_runtime
from rag_platform.service import QueryService
from rag_platform.settings import Settings, SettingsError

pytest.importorskip("qdrant_client")

from test_adapters_neo4j import FakeDriver
from test_adapters_opensearch import FakeOpenSearchClient
from test_adapters_originals import FakeMinio
from test_adapters_postgres import FakeConnection
from test_adapters_qdrant import FakeQdrantClient

ACME = AccessContext(tenant_id="tenant-acme", labels=frozenset({"public"}))
QUESTION = "How do I request annual leave?"


# Settings ----------------------------------------------------------------------


def test_an_empty_environment_configures_nothing_external() -> None:
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.configured_dependencies() == ()
    assert settings.seeds_demo_sources is True
    assert settings.trace_capture_text is False


def test_every_variable_is_read() -> None:
    settings = Settings.from_env(
        {
            "RAG_QDRANT_URL": "http://qdrant:6333",
            "RAG_OPENSEARCH_URL": "http://opensearch:9200",
            "RAG_NEO4J_URL": "bolt://neo4j:7687",
            "RAG_NEO4J_PASSWORD": "secret",
            "RAG_POSTGRES_DSN": "postgresql://rag@postgres/rag",
            "RAG_MINIO_ENDPOINT": "minio:9000",
            "RAG_MINIO_ACCESS_KEY": "key",
            "RAG_MINIO_SECRET_KEY": "secret",
            "RAG_MINIO_SECURE": "true",
            "RAG_REDIS_URL": "redis://redis:6379/0",
            "RAG_CACHE_TTL_SECONDS": "60",
            "RAG_KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
            "RAG_OTLP_ENDPOINT": "http://collector:4318",
            "RAG_TRACE_SAMPLE_RATIO": "0.25",
            "RAG_TRACE_CAPTURE_TEXT": "yes",
            "RAG_MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "RAG_PII_RECOGNIZER": "presidio",
            "RAG_WORKFLOW_EXECUTOR": "langgraph",
            "RAG_CHUNKER": "llamaindex",
            "RAG_LLM_MODEL": "anthropic/claude-sonnet-5-5",
        }
    )
    assert settings.minio_secure is True
    assert settings.cache_ttl_seconds == 60
    assert settings.trace_sample_ratio == 0.25
    assert settings.trace_capture_text is True
    assert settings.workflow_executor == "langgraph"
    assert settings.configured_dependencies() == (
        "qdrant",
        "opensearch",
        "neo4j",
        "postgres",
        "minio",
        "redis",
        "kafka",
        "otlp",
        "mlflow",
        "llm",
    )


def test_blank_values_count_as_unset() -> None:
    assert Settings.from_env({"RAG_QDRANT_URL": "  ", "RAG_SEED_DEMO_SOURCES": ""}) == Settings()


def test_a_durable_catalog_turns_demo_seeding_off_unless_asked() -> None:
    dsn = {"RAG_POSTGRES_DSN": "postgresql://rag@postgres/rag"}
    assert Settings.from_env(dsn).seeds_demo_sources is False
    assert Settings.from_env({**dsn, "RAG_SEED_DEMO_SOURCES": "true"}).seeds_demo_sources is True
    assert Settings.from_env({"RAG_SEED_DEMO_SOURCES": "off"}).seeds_demo_sources is False


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"RAG_TRACE_SAMPLE_RATIO": "2"}, "RAG_TRACE_SAMPLE_RATIO must be between 0 and 1"),
        ({"RAG_TRACE_SAMPLE_RATIO": "half"}, "RAG_TRACE_SAMPLE_RATIO must be a number"),
        ({"RAG_CACHE_TTL_SECONDS": "soon"}, "RAG_CACHE_TTL_SECONDS must be an integer"),
        ({"RAG_CACHE_TTL_SECONDS": "0"}, "RAG_CACHE_TTL_SECONDS must be at least 1"),
        ({"RAG_MINIO_SECURE": "maybe"}, "RAG_MINIO_SECURE must be true or false"),
        ({"RAG_WORKFLOW_EXECUTOR": "airflow"}, "RAG_WORKFLOW_EXECUTOR must be one of"),
        ({"RAG_MINIO_ENDPOINT": "minio:9000"}, "RAG_MINIO_ENDPOINT needs"),
        ({"RAG_NEO4J_URL": "bolt://neo4j:7687"}, "RAG_NEO4J_URL needs RAG_NEO4J_PASSWORD"),
    ],
)
def test_an_unusable_value_is_a_start_up_error_naming_the_variable(
    env: dict[str, str], message: str
) -> None:
    with pytest.raises(SettingsError, match=message):
        Settings.from_env(env)


# Assembly ----------------------------------------------------------------------


@dataclass
class FakeRedis:
    values: dict[str, bytes] = field(default_factory=dict)

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int) -> None:
        self.values[key] = value.encode("utf-8")


@dataclass
class FakeProducer:
    topics: list[str] = field(default_factory=list)

    def produce(self, topic: str, *, key: bytes, value: bytes, on_delivery: Any) -> None:
        self.topics.append(topic)

    def flush(self, timeout: float) -> int:
        return 0


@dataclass
class FakeExporter:
    spans: list[Any] = field(default_factory=list)
    stopped: bool = False

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.stopped = True

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


@dataclass
class FakeClients(ClientFactory):
    qdrant_client: FakeQdrantClient = field(default_factory=FakeQdrantClient)
    opensearch_client: FakeOpenSearchClient = field(default_factory=FakeOpenSearchClient)
    driver: FakeDriver = field(default_factory=FakeDriver)
    connection: FakeConnection = field(default_factory=FakeConnection)
    minio_client: FakeMinio = field(default_factory=FakeMinio)
    redis_client: FakeRedis = field(default_factory=FakeRedis)
    producer: FakeProducer = field(default_factory=FakeProducer)
    exporter: FakeExporter = field(default_factory=FakeExporter)
    closed: list[str] = field(default_factory=list)

    def qdrant(self, url: str) -> Any:
        return self.qdrant_client

    def opensearch(self, url: str) -> Any:
        return self.opensearch_client

    def neo4j(self, url: str, user: str, password: str) -> Any:
        self.driver.close = lambda: self.closed.append("neo4j")  # type: ignore[attr-defined]
        return self.driver

    def postgres(self, dsn: str) -> Any:
        self.connection.close = lambda: self.closed.append("postgres")  # type: ignore[attr-defined]
        return self.connection

    def minio(self, endpoint: str, access_key: str, secret_key: str, secure: bool) -> Any:
        return self.minio_client

    def redis(self, url: str) -> Any:
        return self.redis_client

    def kafka_producer(self, bootstrap_servers: str) -> Any:
        return self.producer

    def span_exporter(self, endpoint: str) -> Any:
        return self.exporter


def test_empty_settings_build_the_in_process_platform() -> None:
    platform = build_platform_from_settings(Settings())
    reference = build_platform()

    assert type(platform.catalog) is IndexCatalog
    assert isinstance(platform.query_service, QueryService)
    ours = platform.query_service.answer(QUESTION, ACME).model_dump()
    theirs = reference.query_service.answer(QUESTION, ACME).model_dump()
    ours["trace"].pop("latency_ms")
    theirs["trace"].pop("latency_ms")
    assert ours == theirs


def test_configured_stores_receive_each_index_version_under_its_own_name() -> None:
    clients = FakeClients()
    settings = Settings(
        qdrant_url="http://qdrant:6333",
        opensearch_url="http://opensearch:9200",
        neo4j_url="bolt://neo4j:7687",
        neo4j_password="secret",
    )
    platform = build_runtime(settings, clients=clients).platform
    active = platform.catalog.active_index_version("tenant-acme")
    assert active is not None

    assert active.physical_dense_index in clients.qdrant_client.created
    assert active.physical_sparse_index in clients.opensearch_client.indices.created
    assert clients.qdrant_client.upserted
    graph_writes = [params for _query, params in clients.driver.calls if "graph_index" in params]
    assert active.physical_graph_index in {params["graph_index"] for params in graph_writes}


def test_queries_read_the_active_index_version_from_the_configured_stores() -> None:
    clients = FakeClients()
    settings = Settings(qdrant_url="http://qdrant:6333", opensearch_url="http://opensearch:9200")
    platform = build_runtime(settings, clients=clients).platform
    active = platform.catalog.active_index_version("tenant-acme")
    assert active is not None

    # The fakes hold no documents, so the stores, not the catalog, decided this answer.
    response = platform.query_service.answer(QUESTION, ACME)

    assert response.status == "insufficient_evidence"
    assert clients.qdrant_client.queries[-1]["collection_name"] == active.physical_dense_index
    assert clients.opensearch_client.searches


def test_a_leg_without_a_store_is_served_from_the_catalog() -> None:
    clients = FakeClients()
    platform = build_runtime(Settings(qdrant_url="http://qdrant:6333"), clients=clients).platform

    response = platform.query_service.answer(QUESTION, ACME)

    assert response.status == "answered"
    assert clients.qdrant_client.queries
    assert clients.opensearch_client.searches == []


def test_an_unknown_tenant_reads_nothing_from_any_store() -> None:
    clients = FakeClients()
    settings = Settings(qdrant_url="http://qdrant:6333", opensearch_url="http://opensearch:9200")
    platform = build_runtime(settings, clients=clients).platform
    queries_before = len(clients.qdrant_client.queries)

    response = platform.query_service.answer(QUESTION, AccessContext(tenant_id="tenant-initech"))

    assert response.status == "insufficient_evidence"
    assert len(clients.qdrant_client.queries) == queries_before


def test_a_durable_catalog_is_loaded_and_not_reseeded() -> None:
    clients = FakeClients()
    runtime = build_runtime(Settings(postgres_dsn="postgresql://rag@postgres/rag"), clients=clients)

    statements = [statement for statement, _params in clients.connection.statements]
    assert any("CREATE TABLE IF NOT EXISTS" in statement for statement in statements)
    assert any("FROM rag_index_versions" in statement for statement in statements)
    assert not any(statement.startswith("INSERT") for statement in statements)
    assert runtime.platform.catalog.active_index_version("tenant-acme") is None


def test_originals_cache_events_and_tracing_are_each_switched_by_their_setting() -> None:
    pytest.importorskip("minio")
    pytest.importorskip("opentelemetry.sdk")
    clients = FakeClients()
    settings = Settings(
        minio_endpoint="minio:9000",
        minio_access_key="key",
        minio_secret_key="secret",
        redis_url="redis://redis:6379/0",
        kafka_bootstrap_servers="kafka:9092",
        otlp_endpoint="http://collector:4318",
    )
    runtime = build_runtime(settings, clients=clients)
    platform = runtime.platform

    assert clients.minio_client.buckets == {"rag-originals"}
    assert len(clients.minio_client.objects) == 3
    assert "rag.ingestion-completed" in clients.producer.topics

    first = platform.query_service.answer(QUESTION, ACME)
    second = platform.query_service.answer(QUESTION, ACME)
    assert first.trace.trace_id is not None
    assert first.trace.served_from_cache is False
    assert second.trace.served_from_cache is True
    assert len(clients.redis_client.values) == 1

    runtime.close()
    assert clients.exporter.stopped is True
    assert clients.exporter.spans
    assert "input.value" not in dict(clients.exporter.spans[-1].attributes or {})


def test_closing_releases_every_connection_even_if_one_fails() -> None:
    clients = FakeClients()
    settings = Settings(
        postgres_dsn="postgresql://rag@postgres/rag",
        neo4j_url="bolt://neo4j:7687",
        neo4j_password="secret",
    )
    runtime = build_runtime(settings, clients=clients)

    def broken() -> None:
        raise RuntimeError("already closed")

    runtime.closers.append(broken)
    runtime.close()
    runtime.close()

    assert sorted(clients.closed) == ["neo4j", "postgres"]


def test_the_composition_settings_select_langgraph_and_llamaindex() -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("llama_index.core")
    platform = build_platform_from_settings(
        Settings(workflow_executor="langgraph", chunker="llamaindex")
    )

    response = platform.query_service.answer(QUESTION, ACME)
    [source_version_id] = {
        chunk.source_version_id for chunk in platform.catalog.active_chunks("tenant-globex")
    }

    assert response.status == "answered"
    assert "HR portal" in response.answer
    assert (
        platform.catalog.source_version(source_version_id).transformation_revision
        == "transform-llamaindex-v1"
    )


def test_presidio_and_dspy_are_selected_by_their_settings() -> None:
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("numpy")
    dspy = pytest.importorskip("dspy")

    @dataclass
    class ModelClients(ClientFactory):
        models: list[str] = field(default_factory=list)

        def language_model(self, model: str) -> Any:
            self.models.append(model)
            return dspy.utils.DummyLM([{"intent": "factual"}] * 4)

    clients = ModelClients()
    settings = Settings(pii_recognizer="presidio", llm_model="provider/model")
    platform = build_runtime(settings, clients=clients).platform

    assert clients.models == ["provider/model"]
    assert platform.query_service.programs.revisions()[0].startswith("programs-dspy-")
    assert type(platform.ingestion.pii.recognizer_factory).__name__ == "PresidioRecognizerFactory"


# API ---------------------------------------------------------------------------


def test_the_api_assembles_itself_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RAG_QDRANT_URL", "RAG_POSTGRES_DSN", "RAG_OTLP_ENDPOINT", "RAG_REDIS_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_MLFLOW_TRACKING_URI", "http://mlflow:5000")

    with TestClient(create_app()) as client:
        ready = client.get("/readyz")
        answer = client.post(
            "/v1/query", headers={"X-Tenant-ID": "tenant-acme"}, json={"question": QUESTION}
        )

    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "dependencies": ["mlflow"]}
    assert answer.json()["status"] == "answered"


def test_an_injected_platform_is_used_as_is() -> None:
    platform = build_platform(seed_demo_sources=False)
    with TestClient(create_app(platform)) as client:
        ready = client.get("/readyz")
        answer = client.post(
            "/v1/query", headers={"X-Tenant-ID": "tenant-acme"}, json={"question": QUESTION}
        )
    assert ready.json()["dependencies"] == []
    assert answer.json()["status"] == "insufficient_evidence"


def test_a_bad_environment_stops_the_api_from_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TRACE_SAMPLE_RATIO", "lots")
    with pytest.raises(SettingsError), TestClient(create_app()):
        pass

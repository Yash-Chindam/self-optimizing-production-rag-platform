"""Assembling the platform from `Settings` (specification section 18).

This is the one place that knows which adapter stands behind which boundary. With empty settings
it builds exactly what `bootstrap.build_platform` builds; each configured dependency replaces
one in-process implementation:

    RAG_POSTGRES_DSN      catalog, feedback PostgreSQL (kept in step across replicas)
    RAG_QDRANT_URL        dense retrieval   Qdrant collection per index version
    RAG_OPENSEARCH_URL    sparse retrieval  OpenSearch index per index version
    RAG_NEO4J_URL         graph expansion   Neo4j subgraph per index version
    RAG_MINIO_ENDPOINT    originals         MinIO bucket
    RAG_REDIS_URL         answer cache      Redis
    RAG_KAFKA_BOOTSTRAP_SERVERS  events     Kafka
    RAG_OTLP_ENDPOINT     tracing           OpenTelemetry Collector
    RAG_PII_RECOGNIZER    recognition       Presidio
    RAG_LLM_MODEL         language programs DSPy
    RAG_WORKFLOW_EXECUTOR executor          LangGraph
    RAG_CHUNKER           chunking          LlamaIndex

Client construction goes through `ClientFactory`, so the assembly can be tested with fakes and
a deployment can substitute pooled or authenticated clients without touching this module.
"""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_platform.bootstrap import DEMO_SOURCES, Platform
from rag_platform.catalog import IndexCatalog
from rag_platform.context import ContextBuilder
from rag_platform.events import EventPublisher
from rag_platform.feedback import FeedbackLog
from rag_platform.ingestion import IngestionPipeline, PieceBuilder
from rag_platform.models import PipelineConfig
from rag_platform.optimization import OptimizationContext
from rag_platform.pii import PiiProcessor
from rag_platform.programs import ProgramSuite
from rag_platform.rerank import LexicalCrossEncoder
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService
from rag_platform.settings import Settings
from rag_platform.workflow import QueryWorkflow, WorkflowObserver


@dataclass(slots=True)
class ClientFactory:
    """Creates the third-party clients. Each method imports its library only when called."""

    def qdrant(self, url: str) -> Any:
        from qdrant_client import QdrantClient

        return QdrantClient(url=url)

    def opensearch(self, url: str) -> Any:
        from opensearchpy import OpenSearch

        return OpenSearch(hosts=[url])

    def neo4j(self, url: str, user: str, password: str) -> Any:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(url, auth=(user, password))

    def postgres(self, dsn: str) -> Any:
        import psycopg

        return psycopg.connect(dsn)

    def minio(self, endpoint: str, access_key: str, secret_key: str, secure: bool) -> Any:
        from minio import Minio

        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def redis(self, url: str) -> Any:
        import redis

        return redis.Redis.from_url(url)

    def kafka_producer(self, bootstrap_servers: str) -> Any:
        from rag_platform.adapters.kafka_events import producer_for

        return producer_for(bootstrap_servers)

    def language_model(self, model: str) -> Any:
        import dspy

        return dspy.LM(model)

    def presidio_analyzer(self) -> Any:
        from rag_platform.adapters.presidio import build_analyzer

        return build_analyzer()

    def span_exporter(self, endpoint: str) -> Any:
        from rag_platform.adapters.tracing import otlp_exporter

        return otlp_exporter(endpoint)


@dataclass(slots=True)
class Runtime:
    """The assembled platform plus what has to be closed when the process stops."""

    platform: Platform
    settings: Settings
    closers: list[Callable[[], None]] = field(default_factory=list)

    def close(self) -> None:
        """Flush telemetry and release connections. Safe to call more than once."""
        while self.closers:
            closer = self.closers.pop()
            # Shutdown has to reach every closer, whatever an earlier one does.
            with contextlib.suppress(Exception):
                closer()


def build_runtime(
    settings: Settings,
    *,
    clients: ClientFactory | None = None,
    config: PipelineConfig | None = None,
) -> Runtime:
    factory = clients or ClientFactory()
    config = config or PipelineConfig()
    closers: list[Callable[[], None]] = []

    catalog = _catalog(settings, factory, closers)
    embedder = _embedder(settings)
    qdrant = factory.qdrant(settings.qdrant_url) if settings.qdrant_url else None
    opensearch = factory.opensearch(settings.opensearch_url) if settings.opensearch_url else None
    neo4j = _neo4j(settings, factory, closers)

    ingestion = IngestionPipeline(
        catalog=catalog,
        pii=_pii(settings, factory),
        # The revision names what embeds the dense index, so it only changes with a real one.
        embedding_revision=(
            str(getattr(embedder, "revision", "embed-v1")) if qdrant is not None else "embed-v1"
        ),
        mirrors=_mirrors(qdrant, opensearch, neo4j, embedder),
        originals=_originals(settings, factory),
        piece_builder=_piece_builder(settings),
        transformation_revision=_transformation_revision(settings),
        events=_events(settings, factory),
    )
    if settings.seeds_demo_sources:
        for registration, content in DEMO_SOURCES:
            ingestion.ingest(registration, content)

    from rag_platform.adapters.active_index import ActiveIndexRepository, graph_retriever_for

    repository = ActiveIndexRepository(
        catalog=catalog, qdrant_client=qdrant, opensearch_client=opensearch, embedder=embedder
    )
    graph_retriever = graph_retriever_for(catalog, neo4j, config.graph_extractor_revision)
    reranker = LexicalCrossEncoder(revision=config.reranker_revision)
    programs = _programs(settings, factory, config)
    service = QueryService(
        HybridRetriever(repository, config, graph_retriever=graph_retriever, reranker=reranker),
        config,
        ContextBuilder(config, catalog.chunk),
        programs,
        observer=_observer(settings, factory, closers),
        workflow_class=_workflow_class(settings),
    )
    platform = Platform(
        config=config,
        catalog=catalog,
        ingestion=ingestion,
        query_service=_cached(settings, factory, service, catalog),
        feedback=_feedback(settings, catalog),
        optimization=OptimizationContext(
            repository=repository,
            programs=service.programs,
            graph_retriever=graph_retriever,
            reranker=reranker,
            parent_lookup=catalog.chunk,
        ),
    )
    return Runtime(platform=platform, settings=settings, closers=closers)


def build_platform_from_settings(settings: Settings) -> Platform:
    return build_runtime(settings).platform


# Parts -------------------------------------------------------------------------


def _catalog(
    settings: Settings, factory: ClientFactory, closers: list[Callable[[], None]]
) -> IndexCatalog:
    if not settings.postgres_dsn:
        return IndexCatalog()
    from rag_platform.adapters.postgres import PostgresIndexCatalog

    connection = factory.postgres(settings.postgres_dsn)
    closers.append(connection.close)
    catalog = PostgresIndexCatalog(
        connection, sync_interval_seconds=float(settings.catalog_sync_seconds)
    )
    catalog.ensure_schema()
    catalog.load()
    return catalog


def _feedback(settings: Settings, catalog: IndexCatalog) -> FeedbackLog:
    if not settings.postgres_dsn:
        return FeedbackLog()
    from rag_platform.adapters.postgres import PostgresFeedbackLog, PostgresIndexCatalog

    # Shares the catalog's connection: one connection per process to close.
    assert isinstance(catalog, PostgresIndexCatalog)
    log = PostgresFeedbackLog(connection=catalog.connection)
    log.ensure_schema()
    return log


def _embedder(settings: Settings) -> Any:
    if settings.embedder == "sentence-transformers":
        from rag_platform.adapters.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=settings.embedding_model)
    from rag_platform.adapters.embeddings import HashingEmbedder

    return HashingEmbedder()


def _neo4j(
    settings: Settings, factory: ClientFactory, closers: list[Callable[[], None]]
) -> Any | None:
    if not settings.neo4j_url:
        return None
    driver = factory.neo4j(settings.neo4j_url, settings.neo4j_user, settings.neo4j_password or "")
    closers.append(driver.close)
    return driver


def _mirrors(qdrant: Any, opensearch: Any, neo4j: Any, embedder: Any) -> tuple[Any, ...]:
    mirrors: list[Any] = []
    if qdrant is not None:
        from rag_platform.adapters.mirror import QdrantMirror

        mirrors.append(QdrantMirror(client=qdrant, embedder=embedder))
    if opensearch is not None:
        from rag_platform.adapters.mirror import OpenSearchMirror

        mirrors.append(OpenSearchMirror(client=opensearch))
    if neo4j is not None:
        from rag_platform.adapters.neo4j_graph import Neo4jGraphMirror

        graph_mirror = Neo4jGraphMirror(driver=neo4j)
        graph_mirror.ensure_schema()
        mirrors.append(graph_mirror)
    return tuple(mirrors)


def _originals(settings: Settings, factory: ClientFactory) -> Any | None:
    if not settings.minio_endpoint:
        return None
    from rag_platform.adapters.originals import MinioOriginalStore

    store = MinioOriginalStore(
        client=factory.minio(
            settings.minio_endpoint,
            settings.minio_access_key or "",
            settings.minio_secret_key or "",
            settings.minio_secure,
        ),
        bucket=settings.minio_bucket,
    )
    store.ensure_bucket()
    return store


def _pii(settings: Settings, factory: ClientFactory) -> PiiProcessor:
    if settings.pii_recognizer != "presidio":
        return PiiProcessor()
    from rag_platform.adapters.presidio import PresidioRecognizerFactory

    return PiiProcessor(
        recognizer_factory=PresidioRecognizerFactory(analyzer=factory.presidio_analyzer())
    )


def _piece_builder(settings: Settings) -> PieceBuilder:
    if settings.chunker == "llamaindex":
        from rag_platform.adapters.llamaindex_ingestion import LlamaIndexPieceBuilder

        return LlamaIndexPieceBuilder()
    from rag_platform.chunking import build_pieces

    return build_pieces


def _transformation_revision(settings: Settings) -> str:
    return "transform-llamaindex-v1" if settings.chunker == "llamaindex" else "transform-v1"


def _events(settings: Settings, factory: ClientFactory) -> EventPublisher | None:
    if not settings.kafka_bootstrap_servers:
        return None
    from rag_platform.adapters.kafka_events import KafkaEventPublisher

    return KafkaEventPublisher(
        producer=factory.kafka_producer(settings.kafka_bootstrap_servers),
        prefix=settings.kafka_topic_prefix,
    )


def _programs(settings: Settings, factory: ClientFactory, config: PipelineConfig) -> ProgramSuite:
    if not settings.llm_model:
        return ProgramSuite(revision=config.program_suite_revision)
    from rag_platform.adapters.dspy_programs import build_program_suite

    artifacts = Path(settings.program_artifacts) if settings.program_artifacts else None
    return build_program_suite(factory.language_model(settings.llm_model), artifacts=artifacts)


def _observer(
    settings: Settings, factory: ClientFactory, closers: list[Callable[[], None]]
) -> WorkflowObserver | None:
    if not settings.otlp_endpoint:
        return None
    from rag_platform.adapters.tracing import (
        OpenTelemetryObserver,
        TracePolicy,
        build_tracer_provider,
    )

    provider = build_tracer_provider(
        factory.span_exporter(settings.otlp_endpoint),
        sample_ratio=settings.trace_sample_ratio,
        service_name=settings.service_name,
        project_name=settings.service_name,
    )
    closers.append(provider.shutdown)
    return OpenTelemetryObserver(provider, TracePolicy(capture_text=settings.trace_capture_text))


def _workflow_class(settings: Settings) -> type[QueryWorkflow]:
    if settings.workflow_executor == "langgraph":
        from rag_platform.adapters.langgraph_workflow import LangGraphQueryWorkflow

        return LangGraphQueryWorkflow
    return QueryWorkflow


def _cached(
    settings: Settings, factory: ClientFactory, service: QueryService, catalog: IndexCatalog
) -> Any:
    if not settings.redis_url:
        return service
    from rag_platform.adapters.cache import CachedQueryService, RedisAnswerCache

    def active_index_version(tenant_id: str) -> str | None:
        active = catalog.active_index_version(tenant_id)
        return None if active is None else active.index_version_id

    return CachedQueryService(
        service=service,
        cache=RedisAnswerCache(client=factory.redis(settings.redis_url)),
        active_index_version=active_index_version,
        ttl_seconds=settings.cache_ttl_seconds,
    )

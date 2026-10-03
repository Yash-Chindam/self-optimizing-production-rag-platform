from dataclasses import dataclass, field

from rag_platform.adapters.cache import (
    CachedQueryService,
    InMemoryAnswerCache,
    RedisAnswerCache,
    cache_key,
)
from rag_platform.context import ContextBuilder
from rag_platform.models import AccessContext, DocumentChunk, PipelineConfig, QueryResponse
from rag_platform.reliability import CircuitBreaker
from rag_platform.repository import InMemoryChunkRepository
from rag_platform.retrieval import HybridRetriever
from rag_platform.service import QueryService

EMPLOYEE = AccessContext(tenant_id="tenant-a", labels=frozenset({"public", "employees"}))
FINANCE = AccessContext(tenant_id="tenant-a", labels=frozenset({"public", "finance"}))
QUESTION = "How do I request annual leave?"

CORPUS = [
    DocumentChunk(
        chunk_id="leave",
        tenant_id="tenant-a",
        access_labels=frozenset({"public"}),
        text="Annual leave requests use the HR portal.",
        source_uri="https://example.test/leave",
        source_title="Handbook",
        index_version="iv-1",
    )
]


@dataclass
class CountingService:
    inner: QueryService
    calls: int = 0

    @property
    def config(self) -> PipelineConfig:
        return self.inner.config

    @property
    def programs(self) -> object:
        return self.inner.programs

    def answer(self, question: str, access: AccessContext) -> QueryResponse:
        self.calls += 1
        return self.inner.answer(question, access)


@dataclass
class ActiveIndex:
    version: str | None = "iv-1"

    def __call__(self, tenant_id: str) -> str | None:
        return self.version


def build(
    cache: object | None = None, active: ActiveIndex | None = None
) -> tuple[CachedQueryService, CountingService]:
    config = PipelineConfig()
    inner = CountingService(
        QueryService(
            HybridRetriever(InMemoryChunkRepository(CORPUS), config), config, ContextBuilder(config)
        )
    )
    cached = CachedQueryService(
        service=inner,  # type: ignore[arg-type]
        cache=cache or InMemoryAnswerCache(),  # type: ignore[arg-type]
        active_index_version=active or ActiveIndex(),
    )
    return cached, inner


def key(**overrides: object) -> str:
    arguments: dict[str, object] = {
        "config_version": "pipeline-v1",
        "program_revisions": ("programs-v1",),
        "index_version_id": "iv-1",
    }
    arguments.update(overrides)
    return cache_key(QUESTION, EMPLOYEE, **arguments)  # type: ignore[arg-type]


# Key construction --------------------------------------------------------------


def test_the_key_is_stable_and_ignores_case_and_spacing() -> None:
    spaced = cache_key(
        "  how do I   request ANNUAL leave? ",
        EMPLOYEE,
        config_version="pipeline-v1",
        program_revisions=("programs-v1",),
        index_version_id="iv-1",
    )
    assert spaced == key()


def test_callers_with_different_labels_never_share_an_entry() -> None:
    other = cache_key(
        QUESTION,
        FINANCE,
        config_version="pipeline-v1",
        program_revisions=("programs-v1",),
        index_version_id="iv-1",
    )
    assert other != key()


def test_a_new_index_version_config_or_program_changes_the_key() -> None:
    assert key(index_version_id="iv-2") != key()
    assert key(config_version="pipeline-v2") != key()
    assert key(program_revisions=("programs-v2",)) != key()


# Behavior ----------------------------------------------------------------------


def test_a_repeated_question_is_served_from_the_cache() -> None:
    cached, inner = build()
    first = cached.answer(QUESTION, EMPLOYEE)
    second = cached.answer(QUESTION, EMPLOYEE)

    assert inner.calls == 1
    assert first.trace.served_from_cache is False
    assert second.trace.served_from_cache is True
    assert second.answer == first.answer
    assert second.citations == first.citations


def test_a_different_label_set_misses_the_cache() -> None:
    cached, inner = build()
    cached.answer(QUESTION, EMPLOYEE)
    cached.answer(QUESTION, FINANCE)
    assert inner.calls == 2


def test_activating_a_new_index_version_misses_the_cache() -> None:
    active = ActiveIndex("iv-1")
    cached, inner = build(active=active)
    cached.answer(QUESTION, EMPLOYEE)
    active.version = "iv-2"
    response = cached.answer(QUESTION, EMPLOYEE)

    assert inner.calls == 2
    assert response.trace.served_from_cache is False


def test_the_wrapper_reports_the_wrapped_services_config_and_programs() -> None:
    cached, inner = build()
    assert cached.config is inner.config
    assert cached.programs is inner.programs


# Degradation -------------------------------------------------------------------


@dataclass
class BrokenCache:
    fail_on: str = "get"
    writes: int = 0

    def get(self, key: str) -> str | None:
        if self.fail_on == "get":
            raise ConnectionError("redis unavailable")
        return None

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        self.writes += 1
        raise ConnectionError("redis unavailable")


def test_a_failing_cache_read_degrades_to_answering_directly() -> None:
    cached, inner = build(cache=BrokenCache("get"))
    response = cached.answer(QUESTION, EMPLOYEE)

    assert response.status == "answered"
    assert inner.calls == 1
    assert any(
        note.startswith("answer_cache_failed") for note in response.trace.degraded_dependencies
    )


def test_a_failing_cache_write_still_returns_the_answer() -> None:
    cached, _ = build(cache=BrokenCache("set"))
    response = cached.answer(QUESTION, EMPLOYEE)

    assert response.status == "answered"
    assert any(
        note.startswith("answer_cache_failed") for note in response.trace.degraded_dependencies
    )


def test_an_open_circuit_stops_calling_the_cache() -> None:
    broken = BrokenCache("set")
    cached, _ = build(cache=broken)
    cached.breaker = CircuitBreaker(name="answer_cache", failure_threshold=1)
    cached.answer(QUESTION, EMPLOYEE)
    cached.answer(QUESTION, EMPLOYEE)
    assert broken.writes == 1


# Redis -------------------------------------------------------------------------


@dataclass
class FakeRedis:
    store: dict[str, bytes] = field(default_factory=dict)
    expiries: dict[str, int] = field(default_factory=dict)

    def get(self, name: str) -> bytes | None:
        return self.store.get(name)

    def set(self, name: str, value: str, ex: int | None = None) -> None:
        self.store[name] = value.encode("utf-8")
        if ex is not None:
            self.expiries[name] = ex


def test_redis_entries_are_prefixed_and_expire() -> None:
    client = FakeRedis()
    cache = RedisAnswerCache(client=client)  # type: ignore[arg-type]
    cache.set("abc", "value", ttl_seconds=60)

    assert client.store == {"rag:answer:abc": b"value"}
    assert client.expiries == {"rag:answer:abc": 60}


def test_redis_returns_text_and_none_for_a_miss() -> None:
    cache = RedisAnswerCache(client=FakeRedis())  # type: ignore[arg-type]
    assert cache.get("missing") is None
    cache.set("abc", "value", ttl_seconds=60)
    assert cache.get("abc") == "value"


def test_a_client_that_already_decodes_responses_is_supported() -> None:
    class DecodingRedis(FakeRedis):
        def get(self, name: str) -> str | None:  # type: ignore[override]
            return "decoded"

    cache = RedisAnswerCache(client=DecodingRedis())  # type: ignore[arg-type]
    assert cache.get("abc") == "decoded"

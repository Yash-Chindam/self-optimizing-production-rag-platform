"""Answer caching in Redis (specification sections 16 and 18).

A cached answer is only reusable if nothing that produced it has changed, so the key is built
from everything that did: tenant, the caller's exact label set, the question, the pipeline
configuration version, the program revisions and the tenant's active index version. Activating
or rolling back an index version, promoting a configuration or swapping a program therefore
misses the cache by construction; nothing has to remember to invalidate it.

The label set is part of the key on purpose. Two callers in one tenant with different labels
must never share an entry, because the answer one is entitled to may cite evidence the other is
not.

The cache is an optimization, never a dependency: it sits behind a `CircuitBreaker`, and a Redis
that is slow or down degrades to answering directly (specification section 17).
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from rag_platform.evaluation import ProgramRevisions
from rag_platform.models import AccessContext, PipelineConfig, QueryResponse
from rag_platform.reliability import CircuitBreaker

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from redis import Redis


class AnswerCache(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None: ...


class QueryAnswerer(Protocol):
    @property
    def config(self) -> PipelineConfig: ...

    @property
    def programs(self) -> ProgramRevisions: ...

    def answer(self, question: str, access: AccessContext) -> QueryResponse: ...


@dataclass(slots=True)
class InMemoryAnswerCache:
    entries: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str | None:
        return self.entries.get(key)

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        self.entries[key] = value


@dataclass(slots=True)
class RedisAnswerCache:
    client: "Redis"
    prefix: str = "rag:answer:"

    def get(self, key: str) -> str | None:
        value = self.client.get(self.prefix + key)
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        self.client.set(self.prefix + key, value, ex=ttl_seconds)


def cache_key(
    question: str,
    access: AccessContext,
    *,
    config_version: str,
    program_revisions: tuple[str, ...],
    index_version_id: str | None,
) -> str:
    material = json.dumps(
        {
            "tenant": access.tenant_id,
            "labels": sorted(access.labels),
            "question": " ".join(question.split()).lower(),
            "config": config_version,
            "programs": list(program_revisions),
            "index": index_version_id,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CachedQueryService:
    """Wraps a query service; satisfies the same shape, so the API and evaluator can use it."""

    service: QueryAnswerer
    cache: AnswerCache
    active_index_version: Callable[[str], str | None]
    ttl_seconds: int = 300
    breaker: CircuitBreaker = field(default_factory=lambda: CircuitBreaker(name="answer_cache"))

    @property
    def config(self) -> PipelineConfig:
        return self.service.config

    @property
    def programs(self) -> ProgramRevisions:
        return self.service.programs

    def answer(self, question: str, access: AccessContext) -> QueryResponse:
        key = cache_key(
            question,
            access,
            config_version=self.service.config.version,
            program_revisions=self.service.programs.revisions(),
            index_version_id=self.active_index_version(access.tenant_id),
        )
        missing: str | None = None
        cached, read = self.breaker.call(lambda: self.cache.get(key), fallback=missing)
        if cached is not None:
            response = QueryResponse.model_validate_json(cached)
            response.trace.served_from_cache = True
            return response

        response = self.service.answer(question, access)
        if not read.ok:
            response.trace.degraded_dependencies.append(read.detail)
            return response
        # An answer produced while a dependency was degraded is not worth pinning for the TTL.
        if not response.trace.degraded_dependencies:
            payload = response.model_dump_json()
            _stored, write = self.breaker.call(
                lambda: self.cache.set(key, payload, ttl_seconds=self.ttl_seconds), fallback=None
            )
            if not write.ok:
                response.trace.degraded_dependencies.append(write.detail)
        return response

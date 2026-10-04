"""Durable source, index-version and lineage metadata in PostgreSQL (specification section 7).

`PostgresIndexCatalog` is an `IndexCatalog`: every consumer that takes the in-process catalog
takes this one, and the staging, validation, activation and rollback rules are inherited rather
than reimplemented. What it adds is write-through persistence — each lifecycle transition is
committed to PostgreSQL as it happens — and `load()`, which rebuilds the catalog from the
database at start-up. The metadata therefore survives a restart, and rollback still restores the
previous index version without reingestion (specification section 17).

Documents are stored as JSONB beside the few columns the lifecycle queries by, so a new field on
a model is not a schema migration.
"""

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_platform.adapters.payload import from_payload, to_payload
from rag_platform.catalog import IndexCatalog
from rag_platform.feedback import Feedback, FeedbackLog
from rag_platform.models import DocumentChunk, IndexVersion, SourceVersion

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from psycopg import Connection

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS rag_source_versions (
        source_version_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        body JSONB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rag_index_versions (
        index_version_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        status TEXT NOT NULL,
        body JSONB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rag_chunks (
        index_version_id TEXT NOT NULL REFERENCES rag_index_versions ON DELETE CASCADE,
        position INTEGER NOT NULL,
        chunk_id TEXT NOT NULL,
        body JSONB NOT NULL,
        PRIMARY KEY (index_version_id, position)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rag_activations (
        tenant_id TEXT PRIMARY KEY,
        active_index_version_id TEXT,
        history JSONB NOT NULL
    )
    """,
)

FEEDBACK_SCHEMA = """
    CREATE TABLE IF NOT EXISTS rag_feedback (
        feedback_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        status TEXT NOT NULL,
        body JSONB NOT NULL
    )
"""


class PostgresIndexCatalog(IndexCatalog):
    def __init__(
        self,
        connection: "Connection[Any]",
        *,
        sync_interval_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._connection = connection
        self._sync_interval = sync_interval_seconds
        self._clock = clock
        self._synced_at = clock()

    @property
    def connection(self) -> "Connection[Any]":
        return self._connection

    # Schema and hydration ------------------------------------------------------
    def ensure_schema(self) -> None:
        for statement in SCHEMA:
            self._connection.execute(statement)
        self._connection.commit()

    def load(self) -> None:
        """Rebuild the in-memory view from PostgreSQL."""
        for (body,) in self._connection.execute("SELECT body FROM rag_source_versions").fetchall():
            source_version = SourceVersion.model_validate(_document(body))
            self._source_versions[source_version.source_version_id] = source_version
        for (body,) in self._connection.execute("SELECT body FROM rag_index_versions").fetchall():
            index_version = IndexVersion.model_validate(_document(body))
            self._index_versions[index_version.index_version_id] = index_version
            self._chunks[index_version.index_version_id] = ()
        chunks: dict[str, list[DocumentChunk]] = {}
        for index_version_id, body in self._connection.execute(
            "SELECT index_version_id, body FROM rag_chunks ORDER BY index_version_id, position"
        ).fetchall():
            chunks.setdefault(index_version_id, []).append(from_payload(_document(body)))
        for index_version_id, items in chunks.items():
            self._chunks[index_version_id] = tuple(items)
        for tenant_id, active, history in self._connection.execute(
            "SELECT tenant_id, active_index_version_id, history FROM rag_activations"
        ).fetchall():
            if active is not None:
                self._active[tenant_id] = active
            self._activation_history[tenant_id] = list(_document(history))

        self._synced_at = self._clock()

    def active_index_version(self, tenant_id: str) -> IndexVersion | None:
        """The active version, after catching up with activations made by other replicas.

        Each replica holds its own in-memory view. Without this, an index version activated or
        rolled back through one replica would be invisible to the others until they restart.
        """
        if (
            self._sync_interval is not None
            and self._clock() - self._synced_at >= self._sync_interval
        ):
            self.sync()
        return super().active_index_version(tenant_id)

    def sync(self) -> bool:
        """Reload if another process moved an activation pointer. Returns whether it reloaded."""
        self._synced_at = self._clock()
        stored = {
            tenant_id: active
            for tenant_id, active in self._connection.execute(
                "SELECT tenant_id, active_index_version_id FROM rag_activations"
            ).fetchall()
            if active is not None
        }
        self._connection.commit()
        if stored == self._active:
            return False
        self._active.clear()
        self.load()
        return True

    # Write-through lifecycle ---------------------------------------------------
    def register_source_version(self, source_version: SourceVersion) -> SourceVersion:
        registered = super().register_source_version(source_version)
        self._connection.execute(
            """
            INSERT INTO rag_source_versions
                (source_version_id, source_id, tenant_id, content_hash, body)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (source_version_id) DO NOTHING
            """,
            (
                registered.source_version_id,
                registered.source_id,
                registered.tenant_id,
                registered.content_hash,
                registered.model_dump_json(),
            ),
        )
        self._connection.commit()
        return registered

    def stage_index_version(
        self, index_version: IndexVersion, chunks: Iterable[DocumentChunk]
    ) -> IndexVersion:
        staged = super().stage_index_version(index_version, chunks)
        self._save_index_version(staged)
        self._connection.execute(
            "DELETE FROM rag_chunks WHERE index_version_id = %s", (staged.index_version_id,)
        )
        for position, chunk in enumerate(self._chunks[staged.index_version_id]):
            self._connection.execute(
                """
                INSERT INTO rag_chunks (index_version_id, position, chunk_id, body)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (staged.index_version_id, position, chunk.chunk_id, json.dumps(to_payload(chunk))),
            )
        self._connection.commit()
        return staged

    def mark_validated(self, index_version_id: str) -> IndexVersion:
        validated = super().mark_validated(index_version_id)
        self._save_index_version(validated)
        self._connection.commit()
        return validated

    def retire(self, index_version_id: str) -> IndexVersion:
        retired = super().retire(index_version_id)
        self._save_index_version(retired)
        self._connection.commit()
        return retired

    def activate(self, index_version_id: str) -> IndexVersion:
        active = super().activate(index_version_id)
        self._save_index_version(active)
        self._save_activation(active.tenant_id)
        self._connection.commit()
        return active

    def rollback(self, tenant_id: str) -> IndexVersion:
        restored = super().rollback(tenant_id)
        self._save_index_version(restored)
        self._save_activation(tenant_id)
        self._connection.commit()
        return restored

    # Persistence helpers -------------------------------------------------------
    def _save_index_version(self, index_version: IndexVersion) -> None:
        self._connection.execute(
            """
            INSERT INTO rag_index_versions (index_version_id, tenant_id, status, body)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (index_version_id)
            DO UPDATE SET status = EXCLUDED.status, body = EXCLUDED.body
            """,
            (
                index_version.index_version_id,
                index_version.tenant_id,
                index_version.status,
                index_version.model_dump_json(),
            ),
        )

    def _save_activation(self, tenant_id: str) -> None:
        self._connection.execute(
            """
            INSERT INTO rag_activations (tenant_id, active_index_version_id, history)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (tenant_id)
            DO UPDATE SET active_index_version_id = EXCLUDED.active_index_version_id,
                          history = EXCLUDED.history
            """,
            (
                tenant_id,
                self._active.get(tenant_id),
                json.dumps(self._activation_history.get(tenant_id, [])),
            ),
        )


def _document(value: Any) -> Any:
    """psycopg decodes JSONB itself; a driver configured not to hands back the text."""
    return json.loads(value) if isinstance(value, str) else value


@dataclass
class PostgresFeedbackLog(FeedbackLog):
    """Feedback kept in PostgreSQL, so every replica sees and reviews the same items."""

    connection: "Connection[Any] | None" = None

    def ensure_schema(self) -> None:
        self._db().execute(FEEDBACK_SCHEMA)
        self._db().commit()

    def _db(self) -> "Connection[Any]":
        if self.connection is None:
            raise RuntimeError("PostgresFeedbackLog needs a connection")
        return self.connection

    def _save(self, feedback: Feedback) -> None:
        self._db().execute(
            """
            INSERT INTO rag_feedback (feedback_id, tenant_id, status, body)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (feedback_id)
            DO UPDATE SET status = EXCLUDED.status, body = EXCLUDED.body
            """,
            (
                feedback.feedback_id,
                feedback.tenant_id,
                feedback.status,
                feedback.model_dump_json(),
            ),
        )
        self._db().commit()

    def _find(self, feedback_id: str) -> Feedback | None:
        rows = (
            self._db()
            .execute("SELECT body FROM rag_feedback WHERE feedback_id = %s", (feedback_id,))
            .fetchall()
        )
        self._db().commit()
        return Feedback.model_validate(_document(rows[0][0])) if rows else None

    def _for_tenant(self, tenant_id: str) -> list[Feedback]:
        rows = (
            self._db()
            .execute(
                "SELECT body FROM rag_feedback WHERE tenant_id = %s ORDER BY feedback_id",
                (tenant_id,),
            )
            .fetchall()
        )
        self._db().commit()
        return [Feedback.model_validate(_document(body)) for (body,) in rows]

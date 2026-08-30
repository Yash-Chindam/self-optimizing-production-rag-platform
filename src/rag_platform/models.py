from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AccessContext(BaseModel):
    """Identity-derived retrieval scope.

    A trusted gateway must provide these values in production. They are not accepted in the
    request body so callers cannot widen scope through query input.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(min_length=1)
    labels: frozenset[str] = Field(default_factory=lambda: frozenset({"public"}))


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    tenant_id: str
    access_labels: frozenset[str]
    text: str
    source_uri: str
    source_title: str
    index_version: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2_000)


class Citation(BaseModel):
    citation_id: str
    source_title: str
    source_uri: str
    chunk_id: str


class AnswerTrace(BaseModel):
    config_version: str
    index_version: str | None
    retrieval_strategy: str
    retrieved_chunk_ids: list[str]
    policy: str


class QueryResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[Citation]
    trace: AnswerTrace


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    config_version: str


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str = "pipeline-v1"
    dense_top_k: int = Field(default=8, gt=0, le=100)
    sparse_top_k: int = Field(default=8, gt=0, le=100)
    final_top_k: int = Field(default=4, gt=0, le=20)
    reciprocal_rank_constant: int = Field(default=60, gt=0)
    context_character_budget: int = Field(default=4_000, ge=200)


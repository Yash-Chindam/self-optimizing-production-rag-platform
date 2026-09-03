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
    source_version_id: str = "unversioned"
    parent_chunk_id: str | None = None
    section_path: tuple[str, ...] = ()
    ordinal: int = 0
    retrievable: bool = True
    """Parent chunks are stored for context expansion but never scored directly."""


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
    context_chunk_ids: list[str] = []
    graph_expanded_chunk_ids: list[str] = []
    dropped_chunk_ids: list[str] = []
    context_characters: int = 0
    policy_notes: list[str] = []
    intent: str | None = None
    query_transformations: list[str] = []
    workflow_path: list[str] = []
    program_revisions: list[str] = []
    repair_attempts: int = 0
    unsupported_claims: list[str] = []
    degraded_dependencies: list[str] = []


class QueryResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "clarification_needed"]
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
    fusion_method: Literal["reciprocal_rank", "weighted"] = "reciprocal_rank"
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    graph_expansion_depth: int = Field(default=1, ge=0, le=4)
    graph_candidate_limit: int = Field(default=4, ge=0, le=50)
    graph_extractor_revision: str = "graph-extractor-v1"
    rerank_candidate_count: int = Field(default=12, ge=0, le=100)
    reranker_revision: str = "reranker-lexical-v1"
    max_chunks_per_source: int = Field(default=2, gt=0, le=20)
    near_duplicate_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    expand_parent_context: bool = True
    answer_sentence_limit: int = Field(default=2, gt=0, le=10)
    max_subqueries: int = Field(default=3, gt=0, le=10)
    max_repair_attempts: int = Field(default=1, ge=0, le=3)
    clarify_ambiguous_queries: bool = True
    program_suite_revision: str = "programs-v1"


FailureCategory = Literal[
    "ingestion",
    "chunking",
    "retrieval",
    "reranking",
    "generation",
    "citation",
    "authorization",
    "operational",
]
"""Specification section 13. Assigned to a case by a reviewer, or to a result by the evaluator."""


class EvaluationCase(BaseModel):
    """A reviewed evaluation example (specification section 14)."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    access_labels: frozenset[str] = Field(default_factory=lambda: frozenset({"public"}))
    required_evidence_chunk_ids: frozenset[str] = frozenset()
    forbidden_chunk_ids: frozenset[str] = frozenset()
    acceptable_answer_terms: frozenset[str] = frozenset()
    forbidden_claims: tuple[str, ...] = ()
    expected_status: Literal["answered", "insufficient_evidence", "clarification_needed"] = (
        "answered"
    )
    reviewer: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    failure_tags: tuple[FailureCategory, ...] = ()


class EvaluationSummary(BaseModel):
    """Aggregate metrics for one evaluation run (specification section 16)."""

    model_config = ConfigDict(frozen=True)

    case_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    grounded_rate: float = Field(ge=0.0, le=1.0)
    authorization_violations: int = Field(ge=0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.case_count if self.case_count else 0.0


class ConstraintViolation(BaseModel):
    """A promotion-blocking check the candidate failed (specification sections 12 and 17)."""

    model_config = ConfigDict(frozen=True)

    constraint: Literal["latency", "authorization", "quality_regression"]
    detail: str


CandidatePromotion = Literal[
    "rejected", "approved", "pareto_optimal", "canary", "promoted", "rolled_back"
]


class CandidateRun(BaseModel):
    """One evaluated pipeline configuration in the optimization control loop (section 14)."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    baseline_config_version: str
    config: PipelineConfig
    config_diff: tuple[str, ...]
    dataset_revision: str
    program_revisions: tuple[str, ...]
    summary: EvaluationSummary
    constraint_violations: tuple[ConstraintViolation, ...] = ()
    promotion: CandidatePromotion = "approved"

    def with_promotion(self, promotion: CandidatePromotion) -> "CandidateRun":
        return self.model_copy(update={"promotion": promotion})


class RetentionPolicy(BaseModel):
    """Retention rule recorded with every source version (specification section 7)."""

    model_config = ConfigDict(frozen=True)

    max_age_days: int | None = Field(default=None, gt=0)
    delete_original_after_indexing: bool = False


class PiiPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["off", "detect", "pseudonymize"] = "pseudonymize"
    recognizers: tuple[str, ...] = ("email", "phone", "national_id")


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal["structural", "token"] = "structural"
    max_characters: int = Field(default=600, ge=80, le=8_000)
    overlap_characters: int = Field(default=80, ge=0, le=2_000)
    parent_child: bool = True

    def model_post_init(self, _context: object) -> None:
        if self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters must be smaller than max_characters")


class SourceRegistration(BaseModel):
    """Approved source declaration. Registration precedes any acquisition."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    access_labels: frozenset[str] = Field(default_factory=lambda: frozenset({"public"}))
    source_type: Literal["markdown", "text"] = "markdown"
    source_uri: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    parser_revision: str = "parser-v1"
    retention: RetentionPolicy = RetentionPolicy()
    refresh: Literal["manual", "scheduled", "event"] = "manual"
    pii_policy: PiiPolicy = PiiPolicy()


class SourceVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_version_id: str
    source_id: str
    tenant_id: str
    owner: str
    access_labels: frozenset[str]
    source_uri: str
    source_title: str
    content_hash: str
    parser_revision: str
    transformation_revision: str
    retention: RetentionPolicy
    document_type: Literal["markdown", "text"]
    language: str


class IndexVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    index_version_id: str
    tenant_id: str
    source_version_ids: tuple[str, ...]
    chunking: ChunkingConfig
    embedding_revision: str
    analyzer_revision: str
    graph_extractor_revision: str
    physical_dense_index: str
    physical_sparse_index: str
    physical_graph_index: str
    chunk_count: int = Field(ge=0)
    status: Literal["building", "validated", "active", "retired"] = "building"

    def with_status(
        self, status: Literal["building", "validated", "active", "retired"]
    ) -> "IndexVersion":
        return self.model_copy(update={"status": status})


class IngestRequest(BaseModel):
    """Source registration submitted by a data steward. The tenant comes from the gateway."""

    source_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    access_labels: frozenset[str] = Field(default_factory=lambda: frozenset({"public"}))
    source_type: Literal["markdown", "text"] = "markdown"
    parser_revision: str = "parser-v1"
    retention: RetentionPolicy = RetentionPolicy()
    refresh: Literal["manual", "scheduled", "event"] = "manual"
    pii_policy: PiiPolicy = PiiPolicy()


class IngestionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_version_id: str
    index_version_id: str
    chunk_count: int
    duplicate_chunks_removed: int
    pii_entities_processed: int
    reused_existing_source_version: bool
    validation_checks: tuple[str, ...]


class IngestResponse(BaseModel):
    report: IngestionReport
    index_version: IndexVersion

# Self-Optimizing Production RAG Platform

**Document type:** Technical design specification  
**Purpose:** Define an evaluation-driven, privacy-aware RAG optimization platform  
**Implementation plan:** Intentionally excluded

> Proposed quality and performance values are design targets. Use them on a resume only after measuring a working implementation.

## 1. Use case

RAG systems degrade as documents, schemas, terminology, embeddings, models and user questions change. Manual prompt tuning cannot reliably identify whether failures originate in parsing, chunking, retrieval, reranking or generation.

This platform treats a RAG pipeline as a versioned and measurable system. It collects reviewed failures, evaluates bounded candidate configurations, uses DSPy to optimize declared language-program components and promotes changes only when quality improves without violating latency, cost, privacy or access-control constraints.

The system is self-optimizing, not self-authorizing. Humans define evaluation data, candidate boundaries and production-promotion policy.

### Representative scenarios

1. Enterprise policy assistant with tenant and department permissions.
2. Technical-support system with frequently changing documentation.
3. Regulatory assistant requiring citations and effective-date awareness.
4. Analytics assistant combining text definitions, graph relationships and metadata.

## 2. Portfolio value

The project demonstrates:

- Dense, sparse, hybrid and graph retrieval.
- DSPy program optimization.
- LangGraph recovery, critique and fallback workflows.
- RAGAS and DeepEval evaluation.
- Versioned datasets, indexes and configurations.
- CI evaluation gates.
- Canary releases and rollback.
- Tenant-aware retrieval and local PII processing.
- Retrieval and generation observability.

## 3. Users and actors

| Actor | Responsibility |
|---|---|
| Knowledge user | Asks questions and reviews citations and confidence. |
| Knowledge administrator | Registers sources, owners, retention and access labels. |
| AI engineer | Defines retrieval components, metrics and candidate space. |
| Reviewer | Validates evaluation cases and release decisions. |
| Optimization service | Evaluates candidates outside production. |

## 4. Scope

### In scope

- Versioned ingestion.
- Dense, sparse, hybrid and graph retrieval.
- Query rewriting and decomposition.
- Reranking and parent-child expansion.
- DSPy optimization.
- RAG evaluation and failure clustering.
- CI quality gates.
- Canary promotion and rollback.
- Tenant-aware retrieval.
- Local PII detection and pseudonymization.
- End-to-end observability.

### Out of scope

- Unbounded production self-modification.
- Automatic embedding training on unlabeled production data.
- Synthetic data as the only evaluation source.
- Treating an LLM judge as ground truth.
- Using a retrieval database as the authorization boundary.
- Claiming FHE or TEE without a working deployment and benchmark.

## 5. Architecture

```text
Document sources
       |
       v
Prefect + LlamaIndex ingestion
       |
       +---- originals -> MinIO
       +---- metadata -> PostgreSQL
       +---- dense vectors -> Qdrant
       +---- lexical index -> OpenSearch
       +---- entities/provenance -> Neo4j
       |
       v
Versioned active index
       |
       v
LangGraph query workflow
classify -> rewrite -> retrieve -> fuse -> rerank
         -> generate -> verify -> answer/fallback
       |
       v
Phoenix/OpenInference traces + reviewed feedback
       |
       v
Offline evaluation + DSPy optimization
       |
       v
MLflow registry -> CI gate -> canary -> promote/rollback
```

## 6. Technology selection

| Technology | Responsibility | Selection rationale |
|---|---|---|
| DSPy | Language-program optimization | Compiles typed modules against explicit examples and metrics. |
| LangGraph | Query workflow | Makes critique, repair, fallback and clarification states explicit. |
| LlamaIndex | Ingestion composition | Provides parsers, transformations and storage integrations. |
| Qdrant | Dense retrieval | Supports vector search and mandatory metadata filters. |
| OpenSearch | Sparse retrieval | Provides BM25, analyzers and lexical filtering. |
| Neo4j | GraphRAG and provenance | Represents entity, relationship, lineage and evidence graphs. |
| RAGAS | RAG-oriented evaluation | Measures retrieval and grounding-oriented qualities. |
| DeepEval | Regression evaluation | Supports customizable component and end-to-end tests. |
| MLflow | Experiment governance | Tracks candidates, metrics and promotion evidence. |
| DVC | Dataset versioning | Connects evaluation results to reproducible data revisions. |
| Phoenix/OpenInference | RAG tracing | Captures retrieval and model spans. |
| Presidio | PII processing | Runs configurable PII recognition inside the trusted boundary. |

## 7. Ingestion design

### Source registration

Every source records:

- Tenant and owner.
- Access classification.
- Source type.
- Parser configuration.
- Retention rule.
- Refresh behavior.
- PII policy.

### Ingestion flow

```text
Acquire approved source
    -> store immutable original
    -> calculate content hash
    -> extract text and structure
    -> detect document type and language
    -> detect and process sensitive data
    -> generate chunks and metadata
    -> extract graph entities when justified
    -> build dense, lexical and graph indexes
    -> validate counts, permissions and retrieval
    -> activate IndexVersion
```

### Ingestion stores

| Store | Responsibility |
|---|---|
| MinIO | Immutable originals and derived artifacts |
| PostgreSQL | Source, chunk, lineage and configuration metadata |
| Qdrant | Dense vectors and filterable access metadata |
| OpenSearch | Lexical indexes and analyzers |
| Neo4j | Entities, relationships and provenance |
| Optional Kafka | Source-change, ingestion and evaluation events |

## 8. Retrieval design

### Dense retrieval

Use Qdrant for semantic similarity and mandatory tenant/access filters.

### Sparse retrieval

Use OpenSearch BM25 for exact identifiers, acronyms, terminology and rare vocabulary.

### Hybrid fusion

Combine dense and sparse rankings using a versioned method such as reciprocal-rank fusion or normalized weighted fusion. Treat fusion weights as evaluated configuration.

### Graph retrieval

Use Neo4j when relationships add information:

- Organizational structures.
- Product dependencies.
- Schema and data lineage.
- Regulatory relationships.
- Claim-to-evidence provenance.

Graph retrieval complements rather than automatically replaces vector and lexical retrieval.

### Reranking

Use a cross-encoder or approved reranker on a bounded candidate set. Version the reranker independently.

### Context construction

- Deduplicate overlapping chunks.
- Expand parent context only when needed.
- Enforce a context budget.
- Preserve source and access metadata.
- Encourage evidence diversity.
- Attach citation IDs before generation.

## 9. Query workflow

```text
Authenticate and resolve tenant
    -> classify query
    -> apply access filters
    -> rewrite or decompose if required
    -> execute selected retrieval strategies
    -> fuse and rerank
    -> construct bounded context
    -> generate structured answer
    -> verify grounding and citations
    -> answer, clarify or fall back
```

LangGraph owns explicit workflow state. It should model critic, repair, clarification and fallback paths rather than wrapping one linear chain.

## 10. DSPy optimization design

Candidate DSPy modules:

- Query classification.
- Query rewriting.
- Query decomposition.
- Answer synthesis.
- Claim verification.
- Clarification generation.

Every module requires:

- Typed signatures.
- Curated examples.
- Explicit metrics.
- Training and held-out evaluation splits.
- Versioned compiled artifacts.

DSPy does not choose access policy, database indexes or production promotion.

## 11. Candidate configuration space

- Chunk size and overlap.
- Structural versus token chunking.
- Parent-child retrieval.
- Embedding model.
- Dense and sparse `top_k`.
- Fusion method and weights.
- Query rewrite/decomposition strategy.
- Graph expansion depth.
- Reranker and candidate count.
- Context budget.
- DSPy program revision.
- Generator model and parameters.

The candidate space is bounded before each optimization run.

## 12. Optimization control loop

```text
Reviewed production failures
    -> classify and cluster failures
    -> update versioned evaluation set
    -> freeze the production baseline
    -> generate bounded candidates
    -> run component and end-to-end evaluation
    -> reject privacy, access, latency and cost violations
    -> register Pareto-optimal candidates
    -> canary one approved candidate
    -> promote or roll back
```

## 13. Failure taxonomy

| Category | Example |
|---|---|
| Ingestion | Parser omitted a table or heading relationship. |
| Chunking | Evidence was divided into unusable fragments. |
| Retrieval | Relevant content never entered the candidate set. |
| Reranking | Correct evidence ranked below distractors. |
| Generation | Correct context was present but answer was unsupported. |
| Citation | Citation did not entail the claim. |
| Authorization | Correct content was withheld or unauthorized content appeared. |
| Operational | Quality passed but latency or cost violated constraints. |

## 14. Information model

### SourceVersion

- Source and owner.
- Tenant and access class.
- Content hash.
- Parser and transformation revisions.
- Retention policy.

### IndexVersion

- Source-version set.
- Chunking configuration.
- Embedding revision.
- Lexical analyzer revision.
- Graph extractor revision.
- Physical index identifiers.

### PipelineConfig

- Classifier and rewrite program.
- Retrieval strategies.
- Fusion and reranker.
- Context budget.
- Generator and critic revisions.

### EvaluationCase

- Question and access context.
- Required evidence.
- Acceptable answer properties.
- Forbidden claims.
- Reviewer provenance.
- Difficulty and failure tags.

### CandidateRun

- Baseline and configuration difference.
- Dataset, code and model revisions.
- Metrics and constraint results.
- Promotion disposition.

### AnswerTrace

- Query transformations.
- Retrieved IDs and ranks.
- Filters and policy decisions.
- Final context and citations.
- Model route, latency and token counts.
- User/reviewer feedback.

## 15. Privacy and authorization

- Run Presidio inside the trusted boundary.
- Apply tenant and access labels during ingestion.
- Enforce mandatory filters before retrieval.
- Recheck authorization before context assembly.
- Store pseudonym mappings separately from indexes.
- Permit rehydration only in approved workflows.
- Redact or sample traces to prevent observability leakage.
- Test cross-tenant and permission-removal cases.

## 16. Evaluation

### Ingestion

- Parse success.
- Structural preservation.
- Deduplication.
- Metadata accuracy.
- PII correctness.

### Retrieval

- Recall@k and precision@k.
- Mean reciprocal rank.
- nDCG.
- Hard-negative rejection.
- Authorization correctness.

### Generation

- Faithfulness.
- Answer correctness.
- Citation entailment.
- Answer relevance.
- Structured-output validity.
- Abstention/clarification correctness.

### System

- P50/P95 latency.
- Cost per query.
- Failure and fallback rates.
- Cache effectiveness.
- Canary regressions.
- Rollback success.

Use RAGAS, DeepEval, deterministic checks and human-reviewed evidence. Do not release solely from one LLM-judge score.

## 17. Reliability design

- Immutable source and index versions.
- Idempotent ingestion.
- Staged index activation.
- Timeouts for each dependency.
- Declared degradation when graph or reranking services fail.
- Circuit breakers around embedding, reranking and model calls.
- Canary release by tenant or traffic percentage.
- One-action rollback to the last approved configuration.
- Complete evaluation and configuration provenance.

### Proposed design targets

- Every release reproducible from dataset, code, model and index versions.
- Zero regression in authorization correctness.
- Every answer provides citations or clearly reports insufficient evidence.
- Previous pipeline restored without full reingestion during rollback.
- Candidate promotion blocked by privacy, latency or cost violations.

## 18. Deployment topology

- FastAPI query service.
- LangGraph workers.
- Prefect server and workers.
- Qdrant, OpenSearch and Neo4j.
- PostgreSQL, Redis and MinIO.
- Optional Kafka.
- MLflow and DVC-compatible storage.
- Phoenix/OpenInference.
- OpenTelemetry Collector and shared observability stack.
- Docker, Kubernetes, Helm and GitHub Actions.

## 19. Official references

- [DSPy documentation](https://dspy.ai/)
- [LlamaIndex documentation](https://docs.llamaindex.ai/)
- [RAGAS metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [OpenSearch documentation](https://docs.opensearch.org/)
- [Neo4j documentation](https://neo4j.com/docs/)
- [Microsoft Presidio documentation](https://microsoft.github.io/presidio/)


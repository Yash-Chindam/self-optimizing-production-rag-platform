# Self-Optimizing Production RAG Platform

This repository is the executable implementation of the architecture in
[`03-self-optimizing-production-rag-platform.md`](03-self-optimizing-production-rag-platform.md).
It provides versioned ingestion, tenant-aware hybrid retrieval, citation-or-abstention
behavior, and a small browser client.

By default the stores and models are deterministic in-process implementations that sit behind
the same boundaries the production adapters use: `IndexCatalog` stands in for PostgreSQL plus
the Qdrant, OpenSearch and Neo4j indexes, and `PiiProcessor` stands in for a Presidio analyzer.
Swapping an adapter does not change the ingestion contract, the authorization filters or the
index-version lifecycle.

Real adapters for those stores live in `rag_platform.adapters` and install through the `stores`
extra — see [Production stores](#production-stores).

## Run locally

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m rag_platform
```

Open <http://127.0.0.1:8000>. The demo UI sends the `tenant-acme` tenant and `employees`
access label. API callers must send `X-Tenant-ID`; `X-Access-Labels` is an optional
comma-separated list that defaults to `public`. In production, a trusted identity gateway must
derive and overwrite these headers.

The demo corpus is ingested through the real pipeline on start-up, so every run exercises
source versioning, chunking, sensitive-data processing and staged index activation.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/query` | Answer a question from authorized evidence, or abstain. |
| `POST /v1/sources` | Register and ingest a source. Requires the `data-steward` label. |
| `GET /v1/index-versions/active` | Report the tenant's active index version. |
| `POST /v1/index-versions/rollback` | Restore the previous index version without reingestion. |
| `POST /v1/feedback` | Record feedback on an answer, tied to its trace. |
| `GET /v1/feedback` | List a tenant's feedback. Requires the `data-steward` label. |
| `POST /v1/feedback/{id}/review` | Accept or dismiss feedback as a named reviewer. |
| `GET /healthz` | Report the active pipeline configuration version. |

## Ingestion

`IngestionPipeline.ingest` follows specification section 7: hash the content, reuse the
existing source version when the hash is unchanged, detect document type and language,
pseudonymize sensitive values into a vault held apart from the indexes, chunk structurally with
parent context, stage a new index version, validate it, and only then activate it. Validation
rejects cross-tenant chunks, unlabelled chunks and indexes that fail a retrieval smoke check;
a rejected index version is retired rather than activated.

## Retrieval

A query runs dense and lexical search under mandatory tenant and access filters, fuses the two
rankings with a versioned method (reciprocal-rank or normalized weighted fusion), expands the
candidate set along entity relationships in the knowledge graph, reranks a bounded candidate
set, and then constructs the context: authorization is rechecked, near-duplicates are dropped,
evidence is kept diverse across sources, parent sections replace fragments only when they fit,
and the character budget is enforced before citation identifiers are attached.

Fusion weights, graph depth, reranker revision, diversity and budget are all `PipelineConfig`
fields rather than code, so each is a candidate configuration that evaluation can accept or
reject. Every answer carries the retrieval strategy, the retrieved and context chunk
identifiers, the graph-expanded identifiers and the policy decisions that dropped evidence.

## Query workflow

`QueryWorkflow` follows specification section 9 as an explicit state machine rather than a
linear chain: classify, clarify, rewrite, decompose, retrieve, build context, generate, verify,
repair and fallback are separate states, each recorded on the answer's `workflow_path`. An
ambiguous question is clarified instead of answered; a grounded-but-unverified answer is
repaired by narrowing to its single best-supported sentence before it is re-verified, and only
falls back after the configured repair budget is spent. This is the structure LangGraph owns in
the deployed topology — the executor can be swapped without changing the states or their tests.

The states call a `ProgramSuite` (specification section 10): typed classifier, rewriter,
decomposer, synthesizer, verifier and clarification programs, each with an independent
revision. The default implementations are deterministic — the synthesizer is extractive, so
every claim it produces is a sentence copied from a chunk it cites — and a DSPy-compiled module
can replace an implementation without changing the signature the workflow depends on or the
`program_revisions` an evaluation run records.

## Evaluation and optimization

`Evaluator` (specification section 16) runs a versioned `EvaluationCase` set against a query
service and checks each answer deterministically rather than by LLM judgment: retrieval recall
and reciprocal rank against the required evidence, grounding and forbidden claims against the
answer text, and the expected outcome against the actual status. A failed case is tagged with
the taxonomy a reviewer would assign by hand (section 13) — an unauthorized chunk in context is
always `authorization`; a wrong outcome or missing evidence is `retrieval`; an ungrounded or
forbidden claim is `citation`; a correct, grounded answer missing an expected term is
`generation`.

`OptimizationRun` runs the control loop in section 12 over the bounded candidate space in
section 11: `CandidateSpace` perturbs one `PipelineConfig` field at a time within
reviewer-approved bounds, `Constraints` rejects any candidate that lets an unauthorized chunk
into context or regresses latency or quality against the frozen baseline, and the
non-dominated survivors among what remains are marked Pareto-optimal. `canary` performs the
loop's last two steps: one approved candidate is evaluated again and promoted only if it still
clears every constraint, otherwise it is rolled back. `PipelineConfigRegistry` mirrors
`IndexCatalog`'s staged-activation pattern for `PipelineConfig`, so promotion and one-action
rollback apply to the production configuration the same way they apply to an index version
(section 17).

Optimization runs outside production (section 3, "Optimization service"): nothing in this
module changes which configuration answers a live query, so it is exercised as a library
against its own test corpus rather than through the HTTP API.

## Production stores

Adapters for Qdrant, OpenSearch, Neo4j, PostgreSQL, MinIO and Redis implement the protocols the
platform already depends on, so running on the real stores changes no retrieval, workflow or
authorization code (specification sections 6, 7, 8 and 18):

```bash
python -m pip install -e ".[dev,stores]"
docker compose up -d --wait
set -a && . deploy/local.env && set +a
pytest tests/integration/test_stores.py tests/integration/test_services.py --no-cov
```

| Adapter | Protocol or base it satisfies | Role |
|---|---|---|
| `QdrantDenseIndex` | `DenseIndex` | Vector search, one collection per `IndexVersion` |
| `OpenSearchSparseIndex` | `SparseIndex` | BM25, one index per `IndexVersion` |
| `CompositeChunkRepository` | `ChunkRepository` | Presents both halves to `HybridRetriever` |
| `Neo4jGraphRetriever` | `GraphExpander` | Entity and provenance graph, one subgraph per `IndexVersion` |
| `QdrantMirror` / `OpenSearchMirror` / `Neo4jGraphMirror` | `IndexMirror` | Ingestion's "build the indexes" step |
| `PostgresIndexCatalog` | `IndexCatalog` | Durable source, index-version and lineage metadata |
| `MinioOriginalStore` | `OriginalStore` | Immutable, content-addressed originals |
| `RedisAnswerCache` + `CachedQueryService` | `AnswerCache` | Answer cache keyed by everything that produced the answer |
| `HashingEmbedder` / `SentenceTransformerEmbedder` | `TextEmbedder` | Versioned embeddings |

Two properties are worth calling out because they are what the tests are about.

**The database is never the authorization boundary.** Specification section 4 puts that out of
scope deliberately, so every adapter sends the tenant filter to the store *and* re-applies the
authoritative `is_authorized` check in Python before a chunk leaves the adapter. A store that is
stale, misconfigured or simply wrong cannot widen access on its own. OpenSearch can express the
label subset test exactly (`terms_set` against `required_label_count`); Qdrant cannot, so there
the store filter only narrows and the Python check decides. The integration tests assert the
outcome is identical either way, including that a chunk labelled `finance` *and* `management` is
withheld from a caller holding only `finance`.

**Index versions stay immutable.** Each adapter addresses the store by the `physical_dense_index`
/ `physical_sparse_index` identifier its `IndexVersion` already records, and that name is derived
from the version fingerprint. A new index version is therefore written into its own collection
and only then activated, so activation and rollback switch between whole indexes instead of
mutating the one serving traffic. If a store refuses the write, the index version is retired and
the previous one keeps answering.

**The graph means the same thing in either store.** `Neo4jGraphRetriever` and the in-process
`GraphRetriever` share one extractor, and an integration test asserts they return the same
expansion for the same seeds at every depth — including the case a Cypher path cannot express on
its own, walking out and back to the seed entity.

**The catalog is durable, not different.** `PostgresIndexCatalog` inherits the staging,
validation, activation and rollback rules and adds write-through persistence, so the metadata
survives a restart and rollback still needs no reingestion.

**A cached answer is only reused when nothing that produced it changed.** The cache key covers
the tenant, the caller's exact label set, the question, the configuration version, the program
revisions and the active index version, so activation, rollback or promotion miss the cache by
construction, and two callers with different labels never share an entry. Redis sits behind a
`CircuitBreaker`: if it is down, queries are answered directly.

Embeddings are versioned because the revision is part of an `IndexVersion`'s fingerprint.
`HashingEmbedder` is the default: a genuine vector embedding (feature hashing over token
unigrams and bigrams, L2-normalized) that needs no model download, so the stores can be populated
and queried offline. It captures lexical and co-occurrence similarity, not learned semantics —
install the `embeddings` extra and use `SentenceTransformerEmbedder` for that, behind the same
protocol.

## Language programs and sensitive-data recognition

Two more stand-ins have real implementations behind their protocols.

**DSPy programs** (`rag_platform.adapters.dspy_programs`, the `llm` extra) implement the six
candidate modules in specification section 10 — classification, rewriting, decomposition,
synthesis, claim verification and clarification — with everything that section requires of
each: a typed signature, curated examples, an explicit metric, a training and a held-out split,
and a versioned compiled artifact.

```python
import dspy
from pathlib import Path
from rag_platform.adapters import dspy_examples
from rag_platform.adapters.dspy_programs import (
    ClassifyQuery, build_program_suite, compile_program, intent_metric,
)

lm = dspy.LM("<any model identifier dspy.LM accepts>")
compiled = compile_program(
    "classifier", ClassifyQuery, dspy_examples.classifier_examples(), intent_metric, lm
)
compiled.save(Path("artifacts/programs"))        # records revision and held-out score
programs = build_program_suite(lm, artifacts=Path("artifacts/programs"))
```

A program's revision is a hash of its compiled state, so the `program_revisions` on an
`AnswerTrace` identify exactly the prompt and demonstrations that produced an answer. The
adapters keep the guarantees the deterministic programs gave for free: a citation the model
invents is discarded, an answer with no surviving citation is not an answer, and the verifier
only sees the evidence the answer cites. Classification, rewriting, decomposition and
clarification fall back to the deterministic program if the model call fails; synthesis and
verification raise instead, and the workflow's circuit breakers turn that into abstention.
The suite is tested offline against DSPy's `DummyLM`, including the full query workflow.

**Presidio** (`rag_platform.adapters.presidio`, the `privacy` extra) is a `PiiRecognizer`, so it
replaces the regex recognizers inside `PiiProcessor` without touching pseudonymization, the
vault or rehydration. It runs in-process and finds what no pattern can — a person's name:

```python
from rag_platform.adapters.presidio import PresidioRecognizerFactory, build_analyzer
from rag_platform.pii import PiiProcessor

processor = PiiProcessor(recognizer_factory=PresidioRecognizerFactory(analyzer=build_analyzer()))
```

A `PiiPolicy` opts into a kind by naming it (`recognizers=("email", "phone", "person")`); a kind
that is not named is never analyzed.

## Observability and experiment governance

Install with `python -m pip install -e ".[observability,governance]"`.

**Tracing.** `QueryWorkflow` reports to a `WorkflowObserver`; the core has no telemetry
dependency. `rag_platform.adapters.tracing.OpenTelemetryObserver` turns each query into one
trace in OpenInference conventions: a root `rag.query` span and one child span per workflow
state (`RETRIEVER` for retrieval, `LLM` for generation, `GUARDRAIL` for verification). Spans are
exported over OTLP to the OpenTelemetry Collector (`deploy/otel-collector.yaml`), which forwards
them to Phoenix, so the backend is a deployment decision.

Traces leave the trusted boundary, so the adapter is narrow about what it exports
(specification section 15):

- chunk text is never exported; retrieval spans carry chunk ids, scores and versions;
- the question and answer are exported only if `TracePolicy.capture_text` allows it, after
  identifiers are redacted and the text is truncated;
- a degraded dependency is named, but its error message is not exported;
- sampling is decided at the root span (`sample_ratio`), so a query is traced whole or not at
  all, and an unsampled query has no `trace_id`.

`AnswerTrace` carries the `trace_id`, measured `latency_ms`, the `model_route` that generated
the answer and whitespace-token estimates (`prompt_tokens`, `completion_tokens`).

**Feedback.** `POST /v1/feedback` records a rating against a trace. Free text is redacted on
the way in. Feedback becomes evaluation material only after a steward reviews it;
`Feedback.to_case` drafts an `EvaluationCase` from accepted feedback, with the reviewer
supplying the expected evidence.

**Versioned evaluation set.** `data/evaluation/cases.jsonl` is the reviewed evaluation set for
the demo corpus. Its revision is the MD5 of the file, which is the hash DVC records in
`dvc.lock`, so the revision an evaluation or an MLflow run reports is the one DVC tracks.

```bash
python -m rag_platform evaluate     # run the set, write reports/, exit 1 if the gate fails
dvc repro                           # the same, as the DVC stage in dvc.yaml
dvc metrics show
python -m rag_platform optimize --mlflow-tracking-uri http://127.0.0.1:5000
```

`evaluate` applies `ReleaseGate`: every case passes, every answer is grounded and no
unauthorized chunk reaches context. The reports DVC tracks contain no timing, so they are
byte-identical for identical data and code. CI reruns the stage and fails if the gate fails or
if `dvc.lock` no longer matches what the platform answers. Evaluation outputs are pushed to an
S3-compatible DVC remote (MinIO in the local stack; credentials come from the standard AWS
environment variables).

**MLflow.** `MlflowRunLogger` records every evaluation and every optimization candidate,
including rejected ones, with its configuration, metrics, dataset revision, program revisions,
constraint results and promotion disposition. `MlflowConfigRegistry` persists
`PipelineConfigRegistry` semantics in the MLflow model registry: each configuration is a model
version, the `champion` alias is the active one, and promotion and rollback move the alias, so a
restarted process reads the active configuration back.

`docker compose up -d phoenix otel-collector mlflow` starts the three services; the `Store
adapters` CI job runs `tests/integration/test_observability.py` against them and round-trips the
DVC outputs through MinIO.

## Reliability

Graph expansion, reranking, answer synthesis and claim verification are all optional
enrichments over an already-authorized candidate set, not the retrieval or authorization
boundary itself (specification section 17). Each is called behind a `CircuitBreaker`: a failing
or slow dependency degrades to a declared fallback — no graph expansion, the unranked
candidates, an empty synthesis, or (for verification) "not grounded" — rather than crashing the
query. A verifier that cannot run is never treated as having confirmed grounding, so a failure
there still routes through the same repair-then-fallback path an ungrounded answer does. Every
degradation is recorded on `AnswerTrace.degraded_dependencies`. After enough consecutive
failures the breaker opens and fails fast for a cooldown period instead of retrying a dependency
that is down on every query; one trial call after the cooldown decides whether it closes again.

## Test

```bash
ruff check .
mypy src
pytest --cov --cov-report=term-missing
npm ci
npx playwright install chromium
npm run test:e2e
```

Unit tests cover policy and retrieval behavior, integration tests exercise the HTTP boundary,
and Playwright tests verify the user journey in a real browser. GitHub Actions runs all three
layers and verifies the production container build.

## Delivery policy

All changes are made on a branch and delivered by pull request. The CI workflow is the required
quality gate. Successful repository-owner and Dependabot pull requests are merged automatically
after CI while preserving their individual commits. Repository branch protection should require
the `CI / Python quality and tests`, `CI / Playwright end-to-end`, and `CI / Container build`
checks.


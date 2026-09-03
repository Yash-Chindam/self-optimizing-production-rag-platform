# Self-Optimizing Production RAG Platform

This repository is the executable implementation of the architecture in
[`03-self-optimizing-production-rag-platform.md`](03-self-optimizing-production-rag-platform.md).
It provides versioned ingestion, tenant-aware hybrid retrieval, citation-or-abstention
behavior, and a small browser client.

The stores and models are deterministic in-process implementations that sit behind the same
boundaries the production adapters use: `IndexCatalog` stands in for PostgreSQL plus the
Qdrant, OpenSearch and Neo4j indexes, and `PiiProcessor` stands in for a Presidio analyzer.
Swapping an adapter does not change the ingestion contract, the authorization filters or the
index-version lifecycle.

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


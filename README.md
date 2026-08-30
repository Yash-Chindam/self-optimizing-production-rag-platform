# Self-Optimizing Production RAG Platform

This repository is the executable foundation for the architecture in
[`03-self-optimizing-production-rag-platform.md`](03-self-optimizing-production-rag-platform.md).
The first vertical slice provides a tenant-aware query API, deterministic hybrid retrieval,
citation-or-abstention behavior, and a small browser client. Production storage, model, and
optimization adapters will be added behind the same service boundaries.

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


# Contributing

## Pull-request workflow

1. Branch from `main`; use the `codex/` prefix for Codex-authored branches.
2. Add or update unit, integration, and Playwright coverage for behavior changes.
3. Open a pull request rather than pushing directly to `main`.
4. Wait for both CI jobs to pass. Successful repository-owner and Dependabot pull requests are
   squash-merged by the post-CI workflow.

Repository administrators should protect `main`, require pull requests, and require these checks:

- `CI / Python quality and tests`
- `CI / Playwright end-to-end`

The initial bootstrap pull request must be merged manually because GitHub does not execute a new
default-branch `workflow_run` auto-merge workflow until that workflow already exists on `main`.


# AGENTS.md — Workflow Rules for AI Agents

This file defines the branching and release workflow for this repository.
AI agents MUST follow these rules when making changes. Humans may override them.

## Branch Strategy

- **`main`** — Release branch. Only receives commits via `dev` → `main` merge. Never pushed to directly.
- **`dev`** — Integration branch. Acts as a pseudo-main branch. All feature work, fixes, and hotfixes land here first. Never deleted.
- **Feature branches** — Branch from `dev`, merge into `dev` via PR. Delete after merge. Never go directly into `main`.

## Workflow

```
feature branch → PR into dev → accumulate features → PR dev → main → merge (--no-ff) → tag → release
```

### Rules

1. **Everything lands on `dev` first.** Hotfixes, features, doc fixes, version bumps — all go through `dev`. Nothing bypasses `dev` into `main`.

2. **Feature branches** branch from `dev` and merge back into `dev` via PR. They can be deleted after merge. They must never target `main` directly.

3. **`dev` is pseudo-main.** Each PR into `dev` should be thoroughly reviewed and tested. `dev` should be kept in a shippable state at all times.

4. **Version is set on `dev`.** The version bump commit lives on `dev` and is carried into `main` via the merge. Do not use intermediate release branches.

## Release Checklist

When preparing a release PR from `dev` → `main`, the PR must include all of the following:

### Version bump (3 files)

- [ ] `pyproject.toml` — update `version = "..."`
- [ ] `src/seu_injection/version.py` — update `FALLBACK_VERSION = "..."`
- [ ] `CHANGELOG.md` — add new release section with date

### Documentation version references (3 files)

- [ ] `README.md` — update citation block `version = {...}`
- [ ] `examples/README.md` — update citation block `version = {...}`
- [ ] `docs/source/installation.md` — update `Latest stable release (vX.Y.Z)`

### Release PR requirements

- [ ] CI passes on the PR
- [ ] Merge with `--no-ff` (merge commit, not squash or rebase)
- [ ] Tag the merge commit: `git tag vX.Y.Z && git push origin vX.Y.Z`

Tag push triggers `.github/workflows/release.yml` (lint → type check → test → build → verify → publish).

## Anti-Patterns (past mistakes to avoid)

- ❌ Creating a `release/v*` branch from `main` — bypasses `dev`, breaks the workflow
- ❌ Squash-merging `dev` → `main` — loses individual commit history, use `--no-ff` instead
- ❌ Direct pushes to `main` — blocked by branch protection anyway
- ❌ Deleting `dev` — it is persistent and never deleted

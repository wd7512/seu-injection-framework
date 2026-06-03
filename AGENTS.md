# AGENTS.md — Workflow Rules for AI Agents

This file defines the branching and release workflow for this repository.
AI agents MUST follow these rules when making changes. Humans may override them.

## Branch Strategy

- **`main`** — Release branch. Only receives commits via `dev` → `main` merge. Never pushed to directly.
- **`dev`** — Integration branch. Acts as a pseudo-main branch. All feature work, fixes, and hotfixes land here first. Never deleted.
- **Feature branches** — Branch from `dev`, merge into `dev`. Delete after merge. Never go directly into `main`.

## Workflow

```
feature branch → merge to dev (with PR) → accumulate features → PR from dev→main → tag → release
```

### Rules

1. **Everything lands on `dev` first.** Hotfixes, features, doc fixes, version bumps — all go through `dev`. Nothing bypasses `dev` into `main`.

2. **Feature branches** branch from `dev` and merge back into `dev` via PR. They can be deleted after merge. They must never target `main` directly.

3. **`dev` is pseudo-main.** Each PR into `dev` should be thoroughly reviewed and tested. `dev` should be kept in a shippable state at all times.

4. **Release process:**
   - When features on `dev` are ready for release, open a PR from `dev` → `main`
   - The PR must include a version bump (pyproject.toml + version.py + CHANGELOG.md)
   - CI must pass on the PR before merge
   - Merge with `--no-ff` to preserve history
   - Tag the merge commit and push to trigger the release workflow

5. **Version is set on `dev`.** The version bump commit lives on `dev` and is carried into `main` via the merge. Do not use intermediate release branches.

## Anti-Patterns (past mistakes to avoid)

- ❌ Creating a `release/v*` branch from `main` — bypasses `dev`, breaks the workflow
- ❌ Squash-merging `dev` → `main` — loses individual commit history, use `--no-ff` instead
- ❌ Direct pushes to `main` — blocked by branch protection anyway
- ❌ Deleting `dev` — it is persistent and never deleted

## Release Pipeline

Triggered by pushing a tag matching `v*.*.*` to GitHub. See `.github/workflows/release.yml` for details.

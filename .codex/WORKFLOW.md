# Codex workflow for soynlp

This file adapts `.claude/rules/branch.md` and `.claude/rules/commit.md` for Codex. The repository root `AGENTS.md` is Codex's automatically discovered entry point; this file holds the detailed workflow.

## Development environment

- Start Codex in the worktree containing the branch to edit. Verify the branch and working tree with `git status --short --branch` before changing files.
- Use Python 3.12 and `uv`. The package and development tools are defined in `pyproject.toml`.
- The package lives in `soynlp/`; tests live in `tests/unit/` and `tests/integration/`. The CLI is `soynlp/cli.py`, exposed by the `soynlp` script and `python -m soynlp`.
- Use Ruff with a 127-character line limit, Pyright basic mode, and pytest. Do not run fixing or formatting commands when the task calls only for diagnosis.

## Branch and review workflow

- For a new issue-based implementation, use `feature/{issue-number}` off the intended base branch. For work unrelated to an issue, `feature/{short-description}` is permitted.
- Create and use a separate worktree for feature development; run Codex from that worktree so edits land on the intended branch.
- Keep a change focused, verify it, and prepare a PR against the intended base branch when requested. Do not create an issue, commit, or PR merely to investigate a question or produce an uncommitted report.

## Current refactor delivery

- Base and integration branch: `refactor-2026`.
- Implement one reviewable feature per `feature/{short-description}` branch in a separate worktree.
- Validate and commit each feature, then merge it into `refactor-2026` with `git merge --no-ff`.
- Keep feature branches locally. Do not create issues or PRs, or push remote branches during this implementation.
- Record feature commits, merge commits, and validation in `REFACTOR_PROGRESS.md` so the user can decide how to migrate the work.

## Commits

- When a commit is requested, use `<type>(<optional-scope>): <subject>` with `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `chore`, or `perf`.
- Use a Korean, noun-ending subject of at most 50 characters without a final period. Explain the reason and changes in a Korean body; use bullets when useful.
- Keep logically independent changes in separate commits. Use a co-author trailer only when the named person or agent actually contributed and attribution is requested.

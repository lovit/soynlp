# soynlp project guidance

soynlp is a Python package for unsupervised Korean natural language processing. Its public modules cover Korean character handling, corpus-based noun and word extraction, tokenization, normalization, lemmatization, part-of-speech work, and vectorization. `refactor-2026` also adds a YAML-driven pipeline and the `soynlp` CLI.

- Use Python 3.12 and `uv`. Dependencies and the CLI entry point are declared in `pyproject.toml`.
- Preserve the public ways users import and call features from `master` unless the task explicitly approves a breaking change. Check actual behavior and tests; completion markers in `REFACTOR.md` and `todo.md` are historical records.
- For repository work, read [the Codex workflow](.codex/WORKFLOW.md). It adapts the branch and commit conventions from `.claude/rules`.
- Run focused tests for a change, then use `uv run --python 3.12 pytest` for a full regression check when appropriate. Non-mutating quality checks are `uv run --python 3.12 ruff check .`, `uv run --python 3.12 ruff format --check .`, and `uv run --python 3.12 pyright`.
- Prefer public imports from package `__init__.py` modules. Keep Korean NLP terms such as *eojeol* (어절) and LR-Graph consistent with the existing code and docs.

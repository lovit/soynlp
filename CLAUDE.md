# CLAUDE.md - soynlp Project Guide

## Project Overview

**soynlp** is a Python package for Korean Natural Language Processing (NLP).
It provides unsupervised Korean text processing algorithms, particularly focused on word extraction using LR-Graph structures.

- **Author**: lovit <soy.lovit@gmail.com>
- **License**: LGPL-3.0
- **Python**: 3.12
- **Package Manager**: uv

## Project Structure

```
soynlp/
├── soynlp/                    # Main package
│   ├── __init__.py
│   ├── __main__.py            # CLI entry point
│   ├── cli.py                 # argparse-based CLI
│   ├── configs/
│   │   └── config.py          # YAML config parsing (dacite + pyyaml)
│   ├── core/
│   │   └── lrgraph.py         # LR-Graph: bidirectional left-right graph for Korean morpheme analysis
│   └── pipeline/
│       ├── pipeline.py        # Pipeline orchestrator: loads and chains tasks from YAML config
│       └── tasks/
│           ├── __init__.py    # Task registry (exports all task classes)
│           ├── task.py        # Abstract base: Task[TaskArgsType] generic class
│           ├── dummy.py       # DummyTask: no-op for testing
│           ├── read_json.py   # ReadJsonTask: reads JSONL files
│           ├── read_text.py   # ReadTextTask: reads text files
│           ├── write_json.py  # WriteJsonTask: writes JSONL files
│           ├── write_text.py  # WriteTextTask: writes text files
│           └── extract_nouns.py # ExtractNounTask: noun extraction (skeleton)
├── tests/
│   └── unit/                  # Unit tests (pytest)
├── data/                      # Sample Korean text datasets
│   └── loader.py              # Dataset loader utility
├── examples/                  # Example pipeline YAML configs
├── notes/                     # Research notes
└── .github/workflows/         # CI/CD
```

## Key Concepts

### Pipeline Architecture

- Pipelines are defined in YAML config files
- Each pipeline step is a `Task` subclass with typed `TaskArgs`
- Tasks are chained: each receives and returns a `parameters: dict`
- Tasks are loaded dynamically by name from `soynlp.pipeline.tasks`

### LR-Graph

- Core data structure for Korean morpheme analysis
- Splits eojeols (어절, space-separated Korean words) into L (left) and R (right) parts
- Maintains bidirectional frequency maps for L→R and R→L lookups
- Used for unsupervised noun extraction

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run pre-commit run --all-files

# Run a pipeline
uv run soynlp pipeline -c examples/dummy/pipeline.yaml
```

## Conventions

- Line length: 127 characters
- Formatter: ruff format (black-compatible, line-length 127)
- Linter: ruff
- Type checker: pyright (basic mode)
- Test framework: pytest
- All task classes follow the pattern: `{Name}Task` with `{Name}TaskArgs` dataclass
- Korean NLP terminology: eojeol (어절), L/R decomposition

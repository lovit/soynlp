# soynlp

Renewing ...

## Install

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a new, extremely fast Python package installer and resolver.

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
# Install production dependencies
uv pip install -r requirements.txt

# Install development dependencies
uv pip install -r requirements-dev.txt

# Or install all dependencies at once
uv pip install -r requirements-dev.txt
```

4. Install the package in development mode:

```bash
uv pip install -e .
```

### Using Poetry (Legacy)

```bash
pipx install poetry==1.8.0
poetry install
```

## Development

### Code Style

This project uses:

- [black](https://github.com/psf/black) for code formatting
- [ruff](https://github.com/astral-sh/ruff) for linting
- [pyright](https://github.com/microsoft/pyright) for type checking

To run all checks:

```bash
pre-commit run --all-files
```

### Testing

```bash
pytest
```

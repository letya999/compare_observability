# Contributing to PDF Knowledge Explorer

We welcome contributions! This project aims to compare LLM observability platforms.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/pdf-knowledge-rag.git
   cd pdf-knowledge-rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Adding a New Observability Provider

To add a new provider (e.g., `MyNewProvider`):

1. **Create implementation**:
   Create `src/observability/providers/mynewprovider.py`.
   Inherit from `ObservabilityProvider` and implement:
   - `initialize()`
   - `trace()`
   - `span()`
   - `log_llm_call()`
   - `log_retrieval()`
   - `log_error()`

2. **Register**:
   Add it to `src/observability/providers/__init__.py`.

3. **Config**:
   Add it to `ObservabilityProvider` literal in `src/config.py`.

## Testing

Run tests before submitting a PR:
```bash
pytest
```

## Code Style

We use `ruff` for linting.
```bash
ruff check .
```

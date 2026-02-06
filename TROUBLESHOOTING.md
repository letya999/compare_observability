# Observability Providers - Troubleshooting Guide

## Current Status

Based on the screenshots, only **Langfuse** and **Opik** are receiving traces. Let's fix the others.

## Issues Fixed

### 1. **Manager.py Context Manager Bug** ✅
- **Problem**: Context managers weren't being closed properly
- **Fix**: Updated `manager.py` to track and close the same context manager objects

### 2. **Opik Package Missing** ✅
- **Problem**: `opik` package wasn't installed
- **Fix**: Installed via `pip install opik`

### 3. **Opik Span Type** ✅
- **Problem**: Used invalid `retriever` type
- **Fix**: Changed to `general` (valid types: general, tool, llm, guardrail)

### 4. **Arize Import Error** ✅
- **Problem**: `StatusCode` import issue in error handling
- **Fix**: Added proper import in exception block

### 5. **Braintrust Project Name** ✅
- **Problem**: Hardcoded project name `pdf-knowledge-rag`
- **Fix**: Changed to `compare-observability` (or `BRAINTRUST_PROJECT` env var)

## Testing

Run the test script:

```bash
python test_providers.py
```

Expected output:
```
Active providers: ['langfuse', 'langsmith', 'opik', 'arize', 'braintrust']
Total: 5/5 expected
```

## Common Issues

### Arize Phoenix
- **No traces**: Check if packages are installed:
  ```bash
  pip install arize-phoenix opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
  ```
- **Cloud endpoint**: Verify `PHOENIX_API_KEY` and `PHOENIX_COLLECTOR_ENDPOINT` in `.env`

### Braintrust
- **No traces**: Check if package is installed:
  ```bash
  pip install braintrust
  ```
- **Wrong project**: Traces might be in `pdf-knowledge-rag` project instead of `compare-observability`

### LangSmith
- **No traces**: Verify environment variables:
  - `LANGCHAIN_API_KEY` (or `LANGSMITH_API_KEY`)
  - `LANGSMITH_TRACING=true`
  - `LANGCHAIN_TRACING_V2=true`
  - `LANGCHAIN_PROJECT=compare_observability`

## Verification

After running `test_providers.py`, check each dashboard:

1. **Langfuse**: https://cloud.langfuse.com → Project: `compare_observability`
2. **LangSmith**: https://smith.langchain.com → Project: `compare_observability`
3. **Opik**: https://www.comet.com/opik → Workspace: `artem-letyushev`, Project: `compare_observability`
4. **Arize Phoenix**: https://app.phoenix.arize.com → Look for traces
5. **Braintrust**: https://www.braintrust.dev → Project: `compare-observability`

## Environment Variables

Make sure `.env` has:

```env
OBSERVABILITY_PROVIDERS=langfuse,langsmith,opik,arize,braintrust

# LangSmith
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=compare_observability

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Arize Phoenix
PHOENIX_API_KEY=ak-...
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com

# Opik
OPIK_API_KEY=...
OPIK_WORKSPACE=artem-letyushev
OPIK_URL_OVERRIDE=https://www.comet.com/opik/api
OPIK_PROJECT_NAME=compare_observability

# Braintrust
BRAINTRUST_API_KEY=sk-...
BRAINTRUST_PROJECT=compare-observability
```

## Next Steps

If providers still don't work after running the test:

1. Check console output for initialization errors
2. Verify API keys are valid
3. Check if packages are installed: `pip list | grep -i "langsmith\|langfuse\|opik\|phoenix\|braintrust"`
4. Look for error messages in the console during initialization

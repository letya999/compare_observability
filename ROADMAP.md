# 🚀 Project Roadmap: PDF Knowledge Explorer Evolution

This document outlines the strategic development plan to transform this project into a comprehensive benchmarking platform for LLM observability and a top-tier Open Source tool.

---

## Phase 1: Complex Pipeline & Advanced Retrieval
**Goal:** Increase trace complexity to test how platforms handle parallelism, nesting, and diverse data sources.

### 1.1 Hybrid Search Implementation
*   **Tasks:**
    *   Add `rank_bm25` dependency.
    *   Implement BM25 index creation alongside ChromaDB during ingestion.
    *   Modify `src/pipeline/retriever.py` to support `hybrid_retrieve` (Parallel Vector + BM25).
    *   Implement Reciprocal Rank Fusion (RRF) for merging results.
*   **Acceptance Criteria:** Traces clearly show two parallel retrieval branches converging into a single Reranker step.

### 1.2 Agentic Tools (Tool Use)
*   **Tasks:**
    *   Create `src/pipeline/tools.py` with mock/real tools (e.g., `web_search`, `calculator`).
    *   Introduce a "Reasoning" step in the orchestrator before final generation.
    *   Implement tool-calling logic using OpenAI's tool-calling API.
*   **Acceptance Criteria:** Traces display tool calls as distinct nested spans with visible arguments and return values.

---

## Phase 2: Evaluation & Cost Verification
**Goal:** Shift from subjective visual comparison to data-driven benchmarking.

### 2.1 Automated Evals (Ragas/DeepEval)
*   **Tasks:**
    *   Integrate `ragas` for automated metrics (*Faithfulness*, *Answer Relevance*).
    *   Implement a post-query hook to run evaluations.
    *   Export evaluation scores as metadata/tags to all active observability providers.
*   **Acceptance Criteria:** Metrics appear directly within the observability platform UIs (where supported) or in the Streamlit dashboard.

### 2.2 Cost Audit & Verification
*   **Tasks:**
    *   Implement local token counting using `tiktoken`.
    *   Create a pricing ledger for OpenAI models.
    *   Compare "Local Estimated Cost" vs "Provider Reported Cost" in the final report.
*   **Acceptance Criteria:** Streamlit UI displays a "Cost Accuracy" metric for each provider.

---

## Phase 3: New Integrations
**Goal:** Expand the comparison matrix to include the latest industry-standard tools.

### 3.1 Weights & Biases Weave
*   **Tasks:**
    *   Implement `src/observability/providers/weave.py`.
    *   Integrate using `@weave.op()` or the `weave.init()` SDK.
*   **Acceptance Criteria:** Full RAG pipeline traces visible in the W&B Weave interface.

### 3.2 Honeycomb (OpenTelemetry)
*   **Tasks:**
    *   Configure standard OpenTelemetry (OTeL) exporters.
    *   Map RAG pipeline spans to OTeL semantic conventions.
*   **Acceptance Criteria:** Traces appear in Honeycomb with correct timing and hierarchy.

---

## Phase 4: Infrastructure & Developer Experience
**Goal:** Lower the barrier to entry for new developers and users.

### 4.1 Dockerization
*   **Tasks:**
    *   Create a multi-stage `Dockerfile` for the Streamlit app.
    *   Define `docker-compose.yml` including the app and ChromaDB.
    *   Configure persistent volumes for data and PDFs.
*   **Acceptance Criteria:** Project starts successfully with a single `docker-compose up` command.

---

## Phase 5: Open Source Excellence
**Goal:** Build a professional brand and a welcoming community.

### 5.1 Documentation & Branding
*   **Tasks:**
    *   Add interactive "Quick Start" guide to Streamlit.
    *   Create a high-quality architecture diagram using Mermaid.js in `README.md`.
    *   Write `CONTRIBUTING.md` with a "Provider Template" for new integrations.
*   **Acceptance Criteria:** A new contributor can add a provider in under 30 minutes following the docs.

### 5.2 CI/CD & Code Quality
*   **Tasks:**
    *   Setup GitHub Actions for `linting (ruff)` and `testing (pytest)`.
    *   Implement `pre-commit` hooks for code style enforcement.
    *   Configure a "Demo Mode" with pre-captured traces for users without API keys.
*   **Acceptance Criteria:** No code can be merged without passing all linting and unit tests.

### 5.3 Static Comparison Report
*   **Tasks:**
    *   Define a "Gold Dataset" of queries and documents.
    *   Generate a comprehensive "State of LLM Observability" report (Markdown/PDF).
    *   Include this report in the repo as a primary value proposition.

---

## Implementation Priority
1.  **Phase 4 (Docker)** - For clean development environments.
2.  **Phase 1 (Hybrid/Tools)** - To provide "meaty" traces for observation.
3.  **Phase 2 (Evals)** - To add unique benchmarking value.
4.  **Phase 5 (OS Prep)** - For final public launch.
5.  **Phase 3 (Expansion)** - Ongoing addition of new platforms.

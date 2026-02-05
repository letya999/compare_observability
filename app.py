"""Streamlit UI for PDF Knowledge Explorer with Observability Comparison."""

import json
import time
from pathlib import Path

import streamlit as st

from src.config import config
from src.logger import logger
from src.pipeline.traced_orchestrator import TracedRAGOrchestrator
from src.evaluations.comparison_matrix import ComparisonMatrix, PLATFORMS, CRITERIA, CriteriaCategory
from scenarios import SCENARIOS, ScenarioRunner


# Page config
st.set_page_config(
    page_title="PDF Knowledge Explorer - Observability Comparison",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def get_orchestrator():
    """Initialize the traced orchestrator."""
    return TracedRAGOrchestrator()


def main():
    st.title("PDF Knowledge Explorer")
    st.caption("RAG System for Testing Observability Platforms")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Provider selection
        st.subheader("Observability Providers")
        available_providers = list(PLATFORMS.keys())
        selected_providers = st.multiselect(
            "Active providers",
            available_providers,
            default=config.observability_providers or [],
            help="Select which observability platforms to send traces to",
        )

        # Display provider status
        if selected_providers:
            st.write("**Status:**")
            orchestrator = get_orchestrator()
            for provider in selected_providers:
                is_active = provider in orchestrator.obs_manager.active_providers
                status = "" if is_active else ""
                st.write(f"{status} {provider}")

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["Quick Start", "Query Interface", "PDF Management", "Test Scenarios", "Comparison Matrix", "Results"],
            index=0,
        )

    # Main content
    if page == "Quick Start":
        quick_start_page()
    elif page == "Query Interface":
        query_page()
    elif page == "PDF Management":
        pdf_management_page()
    elif page == "Test Scenarios":
        scenarios_page()
    elif page == "Comparison Matrix":
        matrix_page()
    else:
        results_page()


def query_page():
    """Main query interface."""
    st.header("Ask Questions")

    orchestrator = get_orchestrator()
    stats = orchestrator.get_stats()

    # Show stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Indexed Chunks", stats["vector_store"]["total_chunks"])
    with col2:
        st.metric("Active Providers", len(orchestrator.obs_manager.active_providers))
    with col3:
        st.metric("LLM Model", stats["config"]["llm_model"])

    st.divider()

    # Query input
    query = st.text_area(
        "Your question",
        placeholder="What is the attention mechanism in transformers?",
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        stream = st.checkbox("Stream response", value=True)
    with col2:
        skip_graph = st.checkbox("Skip graph extraction", value=False)
        retrieval_only = st.checkbox("Retrieval Only (Debug)", value=False, help="Skip generation to inspect retrieved chunks")

    if st.button("Ask", type="primary", disabled=not query):
        with st.spinner("Processing..."):
            start_time = time.time()

            if stream:
                # Streaming response
                response_container = st.empty()
                full_response = ""

                try:
                    gen = orchestrator.query(
                        query, 
                        stream=True, 
                        skip_graph_extraction=skip_graph,
                        retrieval_only=retrieval_only
                    )
                    for chunk in gen:
                        if isinstance(chunk, str):
                            full_response += chunk
                            response_container.markdown(full_response + "")
                        else:
                            result = chunk
                    response_container.markdown(full_response)
                except Exception as e:
                    logger.error("Streaming query execution failed", exc_info=True)
                    st.error(f"Error: {e}")
                    return
            else:
                # Non-streaming
                try:
                    result = orchestrator.query(
                        query, 
                        stream=False, 
                        skip_graph_extraction=skip_graph,
                        retrieval_only=retrieval_only
                    )
                    st.markdown(result.response.answer)
                except Exception as e:
                    logger.error("Standard query execution failed", exc_info=True)
                    st.error(f"Error: {e}")
                    return

            total_time = time.time() - start_time

            # Show metrics
            st.divider()
            st.subheader("Trace Metrics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Latency", f"{result.total_latency_ms:.0f}ms")
            with col2:
                st.metric("Retrieved Chunks", len(result.retrieved_chunks))
            with col3:
                tokens = result.response.token_usage.get("total_tokens", 0)
                st.metric("Total Tokens", tokens)
            with col4:
                st.metric("Concepts Found", len(result.concepts))

            # Step latencies
            st.subheader("Pipeline Steps")
            for step, latency in result.step_latencies.items():
                st.progress(min(latency / 5000, 1.0), text=f"{step}: {latency:.0f}ms")

            # Retrieved chunks
            with st.expander("Retrieved Chunks", expanded=False):
                for i, chunk in enumerate(result.reranked_chunks):
                    st.markdown(f"**[{i+1}]** Score: {chunk.score:.3f} | Page: {chunk.chunk.page_number}")
                    st.text(chunk.chunk.text[:500] + "..." if len(chunk.chunk.text) > 500 else chunk.chunk.text)
                    st.divider()

            # Concepts
            if result.concepts:
                with st.expander("Extracted Concepts", expanded=False):
                    for concept in result.concepts:
                        st.markdown(f"**{concept.source}** -> *{concept.relation_type}* -> **{concept.target}**")
                        st.caption(f"Confidence: {concept.confidence:.2f}")

            # Trace URLs
            trace_urls = orchestrator.obs_manager.get_trace_urls(
                type("FakeTrace", (), {"provider_spans": {}, "trace_ids": {}})()
            )
            if trace_urls:
                st.subheader("View Traces")
                for provider, url in trace_urls.items():
                    st.markdown(f"[{provider}]({url})")


def pdf_management_page():
    """PDF upload and management."""
    st.header("PDF Management")

    orchestrator = get_orchestrator()

    # Upload
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Ingest PDFs", type="primary"):
            progress = st.progress(0)
            for i, file in enumerate(uploaded_files):
                # Save temporarily
                pdf_path = config.pdf_dir / file.name
                with open(pdf_path, "wb") as f:
                    f.write(file.getvalue())

                # Ingest
                with st.spinner(f"Ingesting {file.name}..."):
                    doc, chunk_count = orchestrator.ingest_pdf(pdf_path)
                    st.success(f"Ingested {file.name}: {chunk_count} chunks")

                progress.progress((i + 1) / len(uploaded_files))

    st.divider()

    # Stats
    stats = orchestrator.get_stats()
    st.metric("Total Chunks in Index", stats["vector_store"]["total_chunks"])

    # Clear data
    if st.button("Clear All Data", type="secondary"):
        orchestrator.clear_data()
        st.success("Data cleared")
        st.rerun()


def scenarios_page():
    """Test scenarios runner."""
    st.header("Test Scenarios")

    st.markdown("""
    Run predefined test scenarios to compare how different observability platforms handle various use cases.
    """)

    # Scenario selection
    scenario_names = list(SCENARIOS.keys())
    selected_scenarios = st.multiselect(
        "Select scenarios to run",
        scenario_names,
        default=scenario_names[:2],
    )

    # Show scenario details
    for name in selected_scenarios:
        scenario = SCENARIOS[name]
        with st.expander(f"**{scenario.name}** - {scenario.description}"):
            st.write(f"**Type:** {scenario.type.value}")
            st.write(f"**Expected spans:** {scenario.expected_spans}")
            st.write(f"**Query:** {scenario.query if isinstance(scenario.query, str) else scenario.query[0]}")
            st.write(f"**Checks:** {[c.value for c in scenario.checks]}")

    if st.button("Run Selected Scenarios", type="primary", disabled=not selected_scenarios):
        runner = ScenarioRunner()

        progress = st.progress(0)
        for i, scenario_name in enumerate(selected_scenarios):
            with st.spinner(f"Running {scenario_name}..."):
                result = runner.run_scenario(scenario_name)

                # Show result
                status = "" if result.success else ""
                st.write(f"{status} **{scenario_name}**: {sum(result.check_results.values())}/{len(result.check_results)} checks passed")

                if result.errors:
                    st.error(f"Errors: {result.errors}")

            progress.progress((i + 1) / len(selected_scenarios))

        # Export results
        output_path = runner.export_results()
        st.success(f"Results exported to: {output_path}")

        # Summary
        runner.print_summary()
        runner.shutdown()


def matrix_page():
    """Comparison matrix management."""
    st.header("Comparison Matrix")

    matrix = ComparisonMatrix()

    # Try to load existing data
    matrix_path = Path("results/comparison_matrix.json")
    if matrix_path.exists():
        matrix.import_json(matrix_path)
        st.info("Loaded existing matrix data")

    # Category tabs
    tabs = st.tabs([cat.value.replace("_", " ").title() for cat in CriteriaCategory])

    for tab, category in zip(tabs, CriteriaCategory):
        with tab:
            category_criteria = [
                (name, c) for name, c in CRITERIA.items()
                if c.category == category
            ]

            # Create editable table
            st.subheader(f"{category.value.replace('_', ' ').title()}")

            for criterion_name, criterion in category_criteria:
                st.markdown(f"**{criterion.name}**")
                st.caption(criterion.description)

                cols = st.columns(len(PLATFORMS))
                for col, (platform, abbrev) in zip(cols, PLATFORMS.items()):
                    with col:
                        current = matrix.get_score(platform, criterion_name)
                        current_value = current.value if current else None

                        if criterion.value_type == "boolean":
                            new_value = st.checkbox(
                                abbrev,
                                value=bool(current_value),
                                key=f"{criterion_name}_{platform}",
                            )
                        elif criterion.value_type == "rating":
                            new_value = st.slider(
                                abbrev,
                                min_value=0,
                                max_value=criterion.max_value or 5,
                                value=current_value or 0,
                                key=f"{criterion_name}_{platform}",
                            )
                        else:
                            new_value = st.text_input(
                                abbrev,
                                value=current_value or "",
                                key=f"{criterion_name}_{platform}",
                            )

                        if new_value != current_value:
                            matrix.set_score(platform, criterion_name, new_value)

                st.divider()

    # Export buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Matrix"):
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            matrix.export_json(matrix_path)
            st.success("Matrix saved!")

    with col2:
        if st.button("Export Markdown"):
            md_path = Path("results/comparison_matrix.md")
            md_path.parent.mkdir(parents=True, exist_ok=True)
            matrix.export_markdown(md_path)
            st.success(f"Exported to {md_path}")

    with col3:
        if st.button("Show Summary"):
            matrix.print_summary()
            totals = matrix.calculate_totals()

            st.subheader("Platform Rankings")
            sorted_platforms = sorted(
                totals.items(),
                key=lambda x: x[1]["bool_features"],
                reverse=True,
            )
            for platform, scores in sorted_platforms:
                st.write(f"**{platform}**: {scores['bool_features']}/{scores['bool_total']} features ({scores['bool_percentage']:.0f}%)")


def results_page():
    """View and analyze results."""
    st.header("Results & Analysis")

    results_dir = Path("results")
    if not results_dir.exists():
        st.warning("No results directory found. Run some scenarios first.")
        return

    # Find result files
    result_files = list(results_dir.glob("scenario_results_*.json"))
    if not result_files:
        st.warning("No scenario results found. Run some scenarios first.")
        return

    # Select result file
    selected_file = st.selectbox(
        "Select result file",
        result_files,
        format_func=lambda x: x.name,
    )

    if selected_file:
        with open(selected_file) as f:
            data = json.load(f)

        st.write(f"**Run date:** {data['run_date']}")
        st.write(f"**Providers:** {', '.join(data['providers'])}")

        st.divider()

        # Show scenarios
        for scenario in data["scenarios"]:
            status = "" if scenario["success"] else ""
            with st.expander(f"{status} {scenario['name']} - {scenario['description']}"):
                st.write(f"**Type:** {scenario['type']}")

                # Check results
                st.subheader("Checks")
                for check, passed in scenario["check_results"].items():
                    st.write(f"{'::white_check_mark::' if passed else ''} {check}")

                # Metrics
                if "metrics" in scenario and "pipeline_result" in scenario["metrics"]:
                    result = scenario["metrics"]["pipeline_result"]
                    st.subheader("Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Latency", f"{result['total_latency_ms']:.0f}ms")
                    with col2:
                        st.metric("Retrieved", result["retrieved_count"])
                    with col3:
                        st.metric("Tokens", result["token_usage"].get("total_tokens", 0))

                # Errors
                if scenario["errors"]:
                    st.error(f"Errors: {scenario['errors']}")


def quick_start_page():
    """Interactive Quick Start Guide."""
    st.header("Quick Start Guide 🚀")
    
    st.markdown("""
    Welcome to the **PDF Knowledge Explorer**! This tool is designed to benchmark and compare LLM observability platforms.
    
    ### How it works
    1. **Configuration**: Use the sidebar to select which observability providers you want to test (e.g., Langfuse, Weave, Honeycomb).
    2. **Ingest Data**: Go to **PDF Management** to upload your own PDFs or use the pre-loaded data.
    3. **Query**: Use the **Query Interface** to ask questions. The system will retrieve relevant chunks, rerank them, and generate an answer with citations.
    4. **Observe**: Click the trace links to see how different platforms visualize the RAG pipeline.
    
    ### Key Features
    * **Hybrid Search**: Combines Vector Search (ChromaDB) and Keyword Search (BM25).
    * **Agentic Tools**: Can use a Calculator or Web Search if needed (Reasoning Step).
    * **Evaluations**: Automatically runs Ragas metrics (Faithfulness, Answer Relevance) and estimates costs.
    * **Comparison**: Run **Test Scenarios** to systematically verify features across platforms.
    """)
    
    st.info("💡 **Tip:** Make sure you have set your API keys in the `.env` file or environment variables.")
    
    st.subheader("Architecture")
    st.code("""
    User Query -> Query Analysis -> Hybrid Retrieval (Vector + BM25) -> Reranking -> Reasoning (Tools) -> Generation -> Graph Extraction
    """, language="text")
    
    if st.button("Go to Query Interface", type="primary"):
        st.switch_page("app.py") # Note: switch_page might not work well with single-script structure logic, better to just user helper
        # Actually in this structure, we just change state, but we can't easily do that without session state triggers.
        # Just let user navigate.
        pass


if __name__ == "__main__":
    main()

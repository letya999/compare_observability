"""Streamlit UI for PDF Knowledge Explorer with Observability Comparison."""

import json
import time
from pathlib import Path

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# Force reload env vars from explicit path
env_path = Path(__file__).parent / ".env"
# st.write(f"Loading config from: {env_path}") # Debug output
load_dotenv(dotenv_path=env_path, override=True)

# Debug: Print loaded keys status (masked)
keys_to_check = ["LANGCHAIN_API_KEY", "OPIK_API_KEY", "ARIZE_API_KEY", "BRAINTRUST_API_KEY"]
status_msg = "Env check: " + ", ".join([f"{k}={'OK' if os.getenv(k) else 'MISSING'}" for k in keys_to_check])
print(status_msg)
# st.caption(status_msg) # Show in UI for debugging

from src.config import config
from src.logger import logger
from src.pipeline.traced_orchestrator import TracedRAGOrchestrator
from src.evaluations.comparison_matrix import ComparisonMatrix, PLATFORMS, CRITERIA, CriteriaCategory
from scenarios import SCENARIOS, ScenarioRunner, DiscoveryGenerator


# Page config
st.set_page_config(
    page_title="PDF Knowledge Explorer - Observability Comparison",
    page_icon="🔍",
    layout="wide",
)

def load_css():
    """Load custom CSS styles."""
    css_path = Path(__file__).parent / "src" / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load styles immediately
load_css()


@st.cache_resource(show_spinner="Initializing Orchestrator...")
def get_orchestrator(selected_providers: list[str] | None = None):
    """Initialize the traced orchestrator with selected providers."""
    # If None, will use default from config
    return TracedRAGOrchestrator(observability_providers=selected_providers)


def main():
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

        if st.button("🔄 Reload & Reset", help="Clear cache and reload configuration"):
            st.cache_resource.clear()
            st.rerun()

        # Display provider status
        if selected_providers:
            st.write("**Status:**")
            # Pass selected providers to init
            orchestrator = get_orchestrator(selected_providers)
            # Store in session state for other pages
            st.session_state.orchestrator = orchestrator
            
            active_count = 0
            for provider in selected_providers:
                is_active = provider in orchestrator.obs_manager.active_providers
                status = "✅" if is_active else "❌"
                if is_active: active_count += 1
                st.write(f"{status} {provider}")
                
                # Show error detail if failed
                if not is_active:
                    # Use getattr to be safe against cached old instances without init_errors
                    init_errors = getattr(orchestrator.obs_manager, 'init_errors', {}) 
                    error_msg = init_errors.get(provider, "Initialization failed. Check logs.")
                    st.caption(f":red[{error_msg}]")
                
            if active_count == 0 and selected_providers:
                st.error("No providers active! Check API keys in .env")

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

    # Get orchestrator with currently selected providers from config (or default)
    # Note: In a real app we might want to pass state, but streamlit reruns script top-down
    # so we need to get the selection from session state or re-read sidebar widget if possible.
    # But sidebar widgets are available here.
    
    # We need to match what was selected in sidebar. 
    # Since we can't easily access sidebar widget value here without session state,
    # we'll rely on the cached resource being the right one if user didn't change it,
    # OR we should really move get_orchestrator call to main and pass it down.
    # Refactoring slightly:
    
    # Actually, simpler: we assume the user configured in sidebar. 
    # To be safe, let's grab the default from config if we can't see the widget,
    # but the widget IS in the same script run.
    # Let's use config.observability_providers as fallback if needed, 
    # but better to use the specific sidebar key if we assigned one, or just trust the cache 
    # (which updates when args change).
    
    # Let's get the orchestrator again with current config
    # We can reconstruct valid providers list from config or defaults
    current_providers = config.observability_providers # Fallback
    
    # Try to find the multiselect value in session state if available, key isn't set above so it's auto-generated.
    # Instead, let's just use get_orchestrator wrapped in a way that uses the latest args.
    # BUT, we can't easily pass args here without passing them to query_page.
    # Let's just instantiate with config.observability_providers for now in this scope 
    # assuming sidebar updated the config or we just use what's cached.
    
    # BETTER FIX: pass orchestrator to pages. But that requires bigger refactor.
    # Hack: sidebar is run before this. st.session_state should have the value if we key it.
    
    # Let's key the multiselect in main() to access it here.
    pass # We will fix the multiselect key in main chunk above next.
    
    orchestrator = get_orchestrator(None) # Expecting this might need fix.
    
    # Wait, st.cache_resource is smart. If we call get_orchestrator(selected_list) in sidebar,
    # and then get_orchestrator(same_list) here, it works. 
    # But here we don't have 'selected_list'.
    
    # Solution: We will inject orchestrator into st.session_state in main()
    if "orchestrator" in st.session_state:
        orchestrator = st.session_state.orchestrator
    else:
        orchestrator = get_orchestrator(None) # Fallback

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
            trace_urls = getattr(result, "trace_urls", {})
            if trace_urls:
                st.subheader("View Traces")
                cols = st.columns(min(len(trace_urls), 4))
                for i, (provider, url) in enumerate(trace_urls.items()):
                    with cols[i % 4]:
                        st.link_button(f"🔍 {provider}", url, use_container_width=True)


def pdf_management_page():
    """PDF upload and management."""
    st.header("PDF Management")

    if "orchestrator" in st.session_state:
        orchestrator = st.session_state.orchestrator
    else:
        orchestrator = get_orchestrator(None)

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
    # Data Management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear All Data", type="secondary", help="Delete all chunks from the index"):
            orchestrator.clear_data()
            st.success("Data cleared")
            st.rerun()

    with col2:
        if st.button("Re-index All PDFs", type="primary", help="Clear index and re-ingest all PDFs from data/pdfs"):
            orchestrator.clear_data()
            pdfs = list(config.pdf_dir.glob("*.pdf"))
            if not pdfs:
                st.warning("No PDFs found in data directory.")
            else:
                progress_bar = st.progress(0)
                for i, pdf_path in enumerate(pdfs):
                    with st.spinner(f"Ingesting {pdf_path.name}..."):
                        orchestrator.ingest_pdf(pdf_path)
                    progress_bar.progress((i + 1) / len(pdfs))
                st.success(f"Re-indexed {len(pdfs)} documents.")
                time.sleep(1) # Give time to read message
                st.rerun()


def scenarios_page():
    """Test scenarios runner."""
    st.header("Test Scenarios")

    if "orchestrator" in st.session_state:
        orchestrator = st.session_state.orchestrator
    else:
        orchestrator = get_orchestrator(None)

    st.markdown("""
    Run test scenarios to compare how different observability platforms handle various use cases.
    """)

    # --- Discovery Section ---
    st.subheader("🔍 Content-Aware Discovery")
    st.caption("Generate tests based on the documents currently in your index (e.g., 'Master and Margarita')")

    if "custom_scenarios" not in st.session_state:
        st.session_state.custom_scenarios = {}

    col1, col2 = st.columns([2, 1])
    with col1:
        num_to_gen = st.slider("Number of scenarios to generate", 1, 10, 3)
    with col2:
        if st.button("✨ Generate Tests", help="Analyze indexed PDFs and create relevant test cases"):
            with st.spinner("Analyzing content and generating scenarios..."):
                generator = DiscoveryGenerator(orchestrator)
                new_scenarios = generator.generate_scenarios(count=num_to_gen)
                for s in new_scenarios:
                    st.session_state.custom_scenarios[s.name] = s
                st.success(f"Generated {len(new_scenarios)} content-aware scenarios!")

    st.divider()

    # Combine predefined and custom scenarios
    all_scenarios = {**SCENARIOS, **st.session_state.custom_scenarios}
    scenario_names = list(all_scenarios.keys())

    # Filter out technical ones if non-technical docs are loaded?
    # Let's just let the user choose.

    selected_names = st.multiselect(
        "Select scenarios to run",
        scenario_names,
        default=[n for n in scenario_names if n in st.session_state.custom_scenarios][:3] or scenario_names[:2],
    )

    # Show scenario details
    if selected_names:
        st.subheader("Scenario Preview")
        for name in selected_names:
            scenario = all_scenarios[name]
            is_generated = scenario.metadata.get("generated", False)
            label = f"**{scenario.name}** {'(✨ Generated)' if is_generated else ''}"
            with st.expander(f"{label} - {scenario.description}"):
                st.write(f"**Type:** {scenario.type.value}")
                st.write(f"**Query:** {scenario.query if isinstance(scenario.query, str) else scenario.query[0]}")
                st.write(f"**Checks:** {[c.value for c in scenario.checks]}")

    if st.button("🚀 Run Selected Scenarios", type="primary", disabled=not selected_names):
        runner = ScenarioRunner()

        progress = st.progress(0)
        results_container = st.container()

        for i, name in enumerate(selected_names):
            scenario = all_scenarios[name]
            with st.spinner(f"Running {name}..."):
                result = runner.run_scenario(scenario)

                # Show result in container
                with results_container:
                    status = "✅" if result.success else "❌"
                    st.write(f"{status} **{name}**: {sum(result.check_results.values())}/{len(result.check_results)} checks passed")
                    if result.errors:
                        st.error(f"Errors: {result.errors}")

            progress.progress((i + 1) / len(selected_names))

        # Export results
        output_path = runner.export_results()
        st.success(f"Results exported to: {output_path}")

        # Summary components
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

    # Platform Legend
    with st.expander("ℹ️ Platform Abbreviation Legend", expanded=True):
        cols = st.columns(3)
        platform_items = list(PLATFORMS.items())
        for i in range(3):
            with cols[i]:
                for j in range(i, len(platform_items), 3):
                    name, abbrev = platform_items[j]
                    st.write(f"**{abbrev}**: {name.title()}")

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
                st.markdown(f"#### {criterion.name}")
                st.caption(criterion.description)

                # Show inputs in a grid that handles many platforms better
                cols = st.columns(len(PLATFORMS))
                for col, (platform, abbrev) in zip(cols, PLATFORMS.items()):
                    with col:
                        current = matrix.get_score(platform, criterion_name)
                        current_value = current.value if current else None
                        
                        # Add platform name to tooltip
                        help_text = f"{platform.title()} ({abbrev})"

                        if criterion.value_type == "boolean":
                            new_value = st.checkbox(
                                abbrev,
                                value=bool(current_value),
                                key=f"{criterion_name}_{platform}",
                                help=help_text
                            )
                        elif criterion.value_type == "rating":
                            new_value = st.number_input(
                                abbrev,
                                min_value=0,
                                max_value=criterion.max_value or 5,
                                value=int(current_value or 0),
                                key=f"{criterion_name}_{platform}",
                                help=help_text,
                                label_visibility="visible"
                            )
                        else:
                            new_value = st.text_input(
                                abbrev,
                                value=current_value or "",
                                key=f"{criterion_name}_{platform}",
                                help=help_text
                            )

                        if new_value != current_value:
                            matrix.set_score(platform, criterion_name, new_value)

                st.divider()

    # Export buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("💾 Save Matrix"):
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            matrix.export_json(matrix_path)
            st.success("Matrix saved!")

    with col2:
        if st.button("📝 Export Markdown"):
            md_path = Path("results/comparison_matrix.md")
            md_path.parent.mkdir(parents=True, exist_ok=True)
            matrix.export_markdown(md_path)
            st.success(f"Exported to {md_path}")

    with col3:
        if st.button("📊 Show Rankings"):
            totals = matrix.calculate_totals()
            
            # 1. Prepare Data for Charts
            platforms = []
            feature_scores = []
            ratings = []
            
            for platform, scores in totals.items():
                platforms.append(PLATFORMS[platform]) # Use abbrev
                feature_scores.append(scores['bool_percentage'])
                ratings.append(scores['avg_rating'] * 20) # Scale 5 to 100 for comparison
            
            # 2. Radar Chart
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=feature_scores,
                theta=platforms,
                fill='toself',
                name='Feature Completion %'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=ratings,
                theta=platforms,
                fill='toself',
                name='User Rating (Scaled %)'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=True,
                title="Platform Comparison: Features vs Ratings",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # 3. Tables
            st.subheader("Detailed Rankings")
            sorted_platforms = sorted(
                totals.items(),
                key=lambda x: (x[1]["bool_features"], x[1]["avg_rating"]),
                reverse=True,
            )
            
            # Display as a dataframe for better look
            data = []
            for platform, scores in sorted_platforms:
                data.append({
                    "Platform": platform.title(),
                    "Features": f"{scores['bool_features']}/{scores['bool_total']} ({scores['bool_percentage']:.0f}%)",
                    "Rating": f"{scores['avg_rating']:.1f}/5"
                })
            st.dataframe(data, use_container_width=True, hide_index=True)

    with col4:
        if st.button("🗑️ Clear Matrix"):
            if st.checkbox("Confirm clear?"):
                if matrix_path.exists():
                    matrix_path.unlink()
                st.rerun()


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
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🚀 PDF Knowledge Explorer</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #E0E0E0; margin-bottom: 3rem;'>
        The ultimate workbench for benchmarking <b>LLM Observability Platforms</b>. 
        Compare traces, latency, and costs across 12+ providers in real-time.
    </div>
    """, unsafe_allow_html=True)

    # 1. Features Grid
    st.subheader("✨ Key Capabilities")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container(border=True):
            st.markdown("### 📚 RAG Engine")
            st.caption("Hybrid Search (Vector + BM25) with reranking for high-precision retrieval.")
    
    with col2:
        with st.container(border=True):
            st.markdown("### 🕵️ Observability")
            st.caption("Send traces to LangSmith, Langfuse, Arize, and 9 others simultaneously.")

    with col3:
        with st.container(border=True):
            st.markdown("### 🧪 Scenarios")
            st.caption("Run automated test cases (Multi-hop, Long Context) to verify platform features.")
            
    with col4:
        with st.container(border=True):
            st.markdown("### 📊 Analytics")
            st.caption("Compare feature matrices, ratings, and performance metrics side-by-side.")

    st.divider()

    # 2. Workflow
    st.subheader("⚡ How to get started")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**1. Configure**\nSelect providers in the sidebar and set API keys.")
    with c2:
        st.info("**2. Ingest**\nUpload PDFs in 'PDF Management' to build your knowledge base.")
    with c3:
        st.info("**3. Query**\nAsk questions in 'Query Interface' and watch the traces flow.")
    with c4:
        st.info("**4. Compare**\nCheck 'Comparison Matrix' to rate and rank platforms.")

    st.divider()
    
    # 3. Architecture
    st.subheader("🏗️ System Architecture")
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        bgcolor="transparent";
        node [style=filled, fillcolor="#262730", fontcolor="white", shape=box, fontname="Sans-Serif"];
        edge [color="#F63366"];
        
        User -> "Query Analysis" [color="white"];
        "Query Analysis" -> "Hybrid Retrieval";
        "Hybrid Retrieval" -> "Reranker";
        "Reranker" -> "Reasoning (Tools)";
        "Reasoning (Tools)" -> "Generation";
        "Generation" -> "Graph Extraction";
        "Graph Extraction" -> User [label="Answer", color="white"];
        
        subgraph cluster_obs {
            label = "Observability Layer (Async)";
            style=dashed;
            color="#F63366";
            fontcolor="white";
            "LangSmith"; "Langfuse"; "Arize"; "Others...";
        }
        
        "Generation" -> "LangSmith" [style=dotted];
        "Generation" -> "Langfuse" [style=dotted];
    }
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Start Querying ➡️", type="primary", use_container_width=True):
            st.switch_page("app.py") # Just reloads, user needs to click nav. 
            # Note: switch_page("app.py") works if it's a multipage app file structure.
            # Here it's a single file app, so this might fail or just reload.
            # Let's just guide user.
            pass
    with col2:
        st.caption("Check the sidebar to navigate to other pages.")


if __name__ == "__main__":
    main()

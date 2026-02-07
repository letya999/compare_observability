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
from src.evaluations.benchmark import PerformanceBenchmarker
from src.evaluations.capability_detector import CapabilityDetector
from scenarios import SCENARIOS, ScenarioRunner, DiscoveryGenerator

# Import sample data utilities
try:
    from src.utils.sample_data import initialize_sample_data, has_sample_data, get_sample_queries
except ImportError:
    logger.warning("Sample data module not available")
    has_sample_data = lambda: False
    initialize_sample_data = lambda force=False: (None, [])
    get_sample_queries = lambda: []


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
            [
                "Quick Start", 
                "Query Interface", 
                "PDF Management", 
                "Test Scenarios", 
                "Comparison Matrix",
                "🚀 Benchmarking",
                "🔍 Auto-Detection",
                "Results"
            ],
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
    elif page == "🚀 Benchmarking":
        benchmarking_page()
    elif page == "🔍 Auto-Detection":
        auto_detection_page()
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
        
    # Sample data initialization
    st.divider()
    st.subheader("📦 Quick Setup")
    
    col1, col2 = st.columns(2)
    with col1:
        if has_sample_data():
            st.success("✓ Sample data loaded")
            if st.button("View Sample Queries"):
                queries = get_sample_queries()
                st.write("**Sample queries you can try:**")
                for i, q in enumerate(queries[:5], 1):
                    st.write(f"{i}. {q}")
        else:
            st.info("No sample data found")
            if st.button("Load Sample Data", type="primary"):
                with st.spinner("Generating sample PDF..."):
                    pdf_path, queries = initialize_sample_data(force=True)
                    st.success(f"Sample data loaded: {pdf_path.name}")
                    st.rerun()
                    
    with col2:
        st.info("**Next Steps:**\n1. Load sample data or upload your own PDFs\n2. Go to 'PDF Management' to index documents\n3. Try queries in 'Query Interface'\n4. Run benchmarks and auto-detection")


def benchmarking_page():
    """Performance benchmarking page."""
    st.header("🚀 Performance Benchmarking")
    
    st.markdown("""
    Measure the actual performance overhead introduced by each observability SDK.
    This helps you understand the latency impact of instrumentation.
    """)
    
    if "orchestrator" in st.session_state:
        orchestrator = st.session_state.orchestrator
    else:
        orchestrator = get_orchestrator(None)
        
    active_providers = list(orchestrator.obs_manager.active_providers)
    
    if not active_providers:
        st.warning("No active providers. Please configure providers in the sidebar.")
        return
        
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        num_iterations = st.slider("Number of iterations per provider", 3, 20, 5)
    with col2:
        test_query = st.text_input("Test query", value="What is observability?")
        
    selected_providers = st.multiselect(
        "Providers to benchmark",
        active_providers,
        default=active_providers
    )
    
    if st.button("🚀 Run Benchmark", type="primary", disabled=not selected_providers):
        benchmarker = PerformanceBenchmarker(num_iterations=num_iterations)
        
        progress = st.progress(0)
        status = st.empty()
        
        # Run baseline
        status.text("Running baseline (no observability)...")
        baseline = benchmarker.benchmark_baseline(test_query)
        st.info(f"Baseline latency: {baseline:.2f}ms")
        
        # Run benchmarks
        for i, provider in enumerate(selected_providers):
            status.text(f"Benchmarking {provider}...")
            benchmarker.benchmark_provider(provider, test_query, baseline)
            progress.progress((i + 1) / len(selected_providers))
            
        status.text("Benchmark complete!")
        
        # Display results
        st.divider()
        st.subheader("Results")
        
        # Create comparison chart
        providers = []
        avg_latencies = []
        overheads = []
        
        for provider, result in benchmarker.results.items():
            providers.append(provider)
            avg_latencies.append(result.avg_latency_ms)
            overheads.append(result.overhead_percent)
            
        # Latency chart
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=providers,
            y=avg_latencies,
            name='Average Latency (ms)',
            marker_color='#F63366'
        ))
        fig1.add_hline(y=baseline, line_dash="dash", line_color="green", 
                      annotation_text="Baseline")
        fig1.update_layout(
            title="Average Latency by Provider",
            yaxis_title="Latency (ms)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Overhead chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=providers,
            y=overheads,
            name='Overhead %',
            marker_color='#FF6B6B'
        ))
        fig2.update_layout(
            title="SDK Overhead by Provider",
            yaxis_title="Overhead (%)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed table
        st.subheader("Detailed Metrics")
        data = []
        for provider, result in sorted(benchmarker.results.items(), key=lambda x: x[1].avg_latency_ms):
            data.append({
                "Provider": provider,
                "Avg (ms)": f"{result.avg_latency_ms:.2f}",
                "Median (ms)": f"{result.median_latency_ms:.2f}",
                "P95 (ms)": f"{result.p95_latency_ms:.2f}",
                "P99 (ms)": f"{result.p99_latency_ms:.2f}",
                "Overhead": f"{result.overhead_percent:.1f}%",
                "Error Rate": f"{result.error_rate*100:.1f}%"
            })
        st.dataframe(data, use_container_width=True, hide_index=True)
        
        # Export
        if st.button("💾 Export Results"):
            output_path = Path("results/benchmark_results.json")
            benchmarker.export_results(output_path)
            st.success(f"Results exported to {output_path}")


def auto_detection_page():
    """Automatic capability detection page."""
    st.header("🔍 Automatic Capability Detection")
    
    st.markdown("""
    Automatically test each provider to detect supported features.
    This saves time by auto-filling the comparison matrix based on actual behavior.
    """)
    
    if "orchestrator" in st.session_state:
        orchestrator = st.session_state.orchestrator
    else:
        orchestrator = get_orchestrator(None)
        
    active_providers = list(orchestrator.obs_manager.active_providers)
    
    if not active_providers:
        st.warning("No active providers. Please configure providers in the sidebar.")
        return
        
    st.subheader("Configuration")
    
    selected_providers = st.multiselect(
        "Providers to test",
        active_providers,
        default=active_providers[:3] if len(active_providers) > 3 else active_providers
    )
    
    col1, col2 = st.columns(2)
    with col1:
        auto_fill = st.checkbox("Auto-fill comparison matrix", value=True)
    with col2:
        confidence_threshold = st.slider("Confidence threshold", 0.5, 1.0, 0.7, 0.1)
        
    if st.button("🔍 Detect Capabilities", type="primary", disabled=not selected_providers):
        detector = CapabilityDetector()
        
        progress = st.progress(0)
        status = st.empty()
        
        # Run detection
        for i, provider in enumerate(selected_providers):
            status.text(f"Testing {provider}...")
            detector.detect_all_capabilities(provider)
            progress.progress((i + 1) / len(selected_providers))
            
        status.text("Detection complete!")
        
        # Display results
        st.divider()
        st.subheader("Detection Results")
        
        for provider, results in detector.test_results.items():
            with st.expander(f"**{provider.upper()}**", expanded=True):
                for result in results:
                    col1, col2, col3 = st.columns([2, 1, 3])
                    with col1:
                        status_icon = "✅" if result.supported else "❌"
                        st.write(f"{status_icon} **{result.criterion}**")
                    with col2:
                        confidence_color = "green" if result.confidence >= 0.8 else "orange" if result.confidence >= 0.6 else "red"
                        st.markdown(f":{confidence_color}[{result.confidence:.0%}]")
                    with col3:
                        st.caption(result.evidence)
                        
        # Auto-fill matrix
        if auto_fill:
            st.divider()
            st.subheader("Auto-Fill Comparison Matrix")
            
            matrix_path = Path("results/comparison_matrix.json")
            if matrix_path.exists():
                matrix = ComparisonMatrix()
                matrix.import_json(matrix_path)
            else:
                matrix = ComparisonMatrix()
                
            # Update matrix
            updated_count = 0
            for provider, results in detector.test_results.items():
                for result in results:
                    if result.confidence >= confidence_threshold:
                        matrix.set_score(
                            provider,
                            result.criterion,
                            result.supported,
                            notes=f"Auto-detected (confidence: {result.confidence:.0%}). {result.evidence}"
                        )
                        updated_count += 1
                        
            st.success(f"Updated {updated_count} matrix entries")
            
            if st.button("💾 Save Matrix"):
                matrix_path.parent.mkdir(parents=True, exist_ok=True)
                matrix.export_json(matrix_path)
                st.success(f"Matrix saved to {matrix_path}")
                
        # Export detection results
        if st.button("💾 Export Detection Results"):
            output_path = Path("results/capability_detection.json")
            detector.export_results(output_path)
            st.success(f"Results exported to {output_path}")



if __name__ == "__main__":
    main()

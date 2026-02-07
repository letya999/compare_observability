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


def sync_state():
    """Sync Streamlit session state with browser localStorage via custom component."""
    # Define keys to persist across browser sessions
    PERSISTENT_KEYS = {
        "active_page": "Quick Start",
        "selected_providers": config.observability_providers or [],
        "messages": [], # Chat history
        "benchmark_results": None,
        "detection_results": None,
        "scenario_results": None,
    }

    # Initialize session state with defaults
    for key, default in PERSISTENT_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # JS Bridge for LocalStorage
    # This component handles bidirectional sync between Python and browser
    from streamlit.components.v1 import html
    
    # Get current values to push to JS
    providers_json = json.dumps(st.session_state.get("selected_providers", []))
    page_json = json.dumps(st.session_state.get("active_page", "Quick Start"))
    
    # JS code that communicates with parents
    js_code = f"""
    <div id="bridge-root"></div>
    <script>
    // Helper to send data to Streamlit
    function sendToStreamlit(data) {{
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            value: data
        }}, '*');
    }}

    // When component loads, try to read from localstore
    const savedState = localStorage.getItem('rag_explorer_state');
    if (savedState) {{
        console.log('Found saved state in localStorage');
        // We use a button or some mechanism to notify Python if needed, 
        // but for now we'll just handle it via the component value
        sendToStreamlit(JSON.parse(savedState));
    }}

    // Listen for updates from Python (passed via attributes or reload)
    // We update localstore whenever Python sends new values
    const currentState = {{
        selected_providers: {providers_json},
        active_page: {page_json}
    }};
    if (currentState.selected_providers.length > 0) {{
        localStorage.setItem('rag_explorer_state', JSON.stringify(currentState));
    }}
    </script>
    """
    
    # Render the bridge and capture its value
    bridge_value = html(js_code, height=0)
    
    # Handle incoming state from LocalStorage
    if bridge_value and isinstance(bridge_value, dict):
        if "selected_providers" in bridge_value and not st.session_state.get("ls_synced"):
            st.session_state.selected_providers = bridge_value["selected_providers"]
            st.session_state.active_page = bridge_value.get("active_page", "Quick Start")
            st.session_state.ls_synced = True
            st.rerun()

def main():
    # Sync with LocalStorage
    sync_state()
    
    # Initialize Persistent State (Fallback)
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Quick Start"
    if "selected_providers" not in st.session_state:
        st.session_state.selected_providers = config.observability_providers or []

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Provider selection
        st.subheader("Observability Providers")
        available_providers = list(PLATFORMS.keys())
        
        # Use session state for multiselect default
        selected_providers = st.multiselect(
            "Active providers",
            available_providers,
            default=st.session_state.selected_providers,
            key="providers_input",
            help="Select which observability platforms to send traces to",
        )
        
        # Update session state when changed
        if selected_providers != st.session_state.selected_providers:
            st.session_state.selected_providers = selected_providers
            st.rerun()

        if st.button("🔄 Reload & Reset", help="Clear cache and reload configuration"):
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()

        # Display provider status
        if st.session_state.selected_providers:
            st.write("**Status:**")
            orchestrator = get_orchestrator(st.session_state.selected_providers)
            st.session_state.orchestrator = orchestrator
            
            active_count = 0
            for provider in st.session_state.selected_providers:
                is_active = provider in orchestrator.obs_manager.active_providers
                status = "✅" if is_active else "❌"
                if is_active: active_count += 1
                st.write(f"{status} {provider}")
                
                if not is_active:
                    init_errors = getattr(orchestrator.obs_manager, 'init_errors', {}) 
                    error_msg = init_errors.get(provider, "Initialization failed.")
                    st.caption(f":red[{error_msg}]")
                
            if active_count == 0 and st.session_state.selected_providers:
                st.error("No providers active!")

        st.divider()

        # Navigation
        pages = [
            "Quick Start", 
            "Chat Interface", 
            "PDF Management", 
            "Test Scenarios", 
            "Comparison Matrix",
            "🚀 Benchmarking",
            "🔍 Auto-Detection",
            "Results"
        ]
        
        # Find index of current page to keep it selected
        try:
            current_index = pages.index(st.session_state.active_page)
        except ValueError:
            current_index = 0

        page = st.radio(
            "Navigation",
            pages,
            index=current_index,
            key="nav_radio"
        )
        
        # Update persistent page state
        if page != st.session_state.active_page:
            st.session_state.active_page = page
            # No rerun needed here as Streamlit handles radio state, 
            # but we update the tracking variable.

    # Main content dispatch
    if st.session_state.active_page == "Quick Start":
        quick_start_page()
    elif st.session_state.active_page == "Chat Interface":
        chat_page()
    elif st.session_state.active_page == "PDF Management":
        pdf_management_page()
    elif st.session_state.active_page == "Test Scenarios":
        scenarios_page()
    elif st.session_state.active_page == "Comparison Matrix":
        matrix_page()
    elif st.session_state.active_page == "🚀 Benchmarking":
        benchmarking_page()
    elif st.session_state.active_page == "🔍 Auto-Detection":
        auto_detection_page()
    else:
        results_page()


def chat_page():
    """Modern Chat interface for PDF Knowledge Explorer."""
    st.header("Chat with Knowledge Base")

    # Initialize orchestrator
    if "orchestrator" in st.session_state:
        orchestrator = st.session_state.orchestrator
    else:
        orchestrator = get_orchestrator(st.session_state.get("selected_providers", []))
        st.session_state.orchestrator = orchestrator

    # Initialize chat history (Persistent in session_state)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize active result for inspection (lasts as long as the page is open)
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # Stats Summary in a single row
    stats = orchestrator.get_stats()
    cols = st.columns(4)
    cols[0].metric("Chunks", stats["vector_store"]["total_chunks"])
    cols[1].metric("Providers", len(orchestrator.obs_manager.active_providers))
    cols[2].metric("Model", stats["config"]["llm_model"])
    if st.session_state.last_result:
        cols[3].metric("Last Latency", f"{st.session_state.last_result.total_latency_ms:.0f}ms")
    else:
        cols[3].metric("Last Latency", "-")

    st.divider()

    # Create two columns: Chat and Inspector
    chat_col, inspect_col = st.columns([2, 1])

    with chat_col:
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "trace_urls" in message and message["trace_urls"]:
                    cols = st.columns(len(message["trace_urls"]))
                    for i, (provider, url) in enumerate(message["trace_urls"].items()):
                        with cols[i]:
                            st.caption(f"[View in {provider}]({url})")

        # Chat input
        if prompt := st.chat_input("What do you want to know?"):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # We'll use streaming by default for chat feel
                try:
                    # Collect result outside of generator loop
                    final_result = None
                    
                    # Logic to pass settings (would be nice to have them in a small settings popover or sidebar)
                    # For now using defaults
                    gen = orchestrator.query(prompt, history=st.session_state.messages[:-1], stream=True)
                    
                    for chunk in gen:
                        if isinstance(chunk, str):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        else:
                            final_result = chunk
                    
                    response_placeholder.markdown(full_response)
                    
                    # Add assistant message to history
                    msg_data = {
                        "role": "assistant", 
                        "content": full_response,
                        "trace_urls": getattr(final_result, "trace_urls", {})
                    }
                    st.session_state.messages.append(msg_data)
                    st.session_state.last_result = final_result
                    
                    # Rerun to show trace links properly in the history view (Streamlit limitation on nested updates)
                    st.rerun()

                except Exception as e:
                    logger.error("Chat query failed", exc_info=True)
                    st.error(f"Error: {e}")

    with inspect_col:
        st.subheader("Trace Inspector")
        if st.session_state.last_result:
            result = st.session_state.last_result
            
            with st.container(border=True):
                st.write(f"**Query:** {result.query}")
                st.write(f"**Tokens:** {result.response.token_usage.get('total_tokens', 0)}")
                st.write(f"**Cost:** ${result.cost_estimate_usd:.5f}")
            
            # Step Latencies
            st.write("#### Pipeline Latencies")
            for step, latency in result.step_latencies.items():
                st.progress(min(latency / 5000, 1.0), text=f"{step}: {latency:.0f}ms")

            # Trace Links
            if result.trace_urls:
                st.write("#### External Traces")
                for provider, url in result.trace_urls.items():
                    st.link_button(f"🔍 {provider}", url, use_container_width=True)

            # Retrieval Details
            with st.expander("Retrieved Chunks", expanded=False):
                for i, chunk in enumerate(result.reranked_chunks):
                    st.markdown(f"**[{i+1}]** Score: {chunk.score:.3f} | Page: {chunk.chunk.page_number}")
                    st.caption(chunk.chunk.text[:300] + "...")
                    st.divider()
            
            # Concepts
            if result.concepts:
                with st.expander("Knowledge Graph", expanded=False):
                    for concept in result.concepts:
                        st.markdown(f"**{concept.source}** → {concept.relation_type} → **{concept.target}**")

            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_result = None
                st.rerun()
        else:
            st.info("Ask a question to see trace details here.")


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

        # Store in session state
        st.session_state.scenario_results = {
            "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "providers": st.session_state.selected_providers,
            "scenarios": [r.__dict__ for r in runner.results] # Simplified
        }

        # Export results
        output_path = runner.export_results()
        st.success(f"Results exported to: {output_path}")

        # Summary components
        runner.shutdown()
        st.rerun() # Refresh to show in results if navigating


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
    """View and analyze results from all modules."""
    st.header("Results & Reports")

    tabs = st.tabs(["Scenario Reports", "Performance Benchmarks", "Feature Detection"])

    with tabs[0]:
        st.subheader("Scenario Execution History")
        results_dir = Path("results")
        if not results_dir.exists():
            st.info("No results directory found.")
        else:
            result_files = list(results_dir.glob("scenario_results_*.json"))
            if not result_files:
                st.write("No saved results found. Run some scenarios.")
            else:
                selected_file = st.selectbox("Select result file", result_files, format_func=lambda x: x.name)
                if selected_file:
                    with open(selected_file) as f:
                        data = json.load(f)
                    st.write(f"**Run date:** {data['run_date']}")
                    st.write(f"**Providers:** {', '.join(data['providers'])}")
                    st.divider()
                    for scenario in data["scenarios"]:
                        with st.expander(f"{scenario['name']} - {scenario['description']}"):
                            st.write(f"**Success:** {'✅' if scenario['success'] else '❌'}")
                            for check, passed in scenario["check_results"].items():
                                st.write(f"{'✅' if passed else '❌'} {check}")

    with tabs[1]:
        st.subheader("Last Benchmark Run")
        if "benchmark_results" in st.session_state and st.session_state.benchmark_results:
            results = st.session_state.benchmark_results
            baseline = results.get("baseline", 0)
            data = results.get("data", [])
            
            if data:
                # Recreate chart 
                providers = [d["Provider"] for d in data]
                latencies = [float(d["Avg (ms)"]) for d in data]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=providers, y=latencies, marker_color='#F63366'))
                fig.add_hline(y=baseline, line_dash="dash", line_color="green", annotation_text="Baseline")
                fig.update_layout(title="Average Latency Comparison", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(data, use_container_width=True, hide_index=True)
            else:
                st.info("No benchmark data in memory.")
        else:
            st.info("Run a benchmark on the 'Benchmarking' page to see results here.")

    with tabs[2]:
        st.subheader("Capability Detection Summary")
        if "detection_results" in st.session_state and st.session_state.detection_results:
            results = st.session_state.detection_results
            for provider, findings in results.items():
                with st.expander(f"**{provider.upper()}** Detected Features"):
                    for res in findings:
                        st.write(f"{'✅' if res['supported'] else '❌'} {res['criterion']} ({res['confidence']:.0%})")
        else:
            st.info("Run auto-detection on the 'Auto-Detection' page to see results here.")


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
        
        # PERSIST RESULTS
        st.session_state.benchmark_results = {
            "baseline": baseline,
            "data": data,
            "raw_results": {p: r.__dict__ for p, r in benchmarker.results.items()} # For deeper analysis if needed
        }
        
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
                        
        # PERSIST RESULTS
        st.session_state.detection_results = {
            provider: [r.__dict__ for r in results] 
            for provider, results in detector.test_results.items()
        }
                        
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

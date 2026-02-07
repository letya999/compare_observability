"""Quick start script for the observability comparison platform.

This script helps new users get started quickly by:
1. Checking dependencies
2. Loading sample data
3. Indexing the sample PDF
4. Running a test query
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import logger
from src.config import config
from src.utils.sample_data import initialize_sample_data, get_sample_queries
from src.pipeline.traced_orchestrator import TracedRAGOrchestrator


def main():
    """Run quick start setup."""
    print("="*80)
    print("OBSERVABILITY PLATFORM COMPARISON - QUICK START")
    print("="*80)
    print()
    
    # Step 1: Initialize sample data
    print("Step 1: Loading sample data...")
    try:
        pdf_path, queries = initialize_sample_data(force=False)
        print(f"✓ Sample PDF ready: {pdf_path}")
        print(f"✓ {len(queries)} sample queries available")
    except Exception as e:
        logger.error(f"Failed to initialize sample data: {e}")
        print(f"✗ Error loading sample data: {e}")
        return
        
    print()
    
    # Step 2: Initialize orchestrator
    print("Step 2: Initializing RAG orchestrator...")
    try:
        orchestrator = TracedRAGOrchestrator()
        active_providers = list(orchestrator.obs_manager.active_providers)
        print(f"✓ Orchestrator initialized")
        print(f"✓ Active observability providers: {', '.join(active_providers) if active_providers else 'None'}")
        
        if not active_providers:
            print("\n⚠️  Warning: No observability providers are active.")
            print("   Add API keys to .env file to enable provider tracking.")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        print(f"✗ Error initializing orchestrator: {e}")
        return
        
    print()
    
    # Step 3: Index sample PDF
    print("Step 3: Indexing sample PDF...")
    try:
        stats = orchestrator.get_stats()
        current_chunks = stats["vector_store"]["total_chunks"]
        
        if current_chunks > 0:
            print(f"✓ Index already contains {current_chunks} chunks")
            reindex = input("  Re-index sample PDF? (y/N): ").strip().lower()
            if reindex != 'y':
                print("  Skipping indexing")
            else:
                doc, chunk_count = orchestrator.ingest_pdf(pdf_path)
                print(f"✓ Indexed {chunk_count} chunks from {pdf_path.name}")
        else:
            doc, chunk_count = orchestrator.ingest_pdf(pdf_path)
            print(f"✓ Indexed {chunk_count} chunks from {pdf_path.name}")
    except Exception as e:
        logger.error(f"Failed to index PDF: {e}")
        print(f"✗ Error indexing PDF: {e}")
        return
        
    print()
    
    # Step 4: Run test query
    print("Step 4: Running test query...")
    test_query = queries[0]
    print(f"  Query: '{test_query}'")
    
    try:
        result = orchestrator.query(test_query, stream=False, skip_graph_extraction=True)
        print(f"✓ Query completed in {result.total_latency_ms:.0f}ms")
        print(f"✓ Retrieved {len(result.retrieved_chunks)} chunks")
        print(f"✓ Used {result.response.token_usage.get('total_tokens', 0)} tokens")
        
        if active_providers:
            print(f"\n  Trace URLs:")
            trace_urls = getattr(result, 'trace_urls', {})
            for provider, url in trace_urls.items():
                print(f"    - {provider}: {url}")
                
        print(f"\n  Answer preview:")
        answer = result.response.answer
        preview = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"  {preview}")
        
    except Exception as e:
        logger.error(f"Failed to run test query: {e}")
        print(f"✗ Error running query: {e}")
        return
        
    print()
    print("="*80)
    print("QUICK START COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Run the Streamlit UI: streamlit run app.py")
    print("2. Try the sample queries in 'Query Interface'")
    print("3. Run benchmarks in '🚀 Benchmarking' page")
    print("4. Auto-detect capabilities in '🔍 Auto-Detection' page")
    print("5. Fill out the comparison matrix")
    print()
    print("For more information, see docs/IMPROVEMENTS.md")
    print()


if __name__ == "__main__":
    main()

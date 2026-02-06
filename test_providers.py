"""Test script to check provider initialization."""
from dotenv import load_dotenv
load_dotenv()

import time
from src.observability import ObservabilityManager
from src.observability.base import SpanType

print("="*60)
print("PROVIDER INITIALIZATION TEST")
print("="*60)
print()

# Initialize
manager = ObservabilityManager()

print(f"Active providers: {list(manager.active_providers.keys())}")
print(f"Total: {len(manager.active_providers)}/5 expected")
print()

if len(manager.active_providers) == 0:
    print("ERROR: No providers initialized!")
    exit(1)

# Test trace
print("Creating test trace...")
try:
    with manager.trace("test_all_providers", inputs={"test": "data"}) as trace:
        print(f"Trace IDs:")
        for provider, trace_id in trace.trace_ids.items():
            print(f"  ✓ {provider}: {trace_id[:30]}...")
        
        # Create a span
        with manager.span("test_span", SpanType.LLM, trace, inputs={"prompt": "test"}) as span:
            # Set outputs
            for pname, pspan in span.provider_spans.items():
                pspan.outputs = {"response": "Test response"}
            print(f"\nSpan created in {len(span.provider_spans)} providers")
    
    print("\n✓ Trace completed successfully")
except Exception as e:
    print(f"\n✗ Error during trace: {e}")
    import traceback
    traceback.print_exc()

# Flush
print("\nFlushing data (waiting 5s)...")
time.sleep(5)
manager.shutdown()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nCheck your dashboards:")
print("  - Langfuse: https://cloud.langfuse.com")
print("  - LangSmith: https://smith.langchain.com")
print("  - Opik: https://www.comet.com/opik")
print("  - Arize Phoenix: https://app.phoenix.arize.com")
print("  - Braintrust: https://www.braintrust.dev")

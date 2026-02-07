from src.evaluations.capability_detector import CapabilityDetector
from src.config import config

# Set dummy key for LangSmith to avoid initialization error if not set
import os
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = "ls__dummy_key_for_test"

detector = CapabilityDetector()
# We can't really test a provider without API keys, but we can check if the methods run structurally
# The actual providers will fail initialization and thus the tests will return "supported=False" or fail gracefully

print("Testing CapabilityDetector structure...")

# Mocking TracedRAGOrchestrator to avoid need for real providers in this quick verification
from unittest.mock import MagicMock, patch
import sys

# Patch TracedRAGOrchestrator
with patch('src.evaluations.capability_detector.TracedRAGOrchestrator') as MockOrchestrator:
    # Setup mock
    mock_instance = MockOrchestrator.return_value
    mock_instance.query.return_value.response.token_usage = {"total_tokens": 100}
    mock_instance.query.return_value.response.answer = "Test answer"
    
    # Run tests
    print("\nRunning test_cost_calculation...")
    result = detector.test_cost_calculation("dummy_provider")
    print(f"Result: {result.supported}, Confidence: {result.confidence}, Evidence: {result.evidence}")
    
    print("\nRunning test_nested_spans...")
    result = detector.test_nested_spans("dummy_provider")
    print(f"Result: {result.supported}, Confidence: {result.confidence}, Evidence: {result.evidence}")
    
    print("\nRunning test_error_handling...")
    # This one uses context managers, so it's harder to check output without deeper mocking,
    # but we just want to ensure code doesn't crash conformally
    try:
        result = detector.test_error_handling("dummy_provider")
        print(f"Result: {result.supported}, Confidence: {result.confidence}, Evidence: {result.evidence}")
    except Exception as e:
        print(f"Error in test_error_handling: {e}")

print("\nVerification complete.")

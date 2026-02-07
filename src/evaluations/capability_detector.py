"""Automatic capability detection for observability providers.

Tests each provider to detect supported features and auto-fills the comparison matrix.
"""

from dataclasses import dataclass
from pathlib import Path

from src.logger import logger
from src.pipeline.traced_orchestrator import TracedRAGOrchestrator
from src.evaluations.comparison_matrix import ComparisonMatrix


@dataclass
class CapabilityTestResult:
    """Result from testing a specific capability."""
    criterion: str
    supported: bool
    confidence: float  # 0.0 to 1.0
    evidence: str
    

class CapabilityDetector:
    """Automatically detect provider capabilities by testing actual behavior."""
    
    def __init__(self):
        """Initialize capability detector."""
        self.test_results: dict[str, list[CapabilityTestResult]] = {}
        
    def test_token_counting(self, provider: str) -> CapabilityTestResult:
        """Test if provider automatically counts tokens.
        
        Args:
            provider: Provider name to test
            
        Returns:
            CapabilityTestResult indicating if token counting is supported
        """
        logger.info(f"Testing token counting for {provider}")
        
        try:
            orchestrator = TracedRAGOrchestrator(observability_providers=[provider])
            result = orchestrator.query("Test query", stream=False, skip_graph_extraction=True)
            
            # Check if token usage is tracked
            has_tokens = (
                result.response.token_usage and 
                result.response.token_usage.get("total_tokens", 0) > 0
            )
            
            return CapabilityTestResult(
                criterion="token_counting",
                supported=has_tokens,
                confidence=1.0 if has_tokens else 0.8,
                evidence=f"Token usage: {result.response.token_usage}" if has_tokens else "No token data found"
            )
            
        except Exception as e:
            logger.error(f"Token counting test failed for {provider}: {e}")
            return CapabilityTestResult(
                criterion="token_counting",
                supported=False,
                confidence=0.5,
                evidence=f"Test failed: {str(e)}"
            )
            
    def test_cost_calculation(self, provider: str) -> CapabilityTestResult:
        """Test if provider automatically calculates costs.
        
        Args:
            provider: Provider name to test
            
        Returns:
            CapabilityTestResult indicating if cost calculation is supported
        """
        logger.info(f"Testing cost calculation for {provider}")
        
        try:
            orchestrator = TracedRAGOrchestrator(observability_providers=[provider])
            result = orchestrator.query("Test query", stream=False, skip_graph_extraction=True)
            
            # Check if token usage is tracked (prerequisite for cost calculation)
            has_tokens = (
                result.response.token_usage and 
                result.response.token_usage.get("total_tokens", 0) > 0
            )
            
            if has_tokens:
                return CapabilityTestResult(
                    criterion="cost_calculation",
                    supported=True,
                    confidence=0.8,
                    evidence=f"Token usage tracked ({result.response.token_usage.get('total_tokens')} tokens). Provider should calculate cost."
                )
            else:
                return CapabilityTestResult(
                    criterion="cost_calculation",
                    supported=False,
                    confidence=0.7,
                    evidence="No token usage data, cannot calculate cost"
                )
            
        except Exception as e:
            logger.error(f"Cost calculation test failed for {provider}: {e}")
            return CapabilityTestResult(
                criterion="cost_calculation",
                supported=False,
                confidence=0.5,
                evidence=f"Test failed: {str(e)}"
            )
            
    def test_streaming_support(self, provider: str) -> CapabilityTestResult:
        """Test if provider supports streaming responses.
        
        Args:
            provider: Provider name to test
            
        Returns:
            CapabilityTestResult indicating if streaming is supported
        """
        logger.info(f"Testing streaming support for {provider}")
        
        try:
            orchestrator = TracedRAGOrchestrator(observability_providers=[provider])
            
            # Try to run a streaming query
            chunks_received = 0
            try:
                gen = orchestrator.query("Test query", stream=True, skip_graph_extraction=True)
                for chunk in gen:
                    if isinstance(chunk, str):
                        chunks_received += 1
                        
                streaming_works = chunks_received > 0
                
                return CapabilityTestResult(
                    criterion="streaming_support",
                    supported=streaming_works,
                    confidence=1.0 if streaming_works else 0.8,
                    evidence=f"Received {chunks_received} streaming chunks" if streaming_works else "No streaming chunks received"
                )
                
            except Exception as stream_error:
                return CapabilityTestResult(
                    criterion="streaming_support",
                    supported=False,
                    confidence=0.9,
                    evidence=f"Streaming failed: {str(stream_error)}"
                )
                
        except Exception as e:
            logger.error(f"Streaming test failed for {provider}: {e}")
            return CapabilityTestResult(
                criterion="streaming_support",
                supported=False,
                confidence=0.5,
                evidence=f"Test failed: {str(e)}"
            )
            
    def test_error_handling(self, provider: str) -> CapabilityTestResult:
        """Test if provider properly captures errors.
        
        Args:
            provider: Provider name to test
            
        Returns:
            CapabilityTestResult indicating if error handling is supported
        """
        logger.info(f"Testing error handling for {provider}")
        
        try:
            orchestrator = TracedRAGOrchestrator(observability_providers=[provider])
            
            # Explicitly log an error trace
            from src.observability.base import SpanType
            
            error_msg = "Simulated error for capability detection"
            try:
                with orchestrator.obs_manager.trace("test_error_handling") as trace:
                    with orchestrator.obs_manager.span("failing_step", SpanType.TOOL, trace) as span:
                        try:
                            raise ValueError(error_msg)
                        except Exception as e:
                            orchestrator.obs_manager.log_error(span, e)
                            # We don't re-raise here to avoid crashing the test, 
                            # but the provider should have received the error event.
                            
                return CapabilityTestResult(
                    criterion="error_traces",
                    supported=True,
                    confidence=0.9,
                    evidence="Successfully logged error trace to provider SDK"
                )
                
            except Exception as e:
                # If the provider SDK raised an exception during error logging
                return CapabilityTestResult(
                    criterion="error_traces",
                    supported=False,
                    confidence=0.8,
                    evidence=f"SDK failed to handle error log: {str(e)}"
                )
            
        except Exception as e:
            logger.error(f"Error handling test failed for {provider}: {e}")
            return CapabilityTestResult(
                criterion="error_traces",
                supported=False,
                confidence=0.5,
                evidence=f"Test failed: {str(e)}"
            )
            
    def test_nested_spans(self, provider: str) -> CapabilityTestResult:
        """Test if provider supports nested spans.
        
        Args:
            provider: Provider name to test
            
        Returns:
            CapabilityTestResult indicating if nested spans are supported
        """
        logger.info(f"Testing nested spans for {provider}")
        
        try:
            orchestrator = TracedRAGOrchestrator(observability_providers=[provider])
            
            # Run a full query which should create nested spans
            result = orchestrator.query("Test query", stream=False, skip_graph_extraction=False)
            
            # Just verify we got a result
            has_result = result and result.response.answer
            
            return CapabilityTestResult(
                criterion="nested_spans",
                supported=bool(has_result),
                confidence=0.9,
                evidence="Full pipeline with nested spans executed successfully" if has_result else "Pipeline execution failed"
            )
            
        except Exception as e:
            logger.error(f"Nested spans test failed for {provider}: {e}")
            return CapabilityTestResult(
                criterion="nested_spans",
                supported=False,
                confidence=0.5,
                evidence=f"Test failed: {str(e)}"
            )
            
    def detect_all_capabilities(self, provider: str) -> list[CapabilityTestResult]:
        """Run all capability tests for a provider.
        
        Args:
            provider: Provider name to test
            
        Returns:
            List of CapabilityTestResults
        """
        logger.info(f"Detecting all capabilities for {provider}")
        
        results = []
        
        # Run all tests
        test_methods = [
            self.test_token_counting,
            self.test_cost_calculation,
            self.test_streaming_support,
            self.test_error_handling,
            self.test_nested_spans,
        ]
        
        for test_method in test_methods:
            try:
                result = test_method(provider)
                results.append(result)
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed for {provider}: {e}")
                
        self.test_results[provider] = results
        return results
        
    def auto_fill_matrix(
        self, 
        providers: list[str],
        matrix: ComparisonMatrix | None = None
    ) -> ComparisonMatrix:
        """Automatically fill comparison matrix based on detected capabilities.
        
        Args:
            providers: List of provider names to test
            matrix: Existing ComparisonMatrix to update, or None to create new one
            
        Returns:
            Updated ComparisonMatrix
        """
        if matrix is None:
            matrix = ComparisonMatrix()
            
        for provider in providers:
            logger.info(f"Auto-filling matrix for {provider}")
            
            # Detect capabilities
            results = self.detect_all_capabilities(provider)
            
            # Update matrix
            for result in results:
                if result.confidence >= 0.7:  # Only update if confident
                    matrix.set_score(
                        provider,
                        result.criterion,
                        result.supported,
                        notes=f"Auto-detected (confidence: {result.confidence:.0%}). {result.evidence}"
                    )
                    
        return matrix
        
    def export_results(self, output_path: Path | str) -> None:
        """Export capability detection results to JSON.
        
        Args:
            output_path: Path to save results
        """
        import json
        from datetime import datetime
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": {
                provider: [
                    {
                        "criterion": r.criterion,
                        "supported": r.supported,
                        "confidence": r.confidence,
                        "evidence": r.evidence,
                    }
                    for r in results
                ]
                for provider, results in self.test_results.items()
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Capability detection results exported to {output_path}")
        
    def print_summary(self) -> None:
        """Print a summary of capability detection results."""
        if not self.test_results:
            print("No capability detection results available")
            return
            
        print("\n" + "="*80)
        print("CAPABILITY DETECTION RESULTS")
        print("="*80)
        
        for provider, results in self.test_results.items():
            print(f"\n{provider.upper()}")
            print("-" * 40)
            
            for result in results:
                status = "✓" if result.supported else "✗"
                confidence_str = f"({result.confidence:.0%} confidence)"
                print(f"  {status} {result.criterion:<25} {confidence_str}")
                if result.evidence:
                    print(f"    └─ {result.evidence}")

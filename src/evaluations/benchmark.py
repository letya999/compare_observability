"""Performance benchmarking for observability providers.

Measures the actual overhead introduced by each observability SDK.
"""

import time
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.logger import logger
from src.pipeline.traced_orchestrator import TracedRAGOrchestrator


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    provider: str
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    overhead_percent: float
    memory_mb: float | None = None
    error_rate: float = 0.0


class PerformanceBenchmarker:
    """Benchmark observability provider performance impact."""
    
    def __init__(self, num_iterations: int = 10):
        """Initialize benchmarker.
        
        Args:
            num_iterations: Number of times to run each test
        """
        self.num_iterations = num_iterations
        self.results: dict[str, BenchmarkResult] = {}
        
    def benchmark_provider(
        self, 
        provider: str, 
        query: str = "What is observability?",
        baseline_latency: float | None = None
    ) -> BenchmarkResult:
        """Benchmark a single provider.
        
        Args:
            provider: Provider name to benchmark
            query: Test query to run
            baseline_latency: Baseline latency without observability (for overhead calc)
            
        Returns:
            BenchmarkResult with performance metrics
        """
        logger.info(f"Benchmarking provider: {provider}")
        
        # Initialize orchestrator with only this provider
        orchestrator = TracedRAGOrchestrator(observability_providers=[provider])
        
        latencies = []
        errors = 0
        
        for i in range(self.num_iterations):
            try:
                start = time.perf_counter()
                result = orchestrator.query(query, stream=False, skip_graph_extraction=True)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed for {provider}: {e}")
                errors += 1
                
        if not latencies:
            logger.warning(f"No successful iterations for {provider}")
            return BenchmarkResult(
                provider=provider,
                avg_latency_ms=0,
                median_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                overhead_percent=0,
                error_rate=1.0
            )
            
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else sorted_latencies[-1]
        p99_latency = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else sorted_latencies[-1]
        
        # Calculate overhead
        overhead_percent = 0.0
        if baseline_latency and baseline_latency > 0:
            overhead_percent = ((avg_latency - baseline_latency) / baseline_latency) * 100
            
        error_rate = errors / self.num_iterations
        
        result = BenchmarkResult(
            provider=provider,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            overhead_percent=overhead_percent,
            error_rate=error_rate
        )
        
        self.results[provider] = result
        return result
        
    def benchmark_baseline(self, query: str = "What is observability?") -> float:
        """Benchmark without any observability providers.
        
        Args:
            query: Test query to run
            
        Returns:
            Average baseline latency in milliseconds
        """
        logger.info("Benchmarking baseline (no observability)")
        
        # Initialize orchestrator with no providers
        orchestrator = TracedRAGOrchestrator(observability_providers=[])
        
        latencies = []
        
        for i in range(self.num_iterations):
            try:
                start = time.perf_counter()
                result = orchestrator.query(query, stream=False, skip_graph_extraction=True)
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                logger.error(f"Baseline benchmark iteration {i} failed: {e}")
                
        if not latencies:
            logger.warning("No successful baseline iterations")
            return 0.0
            
        return statistics.mean(latencies)
        
    def benchmark_all(
        self, 
        providers: list[str],
        query: str = "What is observability?"
    ) -> dict[str, BenchmarkResult]:
        """Benchmark all providers.
        
        Args:
            providers: List of provider names to benchmark
            query: Test query to run
            
        Returns:
            Dictionary mapping provider names to BenchmarkResults
        """
        # First get baseline
        baseline = self.benchmark_baseline(query)
        logger.info(f"Baseline latency: {baseline:.2f}ms")
        
        # Benchmark each provider
        for provider in providers:
            self.benchmark_provider(provider, query, baseline)
            
        return self.results
        
    def export_results(self, output_path: Path | str) -> None:
        """Export benchmark results to JSON.
        
        Args:
            output_path: Path to save results
        """
        import json
        from datetime import datetime
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "num_iterations": self.num_iterations,
            "results": {
                provider: {
                    "avg_latency_ms": result.avg_latency_ms,
                    "median_latency_ms": result.median_latency_ms,
                    "p95_latency_ms": result.p95_latency_ms,
                    "p99_latency_ms": result.p99_latency_ms,
                    "overhead_percent": result.overhead_percent,
                    "error_rate": result.error_rate,
                }
                for provider, result in self.results.items()
            }
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Benchmark results exported to {output_path}")
        
    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        if not self.results:
            print("No benchmark results available")
            return
            
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        print(f"Iterations per provider: {self.num_iterations}\n")
        
        # Sort by average latency
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].avg_latency_ms
        )
        
        print(f"{'Provider':<15} {'Avg (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12} {'Overhead %':<12} {'Errors':<8}")
        print("-" * 80)
        
        for provider, result in sorted_results:
            print(
                f"{provider:<15} "
                f"{result.avg_latency_ms:<12.2f} "
                f"{result.median_latency_ms:<12.2f} "
                f"{result.p95_latency_ms:<12.2f} "
                f"{result.overhead_percent:<12.2f} "
                f"{result.error_rate*100:<8.1f}%"
            )

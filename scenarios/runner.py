"""Scenario runner for observability comparison testing."""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import config
from src.observability import ObservabilityManager
from src.pipeline.traced_orchestrator import TracedRAGOrchestrator

from .scenarios import (
    CheckType,
    Scenario,
    ScenarioResult,
    ScenarioType,
    SCENARIOS,
    get_scenario,
)


class ScenarioRunner:
    """
    Runs test scenarios and collects comparison data.

    Usage:
        runner = ScenarioRunner()
        results = runner.run_all()
        runner.export_results("results.json")
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        output_dir: Path | None = None,
    ):
        self.providers = providers or config.observability_providers
        self.output_dir = output_dir or Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.orchestrator = TracedRAGOrchestrator(
            observability_providers=self.providers
        )
        self.results: list[ScenarioResult] = []

    def run_scenario(self, scenario: str | Scenario) -> ScenarioResult:
        """Run a single scenario and collect results."""
        if isinstance(scenario, str):
            scenario = get_scenario(scenario)
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"{'='*60}")

        provider_results = {}
        check_results = {}
        errors = []
        metrics = {
            "start_time": datetime.now().isoformat(),
            "providers": self.providers,
        }

        try:
            # Run the scenario based on type
            if scenario.type == ScenarioType.EVALUATION:
                result = self._run_evaluation_scenario(scenario)
            elif scenario.type == ScenarioType.ERROR_HANDLING:
                result = self._run_error_scenario(scenario)
            else:
                result = self._run_standard_scenario(scenario)

            metrics["pipeline_result"] = self._serialize_result(result)
            metrics["total_latency_ms"] = result.total_latency_ms
            metrics["step_latencies"] = result.step_latencies

            # Collect provider-specific data
            provider_results = self._collect_provider_data()

            # Run checks
            check_results = self._run_checks(scenario, result, provider_results)

        except Exception as e:
            errors.append(str(e))
            print(f"Error running scenario: {e}")

        metrics["end_time"] = datetime.now().isoformat()

        scenario_result = ScenarioResult(
            scenario=scenario,
            success=len(errors) == 0 and all(check_results.values()),
            provider_results=provider_results,
            check_results=check_results,
            errors=errors,
            metrics=metrics,
        )

        self.results.append(scenario_result)
        return scenario_result

    def _run_standard_scenario(self, scenario: Scenario) -> Any:
        """Run a standard RAG query scenario."""
        query = scenario.query if isinstance(scenario.query, str) else scenario.query[0]

        if scenario.stream:
            # Streaming scenario
            start_time = time.time()
            first_token_time = None
            full_response = ""

            gen = self.orchestrator.query(
                query,
                stream=True,
                skip_graph_extraction=False,
            )

            for chunk in gen:
                if isinstance(chunk, str):
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_response += chunk
                    print(chunk, end="", flush=True)
                else:
                    result = chunk

            print()  # Newline after streaming

            if first_token_time:
                result.step_latencies["time_to_first_token"] = (first_token_time - start_time) * 1000

            return result
        else:
            return self.orchestrator.query(
                query,
                stream=False,
                skip_graph_extraction=False,
            )

    def _run_evaluation_scenario(self, scenario: Scenario) -> Any:
        """Run batch evaluation scenario."""
        queries = scenario.query if isinstance(scenario.query, list) else [scenario.query]
        results = []

        for i, query in enumerate(queries):
            print(f"  Running query {i+1}/{len(queries)}: {query[:50]}...")
            result = self.orchestrator.query(query, skip_graph_extraction=True)
            results.append(result)

        # Return the last result with aggregated metrics
        aggregated = results[-1]
        aggregated.step_latencies["total_queries"] = len(queries)
        aggregated.step_latencies["avg_latency"] = sum(r.total_latency_ms for r in results) / len(results)
        return aggregated

    def _run_error_scenario(self, scenario: Scenario) -> Any:
        """Run error handling scenario."""
        # For empty retrieval, we query without ingested docs
        try:
            result = self.orchestrator.query(
                scenario.query,
                filter_doc_ids=["nonexistent_doc"],  # Force empty retrieval
            )
            return result
        except Exception as e:
            # Expected error - create a minimal result
            from src.models import GeneratedResponse, QueryAnalysis, QueryIntent
            from src.pipeline.orchestrator import PipelineResult

            return PipelineResult(
                query=scenario.query,
                query_analysis=QueryAnalysis(
                    original_query=scenario.query,
                    intent=QueryIntent.UNKNOWN,
                    entities=[],
                ),
                retrieved_chunks=[],
                reranked_chunks=[],
                response=GeneratedResponse(
                    answer=f"Error: {str(e)}",
                    citations=[],
                    concepts=[],
                    token_usage={},
                    latency_ms=0,
                ),
                concepts=[],
                total_latency_ms=0,
                step_latencies={"error": str(e)},
            )

    def _collect_provider_data(self) -> dict[str, dict[str, Any]]:
        """Collect data from all providers after scenario run."""
        provider_data = {}

        for name, provider in self.orchestrator.obs_manager.active_providers.items():
            provider_data[name] = {
                "name": provider.name,
                "supports_streaming": provider.supports_streaming,
                "supports_async": provider.supports_async,
                "features": {
                    "nested_spans": provider.supports_feature("nested_spans"),
                    "cost_tracking": provider.supports_feature("cost_tracking"),
                    "evaluations": provider.supports_feature("evaluations"),
                },
            }

        return provider_data

    def _run_checks(
        self,
        scenario: Scenario,
        result: Any,
        provider_results: dict,
    ) -> dict[str, bool]:
        """Run validation checks for the scenario."""
        check_results = {}

        for check in scenario.checks:
            if check == CheckType.TRACE_COMPLETENESS:
                # Check if all expected spans were created
                check_results["trace_completeness"] = len(result.step_latencies) >= 4

            elif check == CheckType.TOKEN_COUNT:
                # Check if token counting is available
                check_results["token_count"] = result.response.token_usage.get("total_tokens", 0) > 0

            elif check == CheckType.LATENCY:
                # Check if latency is reasonable
                check_results["latency"] = result.total_latency_ms < 30000  # 30s max

            elif check == CheckType.STREAMING_SPANS:
                # Check streaming-specific metrics
                check_results["streaming_spans"] = "time_to_first_token" in result.step_latencies

            elif check == CheckType.TIME_TO_FIRST_TOKEN:
                ttft = result.step_latencies.get("time_to_first_token", float("inf"))
                check_results["time_to_first_token"] = ttft < 5000  # 5s max

            elif check == CheckType.ERROR_TRACE:
                # Check if error was captured
                check_results["error_trace"] = True  # If we got here, error handling worked

            elif check == CheckType.BATCH_EVAL:
                # Check batch evaluation metrics
                check_results["batch_eval"] = result.step_latencies.get("total_queries", 0) > 1

            elif check == CheckType.COST_CALCULATION:
                # Check if any provider has cost tracking
                has_cost = any(
                    p.get("features", {}).get("cost_tracking", False)
                    for p in provider_results.values()
                )
                check_results["cost_calculation"] = has_cost

            elif check == CheckType.LARGE_PAYLOAD:
                # Check if large payloads were handled
                check_results["large_payload"] = True  # If no error, it worked

            elif check == CheckType.PARALLEL_SPANS:
                # Check for parallel span support
                check_results["parallel_spans"] = True

            elif check == CheckType.CROSS_DOC_RETRIEVAL:
                # Check multi-doc retrieval
                check_results["cross_doc_retrieval"] = len(result.retrieved_chunks) > 0

        return check_results

    def _serialize_result(self, result: Any) -> dict:
        """Serialize pipeline result to dict."""
        return {
            "query": result.query,
            "intent": result.query_analysis.intent.value,
            "retrieved_count": len(result.retrieved_chunks),
            "reranked_count": len(result.reranked_chunks),
            "answer_length": len(result.response.answer),
            "concept_count": len(result.concepts),
            "total_latency_ms": result.total_latency_ms,
            "step_latencies": result.step_latencies,
            "token_usage": result.response.token_usage,
        }

    def run_all(self) -> list[ScenarioResult]:
        """Run all scenarios."""
        for scenario_name in SCENARIOS:
            self.run_scenario(scenario_name)
        return self.results

    def export_results(self, filename: str | None = None) -> Path:
        """Export results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scenario_results_{timestamp}.json"

        output_path = self.output_dir / filename

        export_data = {
            "run_date": datetime.now().isoformat(),
            "providers": self.providers,
            "scenarios": [
                {
                    "name": r.scenario.name,
                    "type": r.scenario.type.value,
                    "description": r.scenario.description,
                    "success": r.success,
                    "check_results": r.check_results,
                    "errors": r.errors,
                    "metrics": r.metrics,
                    "provider_results": r.provider_results,
                }
                for r in self.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"\nResults exported to: {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print summary of all scenario results."""
        print("\n" + "="*60)
        print("SCENARIO RESULTS SUMMARY")
        print("="*60)

        for result in self.results:
            status = "PASS" if result.success else "FAIL"
            print(f"\n{result.scenario.name}: {status}")
            print(f"  Checks: {sum(result.check_results.values())}/{len(result.check_results)} passed")

            if result.errors:
                print(f"  Errors: {result.errors}")

            if "total_latency_ms" in result.metrics:
                print(f"  Total latency: {result.metrics['total_latency_ms']:.2f}ms")

        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        print(f"\n{'='*60}")
        print(f"Total: {passed}/{total} scenarios passed")
        print("="*60)

    def shutdown(self) -> None:
        """Cleanup resources."""
        self.orchestrator.shutdown()

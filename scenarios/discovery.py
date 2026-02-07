"""Dynamic scenario generation based on indexed content."""

import json
from typing import Any
from src.pipeline.orchestrator import RAGOrchestrator
from .scenarios import Scenario, ScenarioType, CheckType

class DiscoveryGenerator:
    """Generates test scenarios by analyzing the current document index."""

    def __init__(self, orchestrator: RAGOrchestrator):
        self.orchestrator = orchestrator

    def generate_scenarios(self, count: int = 3) -> list[Scenario]:
        """
        1. Fetch sample chunks from index.
        2. Use LLM to generate interesting questions.
        3. Wrap them into Scenario objects.
        """
        # Get sample content
        stats = self.orchestrator.retriever.get_collection_stats()
        if stats.get("count", 0) == 0:
            return []

        # Fetch some "random" chunks by querying for generic terms or empty string
        # Most vector stores return something even for empty queries if configured
        samples = self.orchestrator.retriever.collection.get(limit=10)
        content_sample = "\n---\n".join(samples.get("documents", []))

        prompt = f"""
You are a QA engineer testing a RAG (Retrieval-Augmented Generation) system. 
Below are samples from the currently indexed documents:

{content_sample}

Generate {count} diverse test scenarios to evaluate how well the system handles this specific content.
Each scenario should test a different aspect:
1. A simple fact retrieval.
2. A complex question requiring reasoning or linking multiple facts.
3. A summary or high-level extraction task.

Return ONLY a JSON array of objects with the following structure:
{{
    "name": "short_snake_case_name",
    "description": "Clear description of what this tests",
    "query": "The actual question to ask",
    "type": "simple_rag" | "multi_hop" | "long_context",
    "checks": ["trace_completeness", "token_count", "latency", "parallel_spans", "cross_doc_retrieval"]
}}
"""
        
        response = self.orchestrator.client.chat.completions.create(
            model=self.orchestrator.generator.model, # Using same model as generator
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        raw_scenarios = data.get("scenarios", data.get("results", []))
        if not raw_scenarios and isinstance(data, list):
            raw_scenarios = data
        elif not raw_scenarios and isinstance(data, dict):
            # Fallback if the LLM wrapped it differently
            for val in data.values():
                if isinstance(val, list):
                    raw_scenarios = val
                    break

        scenarios = []
        for s in raw_scenarios:
            scenarios.append(Scenario(
                name=s["name"],
                type=ScenarioType(s.get("type", "simple_rag")),
                description=s["description"],
                query=s["query"],
                expected_spans=5 if s.get("type") == "simple_rag" else 7,
                checks=[CheckType(c) for c in s.get("checks", ["trace_completeness", "token_count", "latency"])],
                metadata={"generated": True}
            ))
        
        return scenarios

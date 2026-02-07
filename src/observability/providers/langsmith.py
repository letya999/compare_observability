"""LangSmith observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class LangSmithProvider(ObservabilityProvider):
    """
    LangSmith integration.

    Features tested:
    - @traceable decorator / RunTree API
    - Nested runs
    - LangChain callbacks
    - Prompt hub integration
    - Evaluations
    
    Environment variables (new LANGSMITH_* preferred, LANGCHAIN_* still supported):
    - LANGSMITH_TRACING / LANGCHAIN_TRACING_V2: Enable tracing
    - LANGSMITH_API_KEY / LANGCHAIN_API_KEY: API key
    - LANGSMITH_PROJECT / LANGCHAIN_PROJECT: Project name
    - LANGSMITH_ENDPOINT: Custom endpoint (optional)
    """

    name = "langsmith"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.client = None
        # Check both new LANGSMITH_* and legacy LANGCHAIN_* prefixes
        self.project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "compare-observability")

    def initialize(self) -> bool:
        """Initialize LangSmith client."""
        # Check for API key with both naming conventions
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            print("[LangSmith] No API key found (LANGSMITH_API_KEY or LANGCHAIN_API_KEY)")
            return False
        
        # Ensure tracing is enabled
        tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true" or \
                          os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        if not tracing_enabled:
            print("[LangSmith] Warning: Tracing not enabled. Set LANGSMITH_TRACING=true")

        try:
            from langsmith import Client

            self.client = Client()
            # Test connection
            self.client.list_projects(limit=1)
            print(f"[LangSmith] Initialized successfully (project: {self.project_name})")
            return True
        except ImportError:
            print("[LangSmith] langsmith package not installed")
            return False
        except Exception as e:
            print(f"[LangSmith] Failed to connect: {e}")
            return False

    def shutdown(self) -> None:
        """Flush pending data."""
        if self.client:
            try:
                self.client.flush()
                print("[LangSmith] Flushed all pending runs")
            except Exception as e:
                print(f"[LangSmith] Error flushing: {e}")

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace using RunTree."""
        from langsmith.run_trees import RunTree

        trace_id = str(uuid.uuid4())
        run = RunTree(
            name=name,
            run_type="chain",
            project_name=self.project_name,
            id=trace_id,
            client=self.client,
            inputs=kwargs.get("inputs", {}),
        )

        span = SpanContext(
            name=name,
            span_type=SpanType.TRACE,
            trace_id=trace_id,
            span_id=trace_id,
            start_time=datetime.now(),
            metadata={"run_tree": run},
        )

        try:
            yield span
            run.end(outputs=span.outputs)
            run.post()
        except Exception as e:
            run.end(error=str(e))
            run.post()
            raise

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a child run."""
        from langsmith.run_trees import RunTree

        span_id = str(uuid.uuid4())

        # Map span type to LangSmith run type
        run_type_map = {
            SpanType.LLM: "llm",
            SpanType.RETRIEVER: "retriever",
            SpanType.CHAIN: "chain",
            SpanType.TOOL: "tool",
            SpanType.EMBEDDING: "embedding",
        }
        run_type = run_type_map.get(span_type, "chain")

        parent_run = parent.metadata.get("run_tree") if parent else None

        if parent_run:
            run = parent_run.create_child(
                name=name,
                run_type=run_type,
                inputs=kwargs.get("inputs", {}),
            )
        else:
            run = RunTree(
                name=name,
                run_type=run_type,
                project_name=self.project_name,
                client=self.client,
                inputs=kwargs.get("inputs", {}),
            )

        span = SpanContext(
            name=name,
            span_type=span_type,
            trace_id=parent.trace_id if parent else span_id,
            span_id=span_id,
            parent_span_id=parent.span_id if parent else None,
            start_time=datetime.now(),
            metadata={"run_tree": run},
        )

        try:
            yield span
            run.end(outputs=span.outputs)
            run.post()
        except Exception as e:
            span.error = str(e)
            run.end(error=str(e))
            run.post()
            raise

    def log_llm_call(
        self,
        span: SpanContext,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """Log LLM call details."""
        span.model = model
        span.inputs = {"messages": messages}

        if hasattr(response, "choices"):
            span.outputs = {
                "content": response.choices[0].message.content,
                "model": response.model,
            }
            if hasattr(response, "usage"):
                span.prompt_tokens = response.usage.prompt_tokens
                span.completion_tokens = response.usage.completion_tokens
                span.total_tokens = response.usage.total_tokens

        run = span.metadata.get("run_tree")
        if run:
            run.inputs = span.inputs
            run.outputs = span.outputs

    def log_retrieval(
        self,
        span: SpanContext,
        query: str,
        documents: list[dict],
        scores: list[float],
        **kwargs,
    ) -> None:
        """Log retrieval operation."""
        span.inputs = {"query": query}
        span.outputs = {"documents": documents, "scores": scores}
        span.retrieved_documents = documents
        span.scores = scores

        run = span.metadata.get("run_tree")
        if run:
            run.inputs = span.inputs
            run.outputs = span.outputs

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)
        run = span.metadata.get("run_tree")
        if run:
            run.end(error=str(error))

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view trace in LangSmith."""
        return f"https://smith.langchain.com/o/default/projects/p/{self.project_name}/r/{trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": True,
            "nested_spans": True,
            "cost_tracking": True,
            "evaluations": True,
            "prompt_hub": True,
            "datasets": True,
        }
        return features.get(feature, False)

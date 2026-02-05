"""Weights & Biases Weave observability provider."""

import os
from contextlib import contextmanager
from typing import Any, Generator

try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class WeaveProvider(ObservabilityProvider):
    """
    Weights & Biases Weave provider implementation.
    
    Note: Weave is primarily designed for functional tracing using decorators.
    This implementation uses weave.init() for auto-instrumentation of OpenAI calls.
    Manual spans are mapped to simple log events where possible, but full hierarchy
    might rely on Weave's auto-capture.
    """

    name = "weave"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.project_name = os.getenv("WANDB_PROJECT", "pdf-knowledge-rag")
        self.client = None

    def initialize(self) -> bool:
        """Initialize Weave."""
        if not WEAVE_AVAILABLE:
            print("[Weave] weave package not installed.")
            return False

        try:
            # weave.init() automatically captures LLM calls from supported libraries
            self.client = weave.init(self.project_name)
            return True
        except Exception as e:
            print(f"[Weave] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown Weave."""
        # Weave usually handles shutdown automatically or doesn't require explicit close
        pass

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a trace."""
        # Weave doesn't have an explicit manual "start_trace" context manager in the same way.
        # We'll create a generic contact context.
        span = SpanContext(name=name, span_type=SpanType.TRACE, inputs=kwargs.get("inputs", {}))
        yield span

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a span."""
        span = SpanContext(
            name=name, 
            span_type=span_type, 
            parent_span_id=parent.span_id if parent else None,
            inputs=kwargs.get("inputs", {})
        )
        yield span

    def log_llm_call(
        self,
        span: SpanContext,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """Log LLM call."""
        # Weave auto-instruments OpenAI, so manual logging might duplicate events.
        # However, if we want to explicitly attach it to our manual span context, it's hard.
        # We rely on auto-capture for now.
        pass

    def log_retrieval(
        self,
        span: SpanContext,
        query: str,
        documents: list[dict],
        scores: list[float],
        **kwargs,
    ) -> None:
        """Log retrieval."""
        # We can publish a "Retrieval" object or Call to Weave
        if self.client:
            # This is a bit of a hack to force data into proper Weave UI
            # We treat it as a generic object log
            try:
                weave.publish({
                    "type": "retrieval",
                    "query": query,
                    "documents": documents,
                    "scores": scores,
                    "span_name": span.name
                })
            except Exception:
                pass

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        pass

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get trace URL."""
        # Weave URLs are usually structured by project
        return f"https://wandb.ai/context/{self.project_name}/weave"

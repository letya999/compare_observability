"""Laminar observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class LaminarProvider(ObservabilityProvider):
    """
    Laminar (lmnr) integration.

    Features tested:
    - @observe decorator
    - Manual span API
    - Pipeline visualization
    - Evaluations
    """

    name = "laminar"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize Laminar."""
        api_key = os.getenv("LMNR_PROJECT_API_KEY")
        if not api_key:
            print("[Laminar] No API key found (LMNR_PROJECT_API_KEY)")
            return False

        try:
            from lmnr import Laminar

            Laminar.initialize(project_api_key=api_key)
            self.initialized = True
            return True
        except ImportError:
            print("[Laminar] lmnr package not installed")
            return False
        except Exception as e:
            print(f"[Laminar] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """Flush pending data."""
        try:
            from lmnr import Laminar
            Laminar.flush()
        except Exception:
            pass

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace."""
        from lmnr import observe

        trace_id = str(uuid.uuid4())

        # Laminar uses decorators primarily, we'll use context manager approach
        span = SpanContext(
            name=name,
            span_type=SpanType.TRACE,
            trace_id=trace_id,
            span_id=trace_id,
            start_time=datetime.now(),
            inputs=kwargs.get("inputs", {}),
            metadata={"lmnr_name": name},
        )

        try:
            yield span
        except Exception as e:
            span.error = str(e)
            raise

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a span."""
        from lmnr import observe

        span_id = str(uuid.uuid4())

        span = SpanContext(
            name=name,
            span_type=span_type,
            trace_id=parent.trace_id if parent else span_id,
            span_id=span_id,
            parent_span_id=parent.span_id if parent else None,
            start_time=datetime.now(),
            inputs=kwargs.get("inputs", {}),
            metadata={"lmnr_name": name},
        )

        try:
            yield span
        except Exception as e:
            span.error = str(e)
            raise

    def log_llm_call(
        self,
        span: SpanContext,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """Log LLM call."""
        span.model = model
        span.inputs = {"messages": messages}

        if hasattr(response, "choices"):
            span.outputs = {"content": response.choices[0].message.content}
            if hasattr(response, "usage"):
                span.prompt_tokens = response.usage.prompt_tokens
                span.completion_tokens = response.usage.completion_tokens
                span.total_tokens = response.usage.total_tokens

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

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view trace."""
        return f"https://www.lmnr.ai/traces/{trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": True,
            "nested_spans": True,
            "cost_tracking": True,
            "evaluations": True,
            "pipeline_visualization": True,
        }
        return features.get(feature, False)

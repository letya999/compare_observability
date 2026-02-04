"""Pydantic Logfire observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class LogfireProvider(ObservabilityProvider):
    """
    Pydantic Logfire integration.

    Features tested:
    - OpenAI auto-instrumentation
    - @logfire.span decorator
    - Structured logging
    - Pydantic model validation tracking
    """

    name = "logfire"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.logfire = None

    def initialize(self) -> bool:
        """Initialize Logfire."""
        token = os.getenv("LOGFIRE_TOKEN")
        if not token:
            print("[Logfire] No token found (LOGFIRE_TOKEN)")
            return False

        try:
            import logfire

            logfire.configure(token=token)

            # Auto-instrument OpenAI
            logfire.instrument_openai()

            self.logfire = logfire
            return True
        except ImportError:
            print("[Logfire] logfire package not installed")
            return False
        except Exception as e:
            print(f"[Logfire] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """Flush pending data."""
        # Logfire handles this automatically
        pass

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace using Logfire span."""
        trace_id = str(uuid.uuid4())

        with self.logfire.span(name, **kwargs.get("inputs", {})) as lf_span:
            span = SpanContext(
                name=name,
                span_type=SpanType.TRACE,
                trace_id=trace_id,
                span_id=trace_id,
                start_time=datetime.now(),
                metadata={"logfire_span": lf_span},
            )

            try:
                yield span
                if span.outputs:
                    lf_span.set_attribute("output", str(span.outputs))
            except Exception as e:
                lf_span.record_exception(e)
                raise

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a child span."""
        span_id = str(uuid.uuid4())

        # Add span type to attributes
        attrs = kwargs.get("inputs", {}).copy()
        attrs["span_type"] = span_type.value

        with self.logfire.span(name, **attrs) as lf_span:
            span = SpanContext(
                name=name,
                span_type=span_type,
                trace_id=parent.trace_id if parent else span_id,
                span_id=span_id,
                parent_span_id=parent.span_id if parent else None,
                start_time=datetime.now(),
                metadata={"logfire_span": lf_span},
            )

            try:
                yield span
                if span.outputs:
                    lf_span.set_attribute("output", str(span.outputs))
            except Exception as e:
                span.error = str(e)
                lf_span.record_exception(e)
                raise

    def log_llm_call(
        self,
        span: SpanContext,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """
        Log LLM call.
        Note: Logfire auto-instruments OpenAI, so this is supplementary.
        """
        span.model = model
        span.inputs = {"messages": messages}

        lf_span = span.metadata.get("logfire_span")

        if hasattr(response, "choices"):
            content = response.choices[0].message.content
            span.outputs = {"content": content}

            if lf_span:
                lf_span.set_attribute("llm.model", model)
                lf_span.set_attribute("llm.response", content[:500])

            if hasattr(response, "usage"):
                span.prompt_tokens = response.usage.prompt_tokens
                span.completion_tokens = response.usage.completion_tokens
                span.total_tokens = response.usage.total_tokens

                if lf_span:
                    lf_span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens)
                    lf_span.set_attribute("llm.completion_tokens", response.usage.completion_tokens)

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

        lf_span = span.metadata.get("logfire_span")
        if lf_span:
            lf_span.set_attribute("retrieval.query", query)
            lf_span.set_attribute("retrieval.doc_count", len(documents))
            lf_span.set_attribute("retrieval.top_score", scores[0] if scores else 0)

        # Also log as a separate event
        self.logfire.info(
            "Retrieval completed",
            query=query,
            doc_count=len(documents),
            scores=scores,
        )

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)
        self.logfire.error(
            "Error in span",
            span_name=span.name,
            error=str(error),
            error_type=type(error).__name__,
        )

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view trace in Logfire."""
        return f"https://logfire.pydantic.dev/traces/{trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": True,
            "nested_spans": True,
            "cost_tracking": False,  # Manual
            "evaluations": False,
            "auto_instrumentation": True,
            "pydantic_integration": True,
            "structured_logging": True,
        }
        return features.get(feature, False)

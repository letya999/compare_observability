"""Langfuse observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class LangfuseProvider(ObservabilityProvider):
    """
    Langfuse integration.

    Features tested:
    - @observe decorator
    - trace() / span() / generation() API
    - Scores and evaluations
    - Cost tracking
    - Prompt management
    """

    name = "langfuse"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.client = None
        self._active_traces = {}

    def initialize(self) -> bool:
        """Initialize Langfuse client."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")

        if not public_key or not secret_key:
            print("[Langfuse] Missing API keys (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)")
            return False

        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            # Test connection
            self.client.auth_check()
            return True
        except ImportError:
            print("[Langfuse] langfuse package not installed")
            return False
        except Exception as e:
            print(f"[Langfuse] Failed to connect: {e}")
            return False

    def shutdown(self) -> None:
        """Flush pending data."""
        if self.client:
            self.client.flush()

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())

        trace = self.client.trace(
            id=trace_id,
            name=name,
            input=kwargs.get("inputs"),
            metadata=kwargs.get("metadata"),
        )

        self._active_traces[trace_id] = trace

        span = SpanContext(
            name=name,
            span_type=SpanType.TRACE,
            trace_id=trace_id,
            span_id=trace_id,
            start_time=datetime.now(),
            metadata={"langfuse_trace": trace},
        )

        try:
            yield span
            trace.update(output=span.outputs)
        except Exception as e:
            trace.update(
                output={"error": str(e)},
                level="ERROR",
            )
            raise
        finally:
            self._active_traces.pop(trace_id, None)

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a span within a trace."""
        span_id = str(uuid.uuid4())
        trace_id = parent.trace_id if parent else span_id

        # Get parent trace or span
        langfuse_parent = None
        if parent and "langfuse_trace" in parent.metadata:
            langfuse_parent = parent.metadata["langfuse_trace"]
        elif parent and "langfuse_span" in parent.metadata:
            langfuse_parent = parent.metadata["langfuse_span"]

        # Create appropriate Langfuse object based on span type
        if span_type == SpanType.LLM:
            langfuse_obj = (langfuse_parent or self.client).generation(
                id=span_id,
                name=name,
                input=kwargs.get("inputs"),
                metadata=kwargs.get("metadata"),
            )
        else:
            langfuse_obj = (langfuse_parent or self.client).span(
                id=span_id,
                name=name,
                input=kwargs.get("inputs"),
                metadata=kwargs.get("metadata"),
            )

        span = SpanContext(
            name=name,
            span_type=span_type,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent.span_id if parent else None,
            start_time=datetime.now(),
            metadata={
                "langfuse_span": langfuse_obj,
                "langfuse_trace": parent.metadata.get("langfuse_trace") if parent else None,
            },
        )

        try:
            yield span
            langfuse_obj.end(output=span.outputs)
        except Exception as e:
            span.error = str(e)
            langfuse_obj.end(
                output={"error": str(e)},
                level="ERROR",
            )
            raise

    def log_llm_call(
        self,
        span: SpanContext,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """Log LLM call with Langfuse generation tracking."""
        span.model = model
        span.inputs = {"messages": messages}

        langfuse_obj = span.metadata.get("langfuse_span")
        if not langfuse_obj:
            return

        usage = None
        output_content = None

        if hasattr(response, "choices"):
            output_content = response.choices[0].message.content
            if hasattr(response, "usage"):
                usage = {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                }
                span.prompt_tokens = response.usage.prompt_tokens
                span.completion_tokens = response.usage.completion_tokens
                span.total_tokens = response.usage.total_tokens

        span.outputs = {"content": output_content}

        langfuse_obj.update(
            model=model,
            input=messages,
            output=output_content,
            usage=usage,
        )

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

        langfuse_obj = span.metadata.get("langfuse_span")
        if langfuse_obj:
            langfuse_obj.update(
                input={"query": query},
                output={"documents": documents, "scores": scores},
            )

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)
        langfuse_obj = span.metadata.get("langfuse_span")
        if langfuse_obj:
            langfuse_obj.update(
                level="ERROR",
                status_message=str(error),
            )

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view trace in Langfuse."""
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        return f"{host}/trace/{trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": True,
            "nested_spans": True,
            "cost_tracking": True,
            "evaluations": True,
            "prompt_management": True,
            "datasets": True,
            "scores": True,
        }
        return features.get(feature, False)

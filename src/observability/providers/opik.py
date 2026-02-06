"""Opik observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class OpikProvider(ObservabilityProvider):
    """
    Opik (Comet ML) integration.

    Features tested:
    - @track decorator
    - Manual tracing API
    - Evaluations
    - Prompt versioning
    
    Environment variables:
    - OPIK_API_KEY: API key for authentication
    - OPIK_WORKSPACE: Workspace name
    - OPIK_URL_OVERRIDE: API endpoint (default: https://www.comet.com/opik/api)
    - OPIK_PROJECT_NAME: Project name for organizing traces
    """

    name = "opik"
    supports_streaming = True
    supports_async = False

    def __init__(self):
        self.client = None
        self.project_name = os.getenv("OPIK_PROJECT_NAME", "compare-observability")

    def initialize(self) -> bool:
        """Initialize Opik client."""
        api_key = os.getenv("OPIK_API_KEY")
        if not api_key:
            print("[Opik] No API key found (OPIK_API_KEY)")
            return False

        try:
            import opik
            
            # Configure Opik with all available settings
            workspace = os.getenv("OPIK_WORKSPACE", "default")
            url_override = os.getenv("OPIK_URL_OVERRIDE")
            
            config_kwargs = {
                "api_key": api_key,
                "workspace": workspace,
            }
            
            # Add URL override if specified (for cloud or self-hosted)
            if url_override:
                config_kwargs["url"] = url_override
            
            opik.configure(**config_kwargs)
            self.client = opik.Opik(project_name=self.project_name)
            print(f"[Opik] Initialized successfully (workspace: {workspace}, project: {self.project_name})")
            return True
        except ImportError:
            print("[Opik] opik package not installed")
            return False
        except Exception as e:
            print(f"[Opik] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """Flush pending data."""
        if self.client:
            self.client.flush()

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())

        opik_trace = self.client.trace(
            name=name,
            input=kwargs.get("inputs"),
            metadata=kwargs.get("metadata"),
        )

        span = SpanContext(
            name=name,
            span_type=SpanType.TRACE,
            trace_id=opik_trace.id,
            span_id=opik_trace.id,
            start_time=datetime.now(),
            metadata={"opik_trace": opik_trace},
        )

        try:
            yield span
            opik_trace.end(output=span.outputs)
        except Exception as e:
            opik_trace.end(
                output={"error": str(e)},
                error_info={"message": str(e)},
            )
            raise

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a span within a trace."""
        opik_trace = parent.metadata.get("opik_trace") if parent else None
        opik_parent = parent.metadata.get("opik_span") if parent else None

        # Map span type to Opik types: general, tool, llm, guardrail
        # Note: Opik doesn't have 'retriever' type, use 'general' instead
        type_map = {
            SpanType.LLM: "llm",
            SpanType.RETRIEVER: "general",  # Opik doesn't support 'retriever'
            SpanType.CHAIN: "general",
            SpanType.TOOL: "tool",
            SpanType.EMBEDDING: "general",
        }

        if opik_trace:
            opik_span = opik_trace.span(
                name=name,
                type=type_map.get(span_type, "general"),
                input=kwargs.get("inputs"),
                metadata=kwargs.get("metadata"),
                parent_span_id=opik_parent.id if opik_parent else None,
            )
        else:
            # Standalone span
            opik_span = self.client.span(
                name=name,
                type=type_map.get(span_type, "general"),
                input=kwargs.get("inputs"),
            )

        span = SpanContext(
            name=name,
            span_type=span_type,
            trace_id=parent.trace_id if parent else opik_span.id,
            span_id=opik_span.id,
            parent_span_id=parent.span_id if parent else None,
            start_time=datetime.now(),
            metadata={
                "opik_span": opik_span,
                "opik_trace": opik_trace,
            },
        )

        try:
            yield span
            opik_span.end(output=span.outputs)
        except Exception as e:
            span.error = str(e)
            opik_span.end(
                output={"error": str(e)},
                error_info={"message": str(e)},
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
        """Log LLM call."""
        span.model = model
        span.inputs = {"messages": messages}

        opik_span = span.metadata.get("opik_span")
        if not opik_span:
            return

        if hasattr(response, "choices"):
            content = response.choices[0].message.content
            span.outputs = {"content": content}

            usage = {}
            if hasattr(response, "usage"):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                span.prompt_tokens = response.usage.prompt_tokens
                span.completion_tokens = response.usage.completion_tokens
                span.total_tokens = response.usage.total_tokens

            opik_span.update(
                model=model,
                input={"messages": messages},
                output={"content": content},
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

        opik_span = span.metadata.get("opik_span")
        if opik_span:
            opik_span.update(
                input={"query": query},
                output={"documents": documents, "scores": scores},
            )

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)
        opik_span = span.metadata.get("opik_span")
        if opik_span:
            opik_span.update(error_info={"message": str(error)})

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view trace."""
        workspace = os.getenv("OPIK_WORKSPACE", "default")
        return f"https://www.comet.com/opik/{workspace}/traces/{trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": False,
            "nested_spans": True,
            "cost_tracking": True,
            "evaluations": True,
            "prompt_versioning": True,
        }
        return features.get(feature, False)

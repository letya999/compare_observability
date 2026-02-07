"""Arize AX observability provider (2026 version)."""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class ArizePhoenixProvider(ObservabilityProvider):
    """
    Arize AX integration using arize.otel.register() (current 2026 method).
    
    Official docs: https://arize.com/docs/ax/observe/tracing/setup/manual-instrumentation
    """

    name = "arize"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.tracer_provider = None
        self.tracer = None

    def initialize(self) -> bool:
        """Initialize Arize using arize.otel.register()."""
        try:
            from arize.otel import register
            
            # Get configuration
            space_id = os.getenv("ARIZE_SPACE_ID")
            api_key = os.getenv("ARIZE_API_KEY")
            
            if not space_id or not api_key:
                print("[Arize] Missing ARIZE_SPACE_ID or ARIZE_API_KEY")
                return False
            
            # Get project name
            project_name = os.getenv("ARIZE_PROJECT_NAME", "compare-observability")
            
            # Use arize.otel.register() - the official 2026 way
            self.tracer_provider = register(
                space_id=space_id,
                api_key=api_key,
                project_name=project_name,
            )
            
            self.tracer = self.tracer_provider.get_tracer(__name__)
            
            print(f"[Arize] Initialized successfully (space: {space_id}, project: {project_name})")
            return True
            
        except ImportError as e:
            print(f"[Arize] Missing dependencies: {e}")
            print("[Arize] Install with: pip install arize-otel openinference-instrumentation-openai")
            return False
        except Exception as e:
            print(f"[Arize] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """Flush spans."""
        if self.tracer_provider and hasattr(self.tracer_provider, 'force_flush'):
            try:
                # Force flush with timeout (in milliseconds)
                self.tracer_provider.force_flush(timeout_millis=5000)
                print("[Arize] Flushed all pending spans")
            except Exception as e:
                print(f"[Arize] Error flushing: {e}")

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace using OpenTelemetry."""
        with self.tracer.start_as_current_span(name) as otel_span:
            # Set OpenInference attributes
            otel_span.set_attribute("openinference.span.kind", "CHAIN")

            # Set User and Session IDs if provided
            if kwargs.get("user_id"):
                otel_span.set_attribute("user.id", kwargs["user_id"])
            if kwargs.get("session_id"):
                otel_span.set_attribute("session.id", kwargs["session_id"])

            if kwargs.get("inputs"):
                otel_span.set_attribute("input.value", str(kwargs["inputs"]))

            span = SpanContext(
                name=name,
                span_type=SpanType.TRACE,
                trace_id=format(otel_span.get_span_context().trace_id, '032x'),
                span_id=format(otel_span.get_span_context().span_id, '016x'),
                start_time=datetime.now(),
                metadata={"otel_span": otel_span},
            )

            try:
                yield span
                if span.outputs:
                    otel_span.set_attribute("output.value", str(span.outputs))
            except Exception as e:
                from opentelemetry.trace import Status, StatusCode
                otel_span.record_exception(e)
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
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
        # Map span type to OpenInference kind
        kind_map = {
            SpanType.LLM: "LLM",
            SpanType.RETRIEVER: "RETRIEVER",
            SpanType.CHAIN: "CHAIN",
            SpanType.TOOL: "TOOL",
            SpanType.EMBEDDING: "EMBEDDING",
        }

        with self.tracer.start_as_current_span(name) as otel_span:
            otel_span.set_attribute(
                "openinference.span.kind",
                kind_map.get(span_type, "CHAIN")
            )

            if kwargs.get("inputs"):
                otel_span.set_attribute("input.value", str(kwargs["inputs"]))

            span = SpanContext(
                name=name,
                span_type=span_type,
                trace_id=parent.trace_id if parent else format(otel_span.get_span_context().trace_id, '032x'),
                span_id=format(otel_span.get_span_context().span_id, '016x'),
                parent_span_id=parent.span_id if parent else None,
                start_time=datetime.now(),
                metadata={"otel_span": otel_span},
            )

            try:
                yield span
                if span.outputs:
                    otel_span.set_attribute("output.value", str(span.outputs))
            except Exception as e:
                span.error = str(e)
                otel_span.record_exception(e)
                raise

    def log_llm_call(
        self,
        span: SpanContext,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """Log LLM call with OpenInference attributes."""
        otel_span = span.metadata.get("otel_span")
        if not otel_span:
            return

        span.model = model
        span.inputs = {"messages": messages}

        # Set OpenInference LLM attributes
        otel_span.set_attribute("llm.model_name", model)
        otel_span.set_attribute("llm.invocation_parameters", str(kwargs))

        # Log messages
        for i, msg in enumerate(messages):
            otel_span.set_attribute(f"llm.input_messages.{i}.role", msg.get("role", ""))
            otel_span.set_attribute(f"llm.input_messages.{i}.content", msg.get("content", ""))

        if hasattr(response, "choices"):
            content = response.choices[0].message.content
            otel_span.set_attribute("llm.output_messages.0.role", "assistant")
            otel_span.set_attribute("llm.output_messages.0.content", content)
            span.outputs = {"content": content}

            if hasattr(response, "usage"):
                otel_span.set_attribute("llm.token_count.prompt", response.usage.prompt_tokens)
                otel_span.set_attribute("llm.token_count.completion", response.usage.completion_tokens)
                otel_span.set_attribute("llm.token_count.total", response.usage.total_tokens)
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
        """Log retrieval with OpenInference attributes."""
        otel_span = span.metadata.get("otel_span")
        if not otel_span:
            return

        span.inputs = {"query": query}
        span.outputs = {"documents": documents, "scores": scores}
        span.retrieved_documents = documents
        span.scores = scores

        otel_span.set_attribute("retrieval.query", query)
        for i, (doc, score) in enumerate(zip(documents, scores)):
            otel_span.set_attribute(f"retrieval.documents.{i}.content", doc.get("text", "")[:500])
            otel_span.set_attribute(f"retrieval.documents.{i}.score", score)
            if "id" in doc:
                otel_span.set_attribute(f"retrieval.documents.{i}.id", doc["id"])

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)
        otel_span = span.metadata.get("otel_span")
        if otel_span:
            otel_span.record_exception(error)

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view trace in Arize AX."""
        space_id = os.getenv("ARIZE_SPACE_ID", "")
        return f"https://arize.com/spaces/{space_id}/tracing/{trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": True,
            "nested_spans": True,
            "cost_tracking": False,
            "evaluations": True,
            "auto_instrumentation": True,
            "opentelemetry": True,
        }
        return features.get(feature, False)

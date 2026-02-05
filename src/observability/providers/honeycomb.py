"""Honeycomb (OpenTelemetry) observability provider."""

import os
import time
from contextlib import contextmanager
from typing import Any, Generator

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class HoneycombProvider(ObservabilityProvider):
    """
    Honeycomb provider using OpenTelemetry OTLP exporter.
    """

    name = "honeycomb"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.api_key = os.getenv("HONEYCOMB_API_KEY")
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "pdf-knowledge-rag")
        self.tracer = None
        self.tracer_provider = None

    def initialize(self) -> bool:
        """Initialize OpenTelemetry for Honeycomb."""
        if not OTEL_AVAILABLE:
            print("[Honeycomb] OpenTelemetry packages not installed.")
            return False

        if not self.api_key:
            print("[Honeycomb] HONEYCOMB_API_KEY not found.")
            return False

        try:
            # Create Resource
            resource = Resource.create(attributes={
                "service.name": self.service_name,
            })

            # Create Exporter
            # Honeycomb endpoint is usually distinct, but OTLP default often works well if headers are set
            # or endpoint is explicitly: https://api.honeycomb.io
            # Standard OTLP gRPC endpoint for Honeycomb: api.honeycomb.io:443
            exporter = OTLPSpanExporter(
                endpoint="api.honeycomb.io:443",
                headers={"x-honeycomb-team": self.api_key}
            )

            # Create Provider
            self.tracer_provider = TracerProvider(resource=resource)
            self.tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            
            # Setup tracer
            self.tracer = self.tracer_provider.get_tracer(__name__)
            
            return True
        except Exception as e:
            print(f"[Honeycomb] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown provider."""
        if self.tracer_provider:
            try:
                self.tracer_provider.shutdown()
            except Exception as e:
                print(f"[Honeycomb] Error shutting down: {e}")

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a trace."""
        if not self.tracer:
            # Fallback
            yield SpanContext(name=name, span_type=SpanType.TRACE)
            return

        with self.tracer.start_as_current_span(name, attributes=kwargs.get("inputs", {})) as span:
            ctx = SpanContext(
                name=name, 
                span_type=SpanType.TRACE,
                trace_id=format(span.get_span_context().trace_id, "032x"),
                span_id=format(span.get_span_context().span_id, "016x")
            )
            yield ctx

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a span."""
        if not self.tracer:
             yield SpanContext(name=name, span_type=span_type)
             return

        # Note: In standard OTeL, context is propagated via Context objects, 
        # but since we are manually nesting context using 'with', 
        # `start_as_current_span` usually handles nesting correctly if called within the parent `with` block scope.
        # However, our ObservabilityManager architecture might not strictly nest execution in Python context 
        # (though it does using "with manager.span").
        # Provided the user nests "with" blocks, OTeL works automatically.
        
        attributes = kwargs.get("inputs", {})
        attributes["span.type"] = span_type.value
        
        with self.tracer.start_as_current_span(name, attributes=attributes) as span:
             ctx = SpanContext(
                name=name,
                span_type=span_type,
                trace_id=format(span.get_span_context().trace_id, "032x"),
                span_id=format(span.get_span_context().span_id, "016x"),
                parent_span_id=parent.span_id if parent else None
            )
             try:
                 yield ctx
             except Exception as e:
                 span.record_exception(e)
                 raise

    def log_llm_call(
        self,
        span: SpanContext,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """Log LLM call using OTeL events or attributes."""
        # Since we can't easily get the 'current' span object from just SpanContext unless we kept a map,
        # we rely on the fact that the caller likely is inside the 'with' block, 
        # so `otel_trace.get_current_span()` might work?
        # But SpanContext is passed.
        # Ideally we'd store the OTeL span object in SpanContext.metadata or a private map.
        
        # NOTE: For simplicity, assume get_current_span works if called synchronously inside block.
        current_span = otel_trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("llm.model", model)
            current_span.add_event("llm_call", {
                "messages": str(messages),
                "response": str(response)
            })

    def log_retrieval(
        self,
        span: SpanContext,
        query: str,
        documents: list[dict],
        scores: list[float],
        **kwargs,
    ) -> None:
        """Log retrieval."""
        current_span = otel_trace.get_current_span()
        if current_span and current_span.is_recording():
             current_span.add_event("retrieval", {
                 "query": query,
                 "document_count": len(documents),
                 "score_avg": sum(scores)/len(scores) if scores else 0
             })

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        current_span = otel_trace.get_current_span()
        if current_span:
            current_span.record_exception(error)

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get trace URL."""
        # Honeycomb UI URL construction
        # https://ui.honeycomb.io/{team}/environments/{env}/datasets/{dataset}/trace?trace_id={id}
        # Hard to construct without team slug.
        return None

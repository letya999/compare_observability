"""Observability Manager - coordinates multiple providers simultaneously."""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

from src.config import config, ObservabilityProvider as ProviderName

from .base import ObservabilityProvider, SpanContext, SpanType


@dataclass
class MultiProviderSpan:
    """Span that wraps multiple provider spans."""

    name: str
    span_type: SpanType
    provider_spans: dict[str, SpanContext] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def trace_ids(self) -> dict[str, str]:
        """Get trace IDs from all providers."""
        return {
            name: span.trace_id
            for name, span in self.provider_spans.items()
            if span.trace_id
        }


class ObservabilityManager:
    """
    Manages multiple observability providers simultaneously.

    Allows sending the same traces to multiple platforms for comparison.
    """

    def __init__(self, providers: list[ProviderName] | None = None):
        self.active_providers: dict[str, ObservabilityProvider] = {}
        self.provider_names = providers or config.observability_providers

        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        from .providers import PROVIDER_CLASSES

        for provider_name in self.provider_names:
            if provider_name in PROVIDER_CLASSES:
                provider_class = PROVIDER_CLASSES[provider_name]
                provider = provider_class()

                try:
                    if provider.initialize():
                        self.active_providers[provider_name] = provider
                        print(f"[Observability] Initialized: {provider_name}")
                    else:
                        print(f"[Observability] Failed to initialize: {provider_name}")
                except Exception as e:
                    print(f"[Observability] Error initializing {provider_name}: {e}")

    def shutdown(self) -> None:
        """Shutdown all providers."""
        for name, provider in self.active_providers.items():
            try:
                provider.shutdown()
                print(f"[Observability] Shutdown: {name}")
            except Exception as e:
                print(f"[Observability] Error shutting down {name}: {e}")

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[MultiProviderSpan, None, None]:
        """
        Start a trace across all active providers.

        Usage:
            with manager.trace("my_pipeline") as trace:
                # Pipeline code here
                with manager.span("step1", SpanType.LLM, trace) as span:
                    # LLM call
        """
        multi_span = MultiProviderSpan(name=name, span_type=SpanType.TRACE)

        # Start trace in all providers
        provider_contexts = {}
        for pname, provider in self.active_providers.items():
            try:
                ctx = provider.trace(name, **kwargs)
                provider_contexts[pname] = ctx.__enter__()
                multi_span.provider_spans[pname] = provider_contexts[pname]
            except Exception as e:
                print(f"[Observability] Error starting trace in {pname}: {e}")

        try:
            yield multi_span
        finally:
            # End trace in all providers
            for pname, ctx in provider_contexts.items():
                try:
                    self.active_providers[pname].trace(name, **kwargs).__exit__(None, None, None)
                except Exception as e:
                    print(f"[Observability] Error ending trace in {pname}: {e}")

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: MultiProviderSpan | None = None,
        **kwargs,
    ) -> Generator[MultiProviderSpan, None, None]:
        """Create a span across all active providers."""
        multi_span = MultiProviderSpan(name=name, span_type=span_type)

        provider_contexts = {}
        for pname, provider in self.active_providers.items():
            try:
                parent_span = parent.provider_spans.get(pname) if parent else None
                ctx = provider.span(name, span_type, parent_span, **kwargs)
                provider_contexts[pname] = ctx.__enter__()
                multi_span.provider_spans[pname] = provider_contexts[pname]
            except Exception as e:
                print(f"[Observability] Error starting span in {pname}: {e}")

        try:
            yield multi_span
        finally:
            for pname, ctx in provider_contexts.items():
                try:
                    provider = self.active_providers[pname]
                    provider.span(name, span_type).__exit__(None, None, None)
                except Exception as e:
                    print(f"[Observability] Error ending span in {pname}: {e}")

    def log_llm_call(
        self,
        span: MultiProviderSpan,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """Log LLM call to all providers."""
        for pname, provider in self.active_providers.items():
            if pname in span.provider_spans:
                try:
                    provider.log_llm_call(
                        span.provider_spans[pname], model, messages, response, **kwargs
                    )
                except Exception as e:
                    print(f"[Observability] Error logging LLM call in {pname}: {e}")

    def log_retrieval(
        self,
        span: MultiProviderSpan,
        query: str,
        documents: list[dict],
        scores: list[float],
        **kwargs,
    ) -> None:
        """Log retrieval to all providers."""
        for pname, provider in self.active_providers.items():
            if pname in span.provider_spans:
                try:
                    provider.log_retrieval(
                        span.provider_spans[pname], query, documents, scores, **kwargs
                    )
                except Exception as e:
                    print(f"[Observability] Error logging retrieval in {pname}: {e}")

    def log_error(self, span: MultiProviderSpan, error: Exception) -> None:
        """Log error to all providers."""
        for pname, provider in self.active_providers.items():
            if pname in span.provider_spans:
                try:
                    provider.log_error(span.provider_spans[pname], error)
                except Exception as e:
                    print(f"[Observability] Error logging error in {pname}: {e}")

    def get_trace_urls(self, trace: MultiProviderSpan) -> dict[str, str]:
        """Get URLs to view this trace in all providers' UIs."""
        urls = {}
        for pname, provider in self.active_providers.items():
            if pname in trace.provider_spans:
                span = trace.provider_spans[pname]
                if span.trace_id:
                    url = provider.get_trace_url(span.trace_id)
                    if url:
                        urls[pname] = url
        return urls

    def get_provider_status(self) -> dict[str, dict]:
        """Get status of all providers."""
        return {
            name: {
                "active": True,
                "supports_streaming": provider.supports_streaming,
                "supports_async": provider.supports_async,
            }
            for name, provider in self.active_providers.items()
        }

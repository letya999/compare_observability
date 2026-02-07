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
        self.init_errors: dict[str, str] = {}  # Store initialization errors
        self.provider_names = providers or config.observability_providers

        self._initialize_providers()
        
        # Register shutdown to be called on program exit (critical for Streamlit)
        import atexit
        atexit.register(self.shutdown)

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        # Map names to classes
        from .providers.arize import ArizePhoenixProvider
        from .providers.langsmith import LangSmithProvider
        from .providers.langfuse import LangfuseProvider
        from .providers.opik import OpikProvider
        from .providers.braintrust import BraintrustProvider
        from .providers.laminar import LaminarProvider
        from .providers.agentops import AgentOpsProvider
        from .providers.evidently import EvidentlyProvider
        from .providers.logfire import LogfireProvider
        from .providers.helicone import HeliconeProvider

        provider_classes = {
            "langsmith": LangSmithProvider,
            "langfuse": LangfuseProvider,
            "arize": ArizePhoenixProvider,
            "opik": OpikProvider,
            "braintrust": BraintrustProvider,
            "laminar": LaminarProvider,
            "agentops": AgentOpsProvider,
            "evidently": EvidentlyProvider,
            "logfire": LogfireProvider,
            "helicone": HeliconeProvider,
        }

        print(f"[Observability] Initializing providers: {self.provider_names}")

        for name in self.provider_names:
            if name not in provider_classes:
                print(f"[Observability] Unknown provider: {name}")
                self.init_errors[name] = "Unknown provider type"
                continue

            try:
                # Instantiate and initialize
                provider_cls = provider_classes[name]
                # Instantiate and initialize
                provider = provider_cls()
                
                if provider.initialize():
                    self.active_providers[name] = provider
                    print(f"[Observability] Initialized: {name}")
                else:
                    # Try to capture failure reason if provider stored it (we might need to add this to BaseProvider)
                    # For now, we assume it failed silently or printed to stdout
                    # Let's check if provider has 'last_error' or similar, or just generic msg
                    self.init_errors[name] = getattr(provider, "init_error", "Initialization failed (check logs/env)")
                    print(f"[Observability] Failed to initialize: {name}")
            except Exception as e:
                print(f"[Observability] Error initializing {name}: {e}")
                self.init_errors[name] = str(e)
        
        # Give providers time to finish sending data
        import time
        time.sleep(0.5)
        print("[Observability] All providers shutdown complete")

    def shutdown(self) -> None:
        """Shutdown all providers."""
        print("[Observability] Shutting down all providers...")
        for name, provider in self.active_providers.items():
            try:
                provider.shutdown()
            except Exception as e:
                print(f"[Observability] Error shutting down {name}: {e}")
        
        # Give providers time to finish sending data
        import time
        time.sleep(0.5)
        print("[Observability] All providers shutdown complete")

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
        # We need to track BOTH the context manager AND the span it yields
        provider_context_managers = {}  # pname -> context manager object
        for pname, provider in self.active_providers.items():
            try:
                ctx_manager = provider.trace(name, **kwargs)
                provider_context_managers[pname] = ctx_manager
                span_ctx = ctx_manager.__enter__()
                multi_span.provider_spans[pname] = span_ctx
            except Exception as e:
                print(f"[Observability] Error starting trace in {pname}: {e}")

        try:
            yield multi_span
        finally:
            # End trace in all providers - use the SAME context manager we started
            for pname, ctx_manager in provider_context_managers.items():
                try:
                    ctx_manager.__exit__(None, None, None)
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

        # Track context managers separately from spans
        provider_context_managers = {}  # pname -> context manager object
        for pname, provider in self.active_providers.items():
            try:
                parent_span = parent.provider_spans.get(pname) if parent else None
                ctx_manager = provider.span(name, span_type, parent_span, **kwargs)
                provider_context_managers[pname] = ctx_manager
                span_ctx = ctx_manager.__enter__()
                multi_span.provider_spans[pname] = span_ctx
            except Exception as e:
                print(f"[Observability] Error starting span in {pname}: {e}")

        try:
            yield multi_span
        finally:
            # End span in all providers - use the SAME context manager we started
            for pname, ctx_manager in provider_context_managers.items():
                try:
                    ctx_manager.__exit__(None, None, None)
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

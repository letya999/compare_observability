"""Base classes for observability providers."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generator


class SpanType(Enum):
    """Types of spans in the pipeline."""

    TRACE = "trace"
    LLM = "llm"
    RETRIEVER = "retriever"
    CHAIN = "chain"
    TOOL = "tool"
    EMBEDDING = "embedding"


@dataclass
class SpanContext:
    """Context for a span being traced."""

    name: str
    span_type: SpanType
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    # LLM-specific fields
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Retriever-specific fields
    retrieved_documents: list[dict] | None = None
    scores: list[float] | None = None


class ObservabilityProvider(ABC):
    """Abstract base class for observability providers."""

    name: str = "base"
    supports_streaming: bool = False
    supports_async: bool = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider. Returns True if successful.

        Should handle missing API keys gracefully.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources and flush any pending data."""
        pass

    @abstractmethod
    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace (root span)."""
        pass

    @abstractmethod
    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a child span within a trace."""
        pass

    @abstractmethod
    def log_llm_call(
        self,
        span: SpanContext,
        model: str,
        messages: list[dict],
        response: Any,
        **kwargs,
    ) -> None:
        """Log an LLM API call with full details."""
        pass

    @abstractmethod
    def log_retrieval(
        self,
        span: SpanContext,
        query: str,
        documents: list[dict],
        scores: list[float],
        **kwargs,
    ) -> None:
        """Log a retrieval operation."""
        pass

    @abstractmethod
    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log an error within a span."""
        pass

    def get_trace_url(self, trace_id: str) -> str | None:
        """Return URL to view trace in provider's UI (if available)."""
        return None

    def supports_feature(self, feature: str) -> bool:
        """Check if provider supports a specific feature."""
        features = {
            "streaming": self.supports_streaming,
            "async": self.supports_async,
            "nested_spans": True,  # Most support this
            "cost_tracking": False,
            "evaluations": False,
        }
        return features.get(feature, False)

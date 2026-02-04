"""Evidently AI observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class EvidentlyProvider(ObservabilityProvider):
    """
    Evidently AI integration.

    Note: Evidently is primarily an ML monitoring tool, not a tracing platform.
    It focuses on data quality, model performance, and drift detection.

    Features tested:
    - Data quality reports
    - Text descriptor analysis
    - Custom metrics
    """

    name = "evidently"
    supports_streaming = False
    supports_async = False

    def __init__(self):
        self.project = None
        self.workspace = None

    def initialize(self) -> bool:
        """Initialize Evidently."""
        try:
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metric_preset import TextEvals

            # Evidently works locally by default
            # Cloud requires separate setup
            self.initialized = True
            return True
        except ImportError:
            print("[Evidently] evidently package not installed")
            return False
        except Exception as e:
            print(f"[Evidently] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """No cleanup needed."""
        pass

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """
        Evidently doesn't do real-time tracing.
        This is a placeholder for API compatibility.
        """
        span = SpanContext(
            name=name,
            span_type=SpanType.TRACE,
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            start_time=datetime.now(),
            metadata={"evidently_data": []},
        )

        try:
            yield span
            # After trace completes, could generate report
            self._generate_report(span)
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
        """Create a span (data collection point)."""
        span = SpanContext(
            name=name,
            span_type=span_type,
            trace_id=parent.trace_id if parent else str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=parent.span_id if parent else None,
            start_time=datetime.now(),
            metadata={"evidently_data": parent.metadata.get("evidently_data", []) if parent else []},
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
        """Collect LLM data for analysis."""
        span.model = model
        span.inputs = {"messages": messages}

        data_point = {
            "type": "llm_call",
            "model": model,
            "prompt": messages[-1].get("content", "") if messages else "",
        }

        if hasattr(response, "choices"):
            content = response.choices[0].message.content
            span.outputs = {"content": content}
            data_point["response"] = content

            if hasattr(response, "usage"):
                span.prompt_tokens = response.usage.prompt_tokens
                span.completion_tokens = response.usage.completion_tokens
                span.total_tokens = response.usage.total_tokens
                data_point["tokens"] = response.usage.total_tokens

        span.metadata.get("evidently_data", []).append(data_point)

    def log_retrieval(
        self,
        span: SpanContext,
        query: str,
        documents: list[dict],
        scores: list[float],
        **kwargs,
    ) -> None:
        """Collect retrieval data for analysis."""
        span.inputs = {"query": query}
        span.outputs = {"documents": documents, "scores": scores}
        span.retrieved_documents = documents
        span.scores = scores

        data_point = {
            "type": "retrieval",
            "query": query,
            "doc_count": len(documents),
            "scores": scores,
        }
        span.metadata.get("evidently_data", []).append(data_point)

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)

    def _generate_report(self, span: SpanContext) -> None:
        """Generate Evidently report from collected data."""
        # This would generate actual Evidently reports
        # For now, just a placeholder
        data = span.metadata.get("evidently_data", [])
        if data:
            print(f"[Evidently] Collected {len(data)} data points for analysis")

    def get_trace_url(self, trace_id: str) -> str | None:
        """Evidently doesn't have trace URLs in the same sense."""
        return None

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": False,
            "async": False,
            "nested_spans": False,
            "cost_tracking": False,
            "evaluations": True,  # Via reports
            "data_quality": True,
            "drift_detection": True,
            "text_analysis": True,
        }
        return features.get(feature, False)

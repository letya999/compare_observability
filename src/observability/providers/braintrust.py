"""Braintrust observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class BraintrustProvider(ObservabilityProvider):
    """
    Braintrust integration.

    Features tested:
    - @traced decorator
    - init_logger() API
    - Evaluations with Eval()
    - Datasets
    - Experiments
    """

    name = "braintrust"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.logger = None

    def initialize(self) -> bool:
        """Initialize Braintrust logger."""
        api_key = os.getenv("BRAINTRUST_API_KEY")
        if not api_key:
            print("[Braintrust] No API key found (BRAINTRUST_API_KEY)")
            return False

        try:
            import braintrust

            project_name = os.getenv("BRAINTRUST_PROJECT", "compare-observability")

            self.logger = braintrust.init_logger(
                project=project_name,
                api_key=api_key,
            )
            print(f"[Braintrust] Initialized successfully (project: {project_name})")
            return True
        except ImportError:
            print("[Braintrust] braintrust package not installed")
            return False
        except Exception as e:
            print(f"[Braintrust] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """Flush pending data."""
        if self.logger:
            self.logger.flush()

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())

        bt_span = self.logger.start_span(
            name=name,
            input=kwargs.get("inputs"),
            span_attributes={"type": "trace"},
        )

        span = SpanContext(
            name=name,
            span_type=SpanType.TRACE,
            trace_id=bt_span.id,
            span_id=bt_span.id,
            start_time=datetime.now(),
            metadata={"bt_span": bt_span},
        )

        try:
            yield span
            bt_span.log(output=span.outputs)
            bt_span.end()
        except Exception as e:
            bt_span.log(error=str(e))
            bt_span.end()
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
        parent_bt = parent.metadata.get("bt_span") if parent else None

        # Map span type
        type_map = {
            SpanType.LLM: "llm",
            SpanType.RETRIEVER: "retriever",
            SpanType.CHAIN: "chain",
            SpanType.TOOL: "tool",
            SpanType.EMBEDDING: "embedding",
        }

        if parent_bt:
            bt_span = parent_bt.start_span(
                name=name,
                input=kwargs.get("inputs"),
                span_attributes={"type": type_map.get(span_type, "chain")},
            )
        else:
            bt_span = self.logger.start_span(
                name=name,
                input=kwargs.get("inputs"),
                span_attributes={"type": type_map.get(span_type, "chain")},
            )

        span = SpanContext(
            name=name,
            span_type=span_type,
            trace_id=parent.trace_id if parent else bt_span.id,
            span_id=bt_span.id,
            parent_span_id=parent.span_id if parent else None,
            start_time=datetime.now(),
            metadata={"bt_span": bt_span},
        )

        try:
            yield span
            bt_span.log(output=span.outputs)
            bt_span.end()
        except Exception as e:
            span.error = str(e)
            bt_span.log(error=str(e))
            bt_span.end()
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

        bt_span = span.metadata.get("bt_span")
        if not bt_span:
            return

        log_data = {
            "input": {"messages": messages},
            "metadata": {"model": model},
        }

        if hasattr(response, "choices"):
            content = response.choices[0].message.content
            span.outputs = {"content": content}
            log_data["output"] = content

            if hasattr(response, "usage"):
                log_data["metrics"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                span.prompt_tokens = response.usage.prompt_tokens
                span.completion_tokens = response.usage.completion_tokens
                span.total_tokens = response.usage.total_tokens

        bt_span.log(**log_data)

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

        bt_span = span.metadata.get("bt_span")
        if bt_span:
            bt_span.log(
                input={"query": query},
                output={"documents": documents, "scores": scores},
            )

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)
        bt_span = span.metadata.get("bt_span")
        if bt_span:
            bt_span.log(error=str(error))

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view trace."""
        return f"https://www.braintrust.dev/app/pdf-knowledge-rag/logs?span={trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": True,
            "nested_spans": True,
            "cost_tracking": True,
            "evaluations": True,
            "datasets": True,
            "experiments": True,
        }
        return features.get(feature, False)

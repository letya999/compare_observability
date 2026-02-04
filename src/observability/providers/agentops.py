"""AgentOps observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class AgentOpsProvider(ObservabilityProvider):
    """
    AgentOps integration.

    Features tested:
    - Auto-instrumentation
    - Session management
    - Agent tracking
    - Tool usage tracking
    """

    name = "agentops"
    supports_streaming = True
    supports_async = False

    def __init__(self):
        self.session = None

    def initialize(self) -> bool:
        """Initialize AgentOps."""
        api_key = os.getenv("AGENTOPS_API_KEY")
        if not api_key:
            print("[AgentOps] No API key found (AGENTOPS_API_KEY)")
            return False

        try:
            import agentops

            agentops.init(
                api_key=api_key,
                auto_start_session=False,
            )
            return True
        except ImportError:
            print("[AgentOps] agentops package not installed")
            return False
        except Exception as e:
            print(f"[AgentOps] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """End session and flush."""
        if self.session:
            self.session.end_session()
            self.session = None

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new session/trace."""
        import agentops

        self.session = agentops.start_session(tags=[name])

        span = SpanContext(
            name=name,
            span_type=SpanType.TRACE,
            trace_id=str(self.session.session_id) if self.session else str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            start_time=datetime.now(),
            metadata={"agentops_session": self.session},
        )

        try:
            yield span
            if self.session:
                self.session.end_session(end_state="Success")
        except Exception as e:
            if self.session:
                self.session.end_session(end_state="Fail", end_state_reason=str(e))
            raise
        finally:
            self.session = None

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType,
        parent: SpanContext | None = None,
        **kwargs,
    ) -> Generator[SpanContext, None, None]:
        """Create a span (action in AgentOps)."""
        import agentops

        span_id = str(uuid.uuid4())

        span = SpanContext(
            name=name,
            span_type=span_type,
            trace_id=parent.trace_id if parent else span_id,
            span_id=span_id,
            parent_span_id=parent.span_id if parent else None,
            start_time=datetime.now(),
            metadata={"agentops_session": parent.metadata.get("agentops_session") if parent else self.session},
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
        """Log LLM call - AgentOps auto-instruments OpenAI."""
        span.model = model
        span.inputs = {"messages": messages}

        if hasattr(response, "choices"):
            span.outputs = {"content": response.choices[0].message.content}
            if hasattr(response, "usage"):
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
        """Log retrieval as a tool action."""
        import agentops

        span.inputs = {"query": query}
        span.outputs = {"documents": documents, "scores": scores}
        span.retrieved_documents = documents
        span.scores = scores

        # Record as tool event
        agentops.record(
            agentops.ToolEvent(
                name="retrieval",
                params={"query": query},
                returns={"doc_count": len(documents), "top_score": scores[0] if scores else 0},
            )
        )

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        import agentops

        span.error = str(error)
        agentops.record(
            agentops.ErrorEvent(
                error_type=type(error).__name__,
                details=str(error),
            )
        )

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view session."""
        return f"https://app.agentops.ai/session/{trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": False,
            "nested_spans": False,  # Flat structure
            "cost_tracking": True,
            "evaluations": False,
            "auto_instrumentation": True,
            "agent_tracking": True,
        }
        return features.get(feature, False)

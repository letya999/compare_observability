"""Langfuse observability provider - Updated for Langfuse SDK v3."""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator
import uuid

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class LangfuseProvider(ObservabilityProvider):
    """
    Langfuse integration using SDK v3 API.

    Uses:
    - get_client() singleton
    - start_as_current_observation() context manager
    - flush() for ensuring data is sent
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
            from langfuse import get_client
            
            # Set environment variables for the client
            os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
            os.environ["LANGFUSE_SECRET_KEY"] = secret_key
            os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            self.client = get_client()
            
            # Test connection - try auth_check if available
            if hasattr(self.client, 'auth_check'):
                self.client.auth_check()
            
            print(f"[Langfuse] Initialized successfully")
            return True
        except ImportError:
            print("[Langfuse] langfuse package not installed")
            return False
        except Exception as e:
            print(f"[Langfuse] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """Flush pending data."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                print(f"[Langfuse] Error during flush: {e}")

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a new trace using start_as_current_observation."""
        trace_id = str(uuid.uuid4())
        
        try:
            # Use the new SDK v3 API
            with self.client.start_as_current_observation(
                as_type="span",
                name=name,
                input=kwargs.get("inputs"),
                metadata=kwargs.get("metadata"),
            ) as langfuse_span:
                
                self._active_traces[trace_id] = langfuse_span
                
                span = SpanContext(
                    name=name,
                    span_type=SpanType.TRACE,
                    trace_id=trace_id,
                    span_id=trace_id,
                    start_time=datetime.now(),
                    metadata={"langfuse_span": langfuse_span},
                )
                
                try:
                    yield span
                    langfuse_span.update(output=span.outputs)
                except Exception as e:
                    langfuse_span.update(
                        output={"error": str(e)},
                        level="ERROR",
                        status_message=str(e)
                    )
                    raise
                finally:
                    self._active_traces.pop(trace_id, None)
                    
        except Exception as e:
            # Fallback if SDK API is different
            print(f"[Langfuse] Error starting trace: {e}")
            span = SpanContext(
                name=name,
                span_type=SpanType.TRACE,
                trace_id=trace_id,
                span_id=trace_id,
                start_time=datetime.now(),
                metadata={},
            )
            yield span

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
        
        # Determine observation type based on span_type
        if span_type == SpanType.LLM:
            as_type = "generation"
        else:
            as_type = "span"
        
        try:
            with self.client.start_as_current_observation(
                as_type=as_type,
                name=name,
                input=kwargs.get("inputs"),
                metadata=kwargs.get("metadata"),
                model=kwargs.get("model") if span_type == SpanType.LLM else None,
            ) as langfuse_obj:
                
                span = SpanContext(
                    name=name,
                    span_type=span_type,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent.span_id if parent else None,
                    start_time=datetime.now(),
                    metadata={
                        "langfuse_span": langfuse_obj,
                    },
                )
                
                try:
                    yield span
                    langfuse_obj.update(output=span.outputs)
                except Exception as e:
                    span.error = str(e)
                    langfuse_obj.update(
                        output={"error": str(e)},
                        level="ERROR",
                        status_message=str(e)
                    )
                    raise
                    
        except Exception as e:
            print(f"[Langfuse] Error in span: {e}")
            span = SpanContext(
                name=name,
                span_type=span_type,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent.span_id if parent else None,
                start_time=datetime.now(),
                metadata={},
            )
            yield span

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

        try:
            langfuse_obj.update(
                model=model,
                input=messages,
                output=output_content,
                usage=usage,
            )
        except Exception as e:
            print(f"[Langfuse] Error logging LLM call: {e}")

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
            try:
                langfuse_obj.update(
                    input={"query": query},
                    output={"documents": documents, "scores": scores},
                )
            except Exception as e:
                print(f"[Langfuse] Error logging retrieval: {e}")

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        span.error = str(error)
        langfuse_obj = span.metadata.get("langfuse_span")
        if langfuse_obj:
            try:
                langfuse_obj.update(
                    level="ERROR",
                    status_message=str(error),
                )
            except Exception as e:
                print(f"[Langfuse] Error logging error: {e}")

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

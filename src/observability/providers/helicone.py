"""Helicone observability provider."""

import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from src.observability.base import ObservabilityProvider, SpanContext, SpanType


class HeliconeProvider(ObservabilityProvider):
    """
    Helicone integration.

    Uses helicone-helpers for manual logging.
    """

    name = "helicone"
    supports_streaming = True
    supports_async = True

    def __init__(self):
        self.logger = None
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize Helicone."""
        api_key = os.getenv("HELICONE_API_KEY")
        if not api_key:
            print("[Helicone] No API key found (HELICONE_API_KEY)")
            return False

        try:
            # We use direct REST API logging to avoid dependency issues
            # and ensure full control over the payload structure.
            import requests
            self.initialized = True
            print("[Helicone] Initialized (REST API mode)")
            return True
        except ImportError:
            print("[Helicone] requests package not installed")
            return False
        except Exception as e:
            print(f"[Helicone] Failed to initialize: {e}")
            return False

    def shutdown(self) -> None:
        """No specific shutdown needed for Helicone logger."""
        pass

    @contextmanager
    def trace(self, name: str, **kwargs) -> Generator[SpanContext, None, None]:
        """Start a trace."""
        span = SpanContext(
            name=name,
            span_type=SpanType.TRACE,
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            start_time=datetime.now(),
            metadata={"helicone_trace": True},
        )
        try:
            yield span
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
        """Create a span."""
        span = SpanContext(
            name=name,
            span_type=span_type,
            trace_id=parent.trace_id if parent else str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=parent.span_id if parent else None,
            start_time=datetime.now(),
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
        """Log LLM call to Helicone."""
        if not self.initialized:
            return

        api_key = os.getenv("HELICONE_API_KEY")
        if not api_key:
            return

        # Prepare payload
        try:
            import json
            import threading
            import requests

            # Helicone expects:
            # {
            #   "providerRequest": { "url": "...", "json": { ... }, "meta": { ... } },
            #   "providerResponse": { "json": { ... }, "status": 200, "headers": { ... } }
            # }
            # Or simplified structure if using older endpoints, but standard custom-model logging is:
            # POST /v1/custom/v1/chat/completions (etc)
            # Actually, proper endpoint for generic logging is likely:
            # POST https://api.hconeai.com/custom/v1/log
            
            # Let's target the standard OpenAI-compatible logging if possible,
            # or simply structure it as a custom model log.
            
            # Constructing the body manually
            request_body = {
                "model": model,
                "messages": messages,
                "stream": False 
            }

            response_body = {}
            if hasattr(response, "choices"):
                choices = []
                for choice in response.choices:
                    msg_content = ""
                    if hasattr(choice, "message"):
                        msg_content = choice.message.content
                    elif hasattr(choice, "text"):
                        msg_content = choice.text
                    
                    choices.append({
                        "index": getattr(choice, "index", 0),
                        "message": {"role": "assistant", "content": msg_content},
                        "finish_reason": getattr(choice, "finish_reason", None)
                    })
                
                response_body = {"choices": choices}
                
                if hasattr(response, "usage") and response.usage:
                    response_body["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
            elif isinstance(response, str):
                 response_body = {"choices": [{"message": {"content": response}}]}

            # We use the custom logging endpoint structure for Helicone
            # payload = {
            #   "providerRequest": { "url": "custom-model-nolog", "json": request_body, "meta": ... },
            #   "providerResponse": { "json": response_body, "status": 200 }
            # }

            payload = {
                "providerRequest": {
                    "url": "https://api.openai.com/v1/chat/completions", # Pretend to be OpenAI for visualization
                    "json": request_body,
                    "meta": {
                        "Helicone-Auth": f"Bearer {api_key}",
                        "Helicone-Property-TraceId": span.trace_id,
                        "Helicone-Property-SpanId": span.span_id,
                    }
                },
                "providerResponse": {
                    "json": response_body,
                    "status": 200,
                    "headers": {"content-type": "application/json"}
                },
                "timing": {
                    # Optional timing info
                }
            }
            
            def _send():
                try:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    # Helper endpoint for logging custom requests
                    # Using the standard log endpoint
                    res = requests.post(
                        "https://api.hconeai.com/custom/v1/log",
                        headers=headers,
                        json=payload,
                        timeout=5
                    )
                    if res.status_code >= 400:
                        print(f"[Helicone] Failed to log: {res.status_code} {res.text}")
                except Exception as e:
                    print(f"[Helicone] Error sending log: {e}")

            # Send in background thread to avoid blocking
            threading.Thread(target=_send).start()

        except Exception as e:
            print(f"[Helicone] Error preparing log: {e}")

    def log_retrieval(self, span: SpanContext, query: str, documents: list[dict], scores: list[float], **kwargs) -> None:
        """Not supported primarily by Helicone."""
        pass

    def log_error(self, span: SpanContext, error: Exception) -> None:
        """Log error."""
        pass

    def get_trace_url(self, trace_id: str) -> str | None:
        """Get URL to view trace."""
        # Helicone uses request IDs mostly, but we can search by property if we added it
        return f"https://www.helicone.ai/requests?property=trace_id&value={trace_id}"

    def supports_feature(self, feature: str) -> bool:
        """Check feature support."""
        features = {
            "streaming": True,
            "async": True,
            "nested_spans": False,
            "cost_tracking": True,
            "evaluations": False,
        }
        return features.get(feature, False)

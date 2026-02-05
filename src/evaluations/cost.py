"""Cost estimation and verification."""

import tiktoken

from src.config import config


class CostTracker:
    """Tracks token usage and estimates costs locally."""

    # Pricing per 1M tokens (USD)
    PRICING = {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    }

    def __init__(self):
        try:
            self.encoding = tiktoken.encoding_for_model(config.llm_model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate estimated cost in USD."""
        if model not in self.PRICING:
            # Fallback for unknown models, try to match usage intent or return 0
            if "mini" in model:
                pricing = self.PRICING["gpt-4o-mini"]
            elif "gpt-4" in model:
                pricing = self.PRICING["gpt-4o"]
            elif "embedding" in model:
                pricing = self.PRICING["text-embedding-3-small"]
            else:
                return 0.0
        else:
            pricing = self.PRICING[model]

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost

    def calculate_tokens(self, text: str) -> int:
        """Count tokens in text string."""
        return len(self.encoding.encode(text))

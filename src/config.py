"""Configuration management for PDF Knowledge Explorer."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

ObservabilityProvider = Literal[
    "langsmith", "langfuse", "arize", "opik", "braintrust", "laminar", "agentops", "evidently", "logfire", "weave", "honeycomb"
]


@dataclass
class Config:
    """Application configuration."""

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    pdf_dir: Path = field(default_factory=lambda: Path("data/pdfs"))
    chroma_dir: Path = field(default_factory=lambda: Path("data/chroma"))
    chroma_host: str = field(default_factory=lambda: os.getenv("CHROMA_HOST", ""))
    chroma_port: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8000")))

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    rerank_top_k: int = 5

    # Observability
    observability_providers: list[ObservabilityProvider] = field(default_factory=list)
    observability_project_name: str = field(
        default_factory=lambda: os.getenv("OBSERVABILITY_PROJECT_NAME", "compare-observability")
    )

    def __post_init__(self):
        """Parse observability providers from environment."""
        providers_str = os.getenv("OBSERVABILITY_PROVIDERS", "")
        if providers_str:
            self.observability_providers = [p.strip() for p in providers_str.split(",") if p.strip()]

        # Ensure directories exist
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()

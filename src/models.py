"""Data models for PDF Knowledge Explorer."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class QueryIntent(Enum):
    """Types of user query intents."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """Represents a PDF document."""
    id: str
    filename: str
    title: str | None = None
    page_count: int = 0
    ingested_at: datetime | None = None


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    id: str
    document_id: str
    text: str
    page_number: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector search with score."""
    chunk: Chunk
    score: float
    rank: int


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    original_query: str
    intent: QueryIntent
    entities: list[str]
    expanded_query: str | None = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class ConceptRelation:
    """A relation between two concepts extracted from text."""
    source: str
    target: str
    relation_type: str
    confidence: float
    source_text: str


@dataclass
class GeneratedResponse:
    """The final generated response."""
    answer: str
    citations: list[dict[str, Any]]
    concepts: list[ConceptRelation]
    token_usage: dict[str, int]
    latency_ms: float


@dataclass
class TraceMetrics:
    """Metrics collected during a trace for comparison."""
    provider: str
    trace_id: str
    total_latency_ms: float
    span_count: int
    token_count: int
    estimated_cost: float
    errors: list[str] = field(default_factory=list)

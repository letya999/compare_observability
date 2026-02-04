"""Test scenario definitions for observability comparison."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ScenarioType(Enum):
    """Types of test scenarios."""
    SIMPLE_RAG = "simple_rag"
    MULTI_HOP = "multi_hop"
    LONG_CONTEXT = "long_context"
    STREAMING = "streaming"
    ERROR_HANDLING = "error_handling"
    EVALUATION = "evaluation"


class CheckType(Enum):
    """Types of checks to perform."""
    TRACE_COMPLETENESS = "trace_completeness"
    TOKEN_COUNT = "token_count"
    LATENCY = "latency"
    PARALLEL_SPANS = "parallel_spans"
    CROSS_DOC_RETRIEVAL = "cross_doc_retrieval"
    LARGE_PAYLOAD = "large_payload"
    STREAMING_SPANS = "streaming_spans"
    TIME_TO_FIRST_TOKEN = "time_to_first_token"
    ERROR_TRACE = "error_trace"
    BATCH_EVAL = "batch_eval"
    COST_CALCULATION = "cost_calculation"


@dataclass
class Scenario:
    """A test scenario definition."""
    name: str
    type: ScenarioType
    description: str
    query: str | list[str]
    expected_spans: int
    checks: list[CheckType]
    docs: list[str] = field(default_factory=list)
    stream: bool = False
    context_tokens: int | None = None
    force_error: str | None = None
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    scenario: Scenario
    success: bool
    provider_results: dict[str, dict[str, Any]]
    check_results: dict[str, bool]
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


# Define all test scenarios
SCENARIOS = {
    "simple_rag": Scenario(
        name="simple_rag",
        type=ScenarioType.SIMPLE_RAG,
        description="Basic RAG query to test fundamental tracing capabilities",
        query="What is the attention mechanism in transformers?",
        docs=["attention_paper.pdf"],
        expected_spans=5,
        checks=[
            CheckType.TRACE_COMPLETENESS,
            CheckType.TOKEN_COUNT,
            CheckType.LATENCY,
        ],
        metadata={
            "purpose": "Test basic trace structure, input/output capture, and span hierarchy",
        },
    ),

    "multi_hop": Scenario(
        name="multi_hop",
        type=ScenarioType.MULTI_HOP,
        description="Multi-document query requiring cross-document retrieval",
        query="How does BERT's attention mechanism differ from GPT's attention?",
        docs=["bert_paper.pdf", "gpt_paper.pdf"],
        expected_spans=7,  # 2x retrieval
        checks=[
            CheckType.PARALLEL_SPANS,
            CheckType.CROSS_DOC_RETRIEVAL,
            CheckType.TRACE_COMPLETENESS,
        ],
        metadata={
            "purpose": "Test parallel span handling and complex trace visualization",
        },
    ),

    "long_context": Scenario(
        name="long_context",
        type=ScenarioType.LONG_CONTEXT,
        description="Query with large context to test payload handling",
        query="Summarize the entire methodology section with all key details",
        docs=["long_paper.pdf"],
        expected_spans=5,
        context_tokens=30000,
        checks=[
            CheckType.LARGE_PAYLOAD,
            CheckType.TOKEN_COUNT,
            CheckType.COST_CALCULATION,
        ],
        metadata={
            "purpose": "Test large payload handling, truncation behavior, and cost tracking accuracy",
        },
    ),

    "streaming": Scenario(
        name="streaming",
        type=ScenarioType.STREAMING,
        description="Streaming response to test real-time trace updates",
        query="Explain the transformer architecture in detail, covering all major components",
        docs=["attention_paper.pdf"],
        expected_spans=5,
        stream=True,
        checks=[
            CheckType.STREAMING_SPANS,
            CheckType.TIME_TO_FIRST_TOKEN,
            CheckType.TRACE_COMPLETENESS,
        ],
        metadata={
            "purpose": "Test streaming support, TTFT tracking, and incremental updates",
        },
    ),

    "error_handling": Scenario(
        name="error_handling",
        type=ScenarioType.ERROR_HANDLING,
        description="Forced error to test error tracing capabilities",
        query="Find information about XYZ nonexistent topic",
        docs=[],  # No docs = empty retrieval
        expected_spans=5,
        force_error="empty_retrieval",
        checks=[
            CheckType.ERROR_TRACE,
            CheckType.TRACE_COMPLETENESS,
        ],
        metadata={
            "purpose": "Test error capture, error messages, and trace status handling",
            "error_types": ["empty_retrieval", "llm_timeout", "rate_limit"],
        },
    ),

    "evaluation": Scenario(
        name="evaluation",
        type=ScenarioType.EVALUATION,
        description="Batch evaluation run with ground truth",
        query=[
            "What is self-attention?",
            "How do transformers handle sequences?",
            "What is the role of positional encoding?",
            "Explain multi-head attention",
            "What are the benefits of attention over RNNs?",
        ],
        docs=["attention_paper.pdf"],
        expected_spans=25,  # 5 queries x 5 spans
        checks=[
            CheckType.BATCH_EVAL,
            CheckType.TRACE_COMPLETENESS,
        ],
        ground_truth="evaluation_ground_truth.json",
        metadata={
            "purpose": "Test batch evaluation UI, dataset handling, and metrics dashboard",
        },
    ),
}


def get_scenario(name: str) -> Scenario:
    """Get a scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def list_scenarios() -> list[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())

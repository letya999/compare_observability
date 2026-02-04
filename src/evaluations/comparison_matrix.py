"""Comparison matrix for observability platforms."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class CriteriaCategory(Enum):
    """Categories of comparison criteria."""
    SETUP = "setup_integration"
    TRACING = "tracing_features"
    LLM = "llm_specific"
    RETRIEVAL = "retrieval_specific"
    EVALUATIONS = "evaluations"
    PRODUCTION = "production_ui"
    BUSINESS = "business"


@dataclass
class Criterion:
    """A single comparison criterion."""
    name: str
    category: CriteriaCategory
    description: str
    value_type: str  # "boolean", "number", "rating", "text"
    max_value: int | None = None  # For ratings


@dataclass
class PlatformScore:
    """Score for a platform on a criterion."""
    platform: str
    criterion: str
    value: Any
    notes: str = ""
    evidence_url: str | None = None


# Define all comparison criteria
CRITERIA = {
    # A. Setup & Integration
    "time_to_first_trace": Criterion(
        "Time to first trace (min)",
        CriteriaCategory.SETUP,
        "Minutes from signup to seeing first trace in UI",
        "number",
    ),
    "lines_of_code": Criterion(
        "Lines of code for integration",
        CriteriaCategory.SETUP,
        "Number of lines needed for basic integration",
        "number",
    ),
    "auto_instrument_openai": Criterion(
        "Auto-instrumentation OpenAI",
        CriteriaCategory.SETUP,
        "Supports automatic OpenAI SDK instrumentation",
        "boolean",
    ),
    "auto_instrument_langchain": Criterion(
        "Auto-instrumentation LangChain",
        CriteriaCategory.SETUP,
        "Supports automatic LangChain instrumentation",
        "boolean",
    ),
    "docs_quality": Criterion(
        "Documentation quality",
        CriteriaCategory.SETUP,
        "Quality of documentation (1-5)",
        "rating",
        max_value=5,
    ),

    # B. Tracing Features
    "nested_spans": Criterion(
        "Nested spans",
        CriteriaCategory.TRACING,
        "Supports hierarchical span nesting",
        "boolean",
    ),
    "parallel_spans": Criterion(
        "Parallel spans",
        CriteriaCategory.TRACING,
        "Properly visualizes parallel execution",
        "boolean",
    ),
    "span_metadata": Criterion(
        "Span metadata",
        CriteriaCategory.TRACING,
        "Supports custom metadata on spans",
        "boolean",
    ),
    "input_output_capture": Criterion(
        "Input/Output capture",
        CriteriaCategory.TRACING,
        "Automatically captures inputs and outputs",
        "boolean",
    ),
    "streaming_support": Criterion(
        "Streaming support",
        CriteriaCategory.TRACING,
        "Properly handles streaming responses",
        "boolean",
    ),
    "error_traces": Criterion(
        "Error traces",
        CriteriaCategory.TRACING,
        "Captures and displays errors properly",
        "boolean",
    ),

    # C. LLM-Specific
    "token_counting": Criterion(
        "Token counting",
        CriteriaCategory.LLM,
        "Automatically counts tokens",
        "boolean",
    ),
    "cost_calculation": Criterion(
        "Cost calculation",
        CriteriaCategory.LLM,
        "Automatically calculates API costs",
        "boolean",
    ),
    "prompt_completion_split": Criterion(
        "Prompt/completion split",
        CriteriaCategory.LLM,
        "Shows prompt and completion separately",
        "boolean",
    ),
    "model_attribution": Criterion(
        "Model attribution",
        CriteriaCategory.LLM,
        "Tracks which model was used",
        "boolean",
    ),
    "multi_provider_support": Criterion(
        "Multi-provider support",
        CriteriaCategory.LLM,
        "Supports multiple LLM providers",
        "boolean",
    ),

    # D. Retrieval-Specific
    "retrieved_docs_display": Criterion(
        "Retrieved docs display",
        CriteriaCategory.RETRIEVAL,
        "Shows retrieved documents in UI",
        "boolean",
    ),
    "relevance_scores": Criterion(
        "Relevance scores",
        CriteriaCategory.RETRIEVAL,
        "Displays relevance/similarity scores",
        "boolean",
    ),
    "chunk_preview": Criterion(
        "Chunk preview",
        CriteriaCategory.RETRIEVAL,
        "Shows preview of retrieved chunks",
        "boolean",
    ),
    "source_metadata": Criterion(
        "Source metadata",
        CriteriaCategory.RETRIEVAL,
        "Displays document source metadata",
        "boolean",
    ),

    # E. Evaluations
    "builtin_evals": Criterion(
        "Built-in evaluations",
        CriteriaCategory.EVALUATIONS,
        "Has built-in evaluation metrics",
        "boolean",
    ),
    "custom_eval_functions": Criterion(
        "Custom eval functions",
        CriteriaCategory.EVALUATIONS,
        "Supports custom evaluation code",
        "boolean",
    ),
    "human_feedback_ui": Criterion(
        "Human feedback UI",
        CriteriaCategory.EVALUATIONS,
        "Has UI for human feedback/labeling",
        "boolean",
    ),
    "dataset_management": Criterion(
        "Dataset management",
        CriteriaCategory.EVALUATIONS,
        "Supports dataset creation and management",
        "boolean",
    ),
    "ab_comparison": Criterion(
        "A/B comparison",
        CriteriaCategory.EVALUATIONS,
        "Supports comparing different versions",
        "boolean",
    ),
    "regression_detection": Criterion(
        "Regression detection",
        CriteriaCategory.EVALUATIONS,
        "Alerts on performance regressions",
        "boolean",
    ),

    # F. Production & UI
    "latency_dashboards": Criterion(
        "Latency dashboards",
        CriteriaCategory.PRODUCTION,
        "Has latency monitoring dashboards",
        "boolean",
    ),
    "cost_aggregation": Criterion(
        "Cost aggregation",
        CriteriaCategory.PRODUCTION,
        "Aggregates costs over time",
        "boolean",
    ),
    "search_filtering": Criterion(
        "Search & filtering",
        CriteriaCategory.PRODUCTION,
        "Can search and filter traces",
        "boolean",
    ),
    "trace_comparison": Criterion(
        "Trace comparison",
        CriteriaCategory.PRODUCTION,
        "Can compare two traces side by side",
        "boolean",
    ),
    "alerting": Criterion(
        "Alerting",
        CriteriaCategory.PRODUCTION,
        "Supports alerts and notifications",
        "boolean",
    ),
    "ui_responsiveness": Criterion(
        "UI responsiveness",
        CriteriaCategory.PRODUCTION,
        "How fast and responsive is the UI (1-5)",
        "rating",
        max_value=5,
    ),
    "ui_design_quality": Criterion(
        "UI design quality",
        CriteriaCategory.PRODUCTION,
        "Visual design and UX quality (1-5)",
        "rating",
        max_value=5,
    ),

    # G. Business
    "self_hosted_option": Criterion(
        "Self-hosted option",
        CriteriaCategory.BUSINESS,
        "Can be self-hosted",
        "boolean",
    ),
    "free_tier_limits": Criterion(
        "Free tier limits",
        CriteriaCategory.BUSINESS,
        "Description of free tier",
        "text",
    ),
    "pricing_per_trace": Criterion(
        "Pricing per trace",
        CriteriaCategory.BUSINESS,
        "Cost per trace/span",
        "text",
    ),
    "enterprise_features": Criterion(
        "Enterprise features",
        CriteriaCategory.BUSINESS,
        "Has enterprise features (SSO, audit, etc)",
        "boolean",
    ),
}

# Platform abbreviations
PLATFORMS = {
    "langsmith": "LS",
    "langfuse": "LF",
    "arize": "AR",
    "opik": "OP",
    "braintrust": "BT",
    "laminar": "LA",
    "agentops": "AO",
    "evidently": "EV",
    "logfire": "PY",
}


class ComparisonMatrix:
    """
    Manages the comparison matrix for observability platforms.

    Usage:
        matrix = ComparisonMatrix()
        matrix.set_score("langsmith", "token_counting", True, "Automatic via SDK")
        matrix.export_markdown("comparison.md")
    """

    def __init__(self):
        self.criteria = CRITERIA
        self.platforms = list(PLATFORMS.keys())
        self.scores: dict[str, dict[str, PlatformScore]] = {
            platform: {} for platform in self.platforms
        }
        self.last_updated = datetime.now()

    def set_score(
        self,
        platform: str,
        criterion: str,
        value: Any,
        notes: str = "",
        evidence_url: str | None = None,
    ) -> None:
        """Set a score for a platform on a criterion."""
        if platform not in self.platforms:
            raise ValueError(f"Unknown platform: {platform}")
        if criterion not in self.criteria:
            raise ValueError(f"Unknown criterion: {criterion}")

        self.scores[platform][criterion] = PlatformScore(
            platform=platform,
            criterion=criterion,
            value=value,
            notes=notes,
            evidence_url=evidence_url,
        )
        self.last_updated = datetime.now()

    def get_score(self, platform: str, criterion: str) -> PlatformScore | None:
        """Get a score for a platform on a criterion."""
        return self.scores.get(platform, {}).get(criterion)

    def get_category_scores(self, category: CriteriaCategory) -> dict[str, dict[str, Any]]:
        """Get all scores for a category."""
        category_criteria = [
            name for name, c in self.criteria.items() if c.category == category
        ]

        result = {}
        for criterion in category_criteria:
            result[criterion] = {}
            for platform in self.platforms:
                score = self.get_score(platform, criterion)
                result[criterion][platform] = score.value if score else None

        return result

    def export_markdown(self, output_path: Path | str) -> None:
        """Export the matrix as a Markdown file."""
        output_path = Path(output_path)
        lines = []

        lines.append("# Observability Platform Comparison Matrix\n")
        lines.append(f"Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M')}\n")
        lines.append("\n## Platform Legend\n")

        for platform, abbrev in PLATFORMS.items():
            lines.append(f"- **{abbrev}**: {platform.title()}")
        lines.append("\n")

        # Group by category
        for category in CriteriaCategory:
            lines.append(f"\n## {category.value.replace('_', ' ').title()}\n")

            # Table header
            header = "| Criterion |"
            separator = "|-----------|"
            for abbrev in PLATFORMS.values():
                header += f" {abbrev} |"
                separator += "----|"

            lines.append(header)
            lines.append(separator)

            # Table rows
            for criterion_name, criterion in self.criteria.items():
                if criterion.category != category:
                    continue

                row = f"| {criterion.name} |"
                for platform in self.platforms:
                    score = self.get_score(platform, criterion_name)
                    if score is None:
                        cell = " - "
                    elif criterion.value_type == "boolean":
                        cell = " Yes " if score.value else " No "
                    elif criterion.value_type == "rating":
                        cell = f" {score.value}/5 " if score.value else " - "
                    else:
                        cell = f" {score.value} " if score.value else " - "
                    row += cell + "|"

                lines.append(row)

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def export_json(self, output_path: Path | str) -> None:
        """Export the matrix as JSON."""
        output_path = Path(output_path)

        data = {
            "last_updated": self.last_updated.isoformat(),
            "platforms": list(PLATFORMS.keys()),
            "platform_abbreviations": PLATFORMS,
            "criteria": {
                name: {
                    "name": c.name,
                    "category": c.category.value,
                    "description": c.description,
                    "value_type": c.value_type,
                }
                for name, c in self.criteria.items()
            },
            "scores": {
                platform: {
                    criterion: {
                        "value": score.value,
                        "notes": score.notes,
                        "evidence_url": score.evidence_url,
                    }
                    for criterion, score in platform_scores.items()
                }
                for platform, platform_scores in self.scores.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def import_json(self, input_path: Path | str) -> None:
        """Import scores from JSON."""
        input_path = Path(input_path)

        with open(input_path) as f:
            data = json.load(f)

        for platform, platform_scores in data.get("scores", {}).items():
            for criterion, score_data in platform_scores.items():
                self.set_score(
                    platform,
                    criterion,
                    score_data.get("value"),
                    score_data.get("notes", ""),
                    score_data.get("evidence_url"),
                )

    def calculate_totals(self) -> dict[str, dict[str, Any]]:
        """Calculate totals and rankings for each platform."""
        totals = {}

        for platform in self.platforms:
            platform_scores = self.scores.get(platform, {})

            # Count boolean features
            bool_criteria = [
                name for name, c in self.criteria.items()
                if c.value_type == "boolean"
            ]
            bool_score = sum(
                1 for c in bool_criteria
                if platform_scores.get(c) and platform_scores[c].value
            )

            # Average ratings
            rating_criteria = [
                name for name, c in self.criteria.items()
                if c.value_type == "rating"
            ]
            ratings = [
                platform_scores[c].value
                for c in rating_criteria
                if platform_scores.get(c) and platform_scores[c].value
            ]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0

            totals[platform] = {
                "bool_features": bool_score,
                "bool_total": len(bool_criteria),
                "bool_percentage": bool_score / len(bool_criteria) * 100 if bool_criteria else 0,
                "avg_rating": avg_rating,
                "rated_criteria": len(ratings),
            }

        return totals

    def print_summary(self) -> None:
        """Print a summary of the comparison."""
        totals = self.calculate_totals()

        print("\n" + "="*60)
        print("COMPARISON MATRIX SUMMARY")
        print("="*60)

        # Sort by feature count
        sorted_platforms = sorted(
            totals.items(),
            key=lambda x: (x[1]["bool_features"], x[1]["avg_rating"]),
            reverse=True,
        )

        for platform, scores in sorted_platforms:
            print(f"\n{platform.upper()}")
            print(f"  Features: {scores['bool_features']}/{scores['bool_total']} ({scores['bool_percentage']:.0f}%)")
            if scores['rated_criteria'] > 0:
                print(f"  Avg Rating: {scores['avg_rating']:.1f}/5")

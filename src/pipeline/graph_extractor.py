"""Graph extraction module - Step 5 of the RAG pipeline."""

import json
from openai import OpenAI

from src.config import config
from src.models import ConceptRelation, GeneratedResponse


GRAPH_EXTRACTION_PROMPT = """Analyze the following text and extract concept relationships.
Identify key concepts and how they relate to each other.

Text:
{text}

Extract relationships in the following JSON format:
{{
    "relationships": [
        {{
            "source": "concept1",
            "target": "concept2",
            "relation_type": "type of relation (e.g., 'is_part_of', 'causes', 'requires', 'similar_to', 'contrasts_with')",
            "confidence": 0.0-1.0,
            "source_text": "brief quote from text supporting this relation"
        }}
    ]
}}

Focus on:
- Technical concepts and their relationships
- Hierarchical relationships (is_part_of, contains)
- Causal relationships (causes, enables, requires)
- Comparative relationships (similar_to, contrasts_with, extends)

Extract 3-7 most important relationships."""


class GraphExtractor:
    """Extracts concept relationships from generated responses."""

    def __init__(self, openai_client: OpenAI | None = None):
        self.client = openai_client or OpenAI(api_key=config.openai_api_key)

    def extract(self, response: GeneratedResponse) -> list[ConceptRelation]:
        """
        Extract concept relationships from a generated response.

        Args:
            response: The generated response to analyze

        Returns:
            List of ConceptRelation objects
        """
        if not response.answer:
            return []

        llm_response = self.client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledge graph extraction assistant. Extract concept relationships from text. Always respond with valid JSON.",
                },
                {
                    "role": "user",
                    "content": GRAPH_EXTRACTION_PROMPT.format(text=response.answer),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(llm_response.choices[0].message.content)
        relationships = result.get("relationships", [])

        return [
            ConceptRelation(
                source=rel["source"],
                target=rel["target"],
                relation_type=rel["relation_type"],
                confidence=rel.get("confidence", 0.8),
                source_text=rel.get("source_text", ""),
            )
            for rel in relationships
        ]

    def extract_from_chunks(self, chunks_text: list[str]) -> list[ConceptRelation]:
        """Extract relationships from multiple text chunks."""
        combined_text = "\n\n".join(chunks_text[:5])  # Limit to avoid token limits

        # Create a temporary response object
        temp_response = GeneratedResponse(
            answer=combined_text,
            citations=[],
            concepts=[],
            token_usage={},
            latency_ms=0,
        )

        return self.extract(temp_response)

"""Query analysis module - Step 1 of the RAG pipeline."""

import json
from openai import OpenAI

from src.config import config
from src.models import QueryAnalysis, QueryIntent


QUERY_ANALYSIS_PROMPT = """Analyze the following user query and extract:
1. Intent: What type of question is this? (factual, comparison, summary, explanation, unknown)
2. Entities: Key entities or concepts mentioned
3. Keywords: Important search keywords
4. Expanded query: A reformulated version that might retrieve better results

Respond in JSON format:
{
    "intent": "factual|comparison|summary|explanation|unknown",
    "entities": ["entity1", "entity2"],
    "keywords": ["keyword1", "keyword2"],
    "expanded_query": "reformulated query for better retrieval"
}

User query: {query}"""


class QueryAnalyzer:
    """Analyzes user queries to determine intent and extract entities."""

    def __init__(self, openai_client: OpenAI | None = None):
        self.client = openai_client or OpenAI(api_key=config.openai_api_key)

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query.

        Args:
            query: The user's natural language query

        Returns:
            QueryAnalysis with intent, entities, and expanded query
        """
        response = self.client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a query analysis assistant. Always respond with valid JSON.",
                },
                {
                    "role": "user",
                    "content": QUERY_ANALYSIS_PROMPT.format(query=query),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        intent_map = {
            "factual": QueryIntent.FACTUAL,
            "comparison": QueryIntent.COMPARISON,
            "summary": QueryIntent.SUMMARY,
            "explanation": QueryIntent.EXPLANATION,
        }

        return QueryAnalysis(
            original_query=query,
            intent=intent_map.get(result.get("intent", "unknown"), QueryIntent.UNKNOWN),
            entities=result.get("entities", []),
            keywords=result.get("keywords", []),
            expanded_query=result.get("expanded_query"),
        )

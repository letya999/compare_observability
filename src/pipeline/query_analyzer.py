"""Query analysis module - Step 1 of the RAG pipeline."""

import json
from openai import OpenAI

from src.config import config
from src.logger import logger
from src.models import QueryAnalysis, QueryIntent


QUERY_ANALYSIS_PROMPT = """Analyze the following user query and extract:
1. Intent: What type of question is this? (factual, comparison, summary, explanation, unknown)
2. Entities: Key entities or concepts mentioned
3. Keywords: Important search keywords
4. Expanded query: A reformulated version that might retrieve better results. 
   IMPORTANT: Do NOT hallucinate or add specific source names (like book titles, movie names, specific documents) to the expanded query unless they are explicitly mentioned in the user query.
   If the query is about a character or event, do not guess the book/source. Keep it general if unknown.

Respond in JSON format:
{{
    "intent": "factual|comparison|summary|explanation|unknown",
    "entities": ["entity1", "entity2"],
    "keywords": ["keyword1", "keyword2"],
    "expanded_query": "reformulated query for better retrieval"
}}

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


        try:
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Query analysis result: {result}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse query analysis JSON. Raw content: {response.choices[0].message.content}")
            logger.error(f"JSON Error: {e}")
            # Fallback to defaults
            return QueryAnalysis(
                original_query=query,
                intent=QueryIntent.UNKNOWN,
                entities=[],
                keywords=[],
                expanded_query=query
            )

        intent_map = {
            "factual": QueryIntent.FACTUAL,
            "comparison": QueryIntent.COMPARISON,
            "summary": QueryIntent.SUMMARY,
            "explanation": QueryIntent.EXPLANATION,
            "unknown": QueryIntent.UNKNOWN,
        }

        # Clean up keys if necessary (handle potential whitespace issues from LLM)
        cleaned_result = {k.strip(): v for k, v in result.items()} if isinstance(result, dict) else {}
        
        intent_str = cleaned_result.get("intent", "unknown")
        if isinstance(intent_str, str):
            intent_str = intent_str.lower().strip()
            
        intent_enum = intent_map.get(intent_str, QueryIntent.UNKNOWN)

        return QueryAnalysis(
            original_query=query,
            intent=intent_enum,
            entities=cleaned_result.get("entities", []),
            keywords=cleaned_result.get("keywords", []),
            expanded_query=cleaned_result.get("expanded_query"),
        )

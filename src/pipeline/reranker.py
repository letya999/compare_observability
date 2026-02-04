"""Reranking module - Step 3 of the RAG pipeline."""

import json
from openai import OpenAI

from src.config import config
from src.models import QueryAnalysis, RetrievedChunk


RERANK_PROMPT = """You are a relevance scoring assistant. Given a query and a list of text chunks,
score each chunk's relevance to the query on a scale of 0-10.

Query: {query}

Chunks to score:
{chunks}

Respond with a JSON array of objects with "chunk_id" and "score" fields:
[
    {{"chunk_id": "id1", "score": 8.5}},
    {{"chunk_id": "id2", "score": 3.2}}
]

Score based on:
- Direct relevance to the query (0-4 points)
- Information completeness (0-3 points)
- Specificity and detail (0-3 points)"""


class Reranker:
    """Reranks retrieved chunks using LLM-based scoring."""

    def __init__(self, openai_client: OpenAI | None = None):
        self.client = openai_client or OpenAI(api_key=config.openai_api_key)

    def rerank(
        self,
        query_analysis: QueryAnalysis,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank retrieved chunks using LLM scoring.

        Args:
            query_analysis: The analyzed query
            chunks: Retrieved chunks to rerank
            top_k: Number of top chunks to return after reranking

        Returns:
            Reranked list of chunks
        """
        if not chunks:
            return chunks

        k = top_k or config.rerank_top_k

        # Format chunks for the prompt
        chunks_text = "\n\n".join(
            f"[{c.chunk.id}]: {c.chunk.text[:500]}..." if len(c.chunk.text) > 500 else f"[{c.chunk.id}]: {c.chunk.text}"
            for c in chunks
        )

        response = self.client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a relevance scoring assistant. Always respond with valid JSON.",
                },
                {
                    "role": "user",
                    "content": RERANK_PROMPT.format(
                        query=query_analysis.original_query,
                        chunks=chunks_text,
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        # Parse scores - handle both array and object responses
        content = response.choices[0].message.content
        result = json.loads(content)

        # Handle case where LLM returns {"scores": [...]} instead of [...]
        if isinstance(result, dict):
            scores_list = result.get("scores", result.get("results", []))
        else:
            scores_list = result

        scores = {item["chunk_id"]: item["score"] for item in scores_list}

        # Update scores and rerank
        for chunk in chunks:
            if chunk.chunk.id in scores:
                chunk.score = scores[chunk.chunk.id] / 10.0  # Normalize to 0-1

        # Sort by new score and return top_k
        reranked = sorted(chunks, key=lambda x: x.score, reverse=True)[:k]

        # Update ranks
        for i, chunk in enumerate(reranked):
            chunk.rank = i + 1

        return reranked

"""Response generation module - Step 4 of the RAG pipeline."""

import time
from collections.abc import Generator as GenType
from typing import Any

import tiktoken
from openai import OpenAI

from src.config import config
from src.models import GeneratedResponse, QueryAnalysis, RetrievedChunk


GENERATION_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the context to answer the question accurately. Always cite your sources using [1], [2], etc.

Context:
{context}

Reasoning Context (Tool Outputs):
{reasoning_context}

Question: {query}

Instructions:
1. Answer the question based on the provided context AND reasoning context
2. If the context doesn't contain enough information, say so
3. Cite specific chunks using [1], [2], etc. notation
4. Be concise but comprehensive"""


class Generator:
    """Generates responses using retrieved context."""

    def __init__(self, openai_client: OpenAI | None = None):
        self.client = openai_client or OpenAI(api_key=config.openai_api_key)
        self.encoding = tiktoken.encoding_for_model(config.llm_model)

    def generate(
        self,
        query_analysis: QueryAnalysis,
        chunks: list[RetrievedChunk],
        reasoning_steps: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> GeneratedResponse | GenType[str | GeneratedResponse, None, None]:
        """
        Generate a response based on query and retrieved context.

        Args:
            query_analysis: The analyzed query
            chunks: Reranked chunks to use as context
            reasoning_steps: Optional tool outputs from reasoning engine
            stream: Whether to stream the response

        Returns:
            GeneratedResponse or generator yielding chunks (str) then final response (GeneratedResponse)
        """
        # Format context
        context = self._format_context(chunks)

        # Format reasoning context
        reasoning_context = ""
        if reasoning_steps:
             for step in reasoning_steps:
                 reasoning_context += f"- Used tool {step['tool']} with input {step['input']} -> Result: {step['output']}\n"
        
        if not reasoning_context:
            reasoning_context = "No specific tool outputs used."

        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable assistant. Answer questions accurately based on provided context.",
            },
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(
                    context=context,
                    reasoning_context=reasoning_context,
                    query=query_analysis.original_query,
                ),
            },
        ]

        if stream:
            return self._generate_streaming(messages, chunks)
        else:
            return self._generate_sync(messages, chunks)

    def _generate_sync(
        self, messages: list[dict], chunks: list[RetrievedChunk]
    ) -> GeneratedResponse:
        """Generate response synchronously."""
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=0.3,
        )

        latency_ms = (time.time() - start_time) * 1000
        answer = response.choices[0].message.content

        return GeneratedResponse(
            answer=answer,
            citations=self._extract_citations(chunks),
            concepts=[],  # Will be filled by graph extractor
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            latency_ms=latency_ms,
        )

    def _generate_streaming(
        self, messages: list[dict], chunks: list[RetrievedChunk]
    ) -> GenType[str | GeneratedResponse, None, None]:
        """Generate response with streaming."""
        start_time = time.time()
        full_response = ""

        stream = self.client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            temperature=0.3,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
            if chunk.usage:
                usage = chunk.usage

        latency_ms = (time.time() - start_time) * 1000

        # Return final response object
        # Yield final response object so it can be captured by the loop
        yield GeneratedResponse(
            answer=full_response,
            citations=self._extract_citations(chunks),
            concepts=[],
            token_usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            latency_ms=latency_ms,
        )

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks as numbered context."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] (Page {chunk.chunk.page_number}, Score: {chunk.score:.2f})\n{chunk.chunk.text}"
            )
        return "\n\n".join(context_parts)

    def _extract_citations(self, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
        """Extract citation information from chunks."""
        return [
            {
                "index": i + 1,
                "document_id": chunk.chunk.document_id,
                "page_number": chunk.chunk.page_number,
                "text_preview": chunk.chunk.text[:200] + "..." if len(chunk.chunk.text) > 200 else chunk.chunk.text,
                "score": chunk.score,
            }
            for i, chunk in enumerate(chunks)
        ]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

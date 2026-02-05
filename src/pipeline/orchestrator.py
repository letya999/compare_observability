"""RAG Pipeline Orchestrator - coordinates all pipeline steps."""

import time
from collections.abc import Generator as GenType
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import config
from src.models import (
    Chunk,
    ConceptRelation,
    Document,
    GeneratedResponse,
    QueryAnalysis,
    RetrievedChunk,
)

from .generator import Generator
from .graph_extractor import GraphExtractor
from .ingestion import PDFIngestor
from .query_analyzer import QueryAnalyzer
from .query_analyzer import QueryAnalyzer
from .reranker import Reranker
from .reasoning import ReasoningEngine
from .retriever import Retriever


@dataclass
class PipelineResult:
    """Complete result from a RAG pipeline execution."""

    query: str
    query_analysis: QueryAnalysis
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    reasoning_steps: list[dict[str, Any]]
    response: GeneratedResponse
    concepts: list[ConceptRelation]
    total_latency_ms: float
    step_latencies: dict[str, float] = field(default_factory=dict)
    cost_estimate_usd: float = 0.0
    eval_metrics: dict[str, float] = field(default_factory=dict)


class RAGOrchestrator:
    """
    Orchestrates the full RAG pipeline with observability hooks.

    Pipeline steps:
    1. Query Analysis - Understand intent and extract entities
    2. Retrieval - Vector search for relevant chunks
    3. Reranking - LLM-based relevance scoring
    4. Generation - Generate response with citations
    5. Graph Extraction - Extract concept relationships
    """

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        persist_directory: Path | None = None,
    ):
        self.client = openai_client or OpenAI(api_key=config.openai_api_key)

        # Initialize pipeline components
        self.ingestor = PDFIngestor(self.client)
        self.query_analyzer = QueryAnalyzer(self.client)
        self.retriever = Retriever(self.client, persist_directory)
        self.reranker = Reranker(self.client)
        self.retriever = Retriever(self.client, persist_directory)
        self.reranker = Reranker(self.client)
        self.reasoning_engine = ReasoningEngine(self.client)
        self.generator = Generator(self.client)
        self.graph_extractor = GraphExtractor(self.client)

    def ingest_pdf(self, pdf_path: Path) -> tuple[Document, int]:
        """
        Ingest a PDF and store chunks in vector store.

        Returns:
            Tuple of (Document, number of chunks created)
        """
        doc, chunks = self.ingestor.ingest_pdf(pdf_path)
        self.retriever.add_chunks(chunks)
        return doc, len(chunks)

    def ingest_directory(self, directory: Path) -> list[tuple[Document, int]]:
        """Ingest all PDFs in a directory."""
        results = []
        for doc, chunks in self.ingestor.ingest_directory(directory):
            self.retriever.add_chunks(chunks)
            results.append((doc, len(chunks)))
        return results

    def query(
        self,
        query: str,
        filter_doc_ids: list[str] | None = None,
        stream: bool = False,
        skip_graph_extraction: bool = False,
        retrieval_only: bool = False,
    ) -> PipelineResult | GenType[str, None, PipelineResult]:
        """
        Execute the full RAG pipeline.

        Args:
            query: User's natural language query
            filter_doc_ids: Optional list of document IDs to search within
            stream: Whether to stream the response
            skip_graph_extraction: Skip the graph extraction step for speed

        Returns:
            PipelineResult with all intermediate and final results
        """
        total_start = time.time()
        step_latencies = {}

        # Step 1: Query Analysis
        step_start = time.time()
        query_analysis = self.query_analyzer.analyze(query)
        step_latencies["query_analysis"] = (time.time() - step_start) * 1000

        # Step 2: Retrieval
        step_start = time.time()
        retrieved_chunks = self.retriever.hybrid_retrieve(
            query_analysis, filter_doc_ids=filter_doc_ids
        )
        step_latencies["retrieval"] = (time.time() - step_start) * 1000

        # Step 3: Reranking
        step_start = time.time()
        reranked_chunks = self.reranker.rerank(query_analysis, retrieved_chunks)
        step_latencies["reranking"] = (time.time() - step_start) * 1000

        step_latencies["reranking"] = (time.time() - step_start) * 1000

        # Step 4: Reasoning (Agentic Tools)
        step_start = time.time()
        reasoning_steps = []
        if not retrieval_only:
            reasoning_steps = self.reasoning_engine.run(query_analysis, reranked_chunks)
        step_latencies["reasoning"] = (time.time() - step_start) * 1000

        # Step 5: Generation
        step_start = time.time()
        response = None
        concepts = []

        if not retrieval_only:
            if stream:
                return self._stream_response(
                    query,
                    query_analysis,
                    retrieved_chunks,
                    reranked_chunks,
                    reasoning_steps,
                    step_latencies,
                    total_start,
                    skip_graph_extraction,
                )
            else:
                response = self.generator.generate(
                    query_analysis, reranked_chunks, reasoning_steps=reasoning_steps
                )
                step_latencies["generation"] = (time.time() - step_start) * 1000

                # Step 6: Graph Extraction
                if not skip_graph_extraction:
                    step_start = time.time()
                    concepts = self.graph_extractor.extract(response)
                    step_latencies["graph_extraction"] = (time.time() - step_start) * 1000
        
        # If retrieval_only, creating dummy response
        if retrieval_only:
             response = GeneratedResponse(
                 answer="**Retrieval Only Mode**: Generation skipped.",
                 citations=[],
                 token_usage={},
                 model="skipped"
             )

        total_latency = (time.time() - total_start) * 1000

        return PipelineResult(
            query=query,
            query_analysis=query_analysis,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            reasoning_steps=reasoning_steps,
            response=response,
            concepts=concepts,
            total_latency_ms=total_latency,
            step_latencies=step_latencies,
        )

    def _stream_response(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        retrieved_chunks: list[RetrievedChunk],
        reranked_chunks: list[RetrievedChunk],
        reasoning_steps: list[dict[str, Any]],
        step_latencies: dict[str, float],
        total_start: float,
        skip_graph_extraction: bool,
    ) -> GenType[str, None, PipelineResult]:
        """Stream response and yield chunks."""
        step_start = time.time()
        response_gen = self.generator.generate(
            query_analysis, 
            reranked_chunks, 
            reasoning_steps=reasoning_steps, 
            stream=True
        )

        full_response = ""
        for chunk in response_gen:
            if isinstance(chunk, str):
                full_response += chunk
                yield chunk
            else:
                # Final response object
                response = chunk

        step_latencies["generation"] = (time.time() - step_start) * 1000

        # Graph extraction after streaming
        concepts = []
        if not skip_graph_extraction:
            step_start = time.time()
            concepts = self.graph_extractor.extract(response)
            step_latencies["graph_extraction"] = (time.time() - step_start) * 1000

        total_latency = (time.time() - total_start) * 1000

        return PipelineResult(
            query=query,
            query_analysis=query_analysis,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            reasoning_steps=reasoning_steps,
            response=response,
            concepts=concepts,
            total_latency_ms=total_latency,
            step_latencies=step_latencies,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "vector_store": self.retriever.get_collection_stats(),
            "config": {
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model,
                "chunk_size": config.chunk_size,
                "top_k": config.top_k,
                "rerank_top_k": config.rerank_top_k,
            },
        }

    def clear_data(self) -> None:
        """Clear all stored data."""
        self.retriever.clear()

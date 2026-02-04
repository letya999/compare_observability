"""Traced RAG Pipeline Orchestrator - wraps orchestrator with observability."""

import time
from collections.abc import Generator as GenType
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import config
from src.models import (
    ConceptRelation,
    Document,
    GeneratedResponse,
    QueryAnalysis,
    RetrievedChunk,
)
from src.observability import ObservabilityManager
from src.observability.base import SpanType

from .orchestrator import PipelineResult, RAGOrchestrator


class TracedRAGOrchestrator:
    """
    RAG Orchestrator with full observability tracing.

    Wraps the base orchestrator and adds tracing to all steps.
    Traces are sent to all configured observability providers.
    """

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        persist_directory: Path | None = None,
        observability_providers: list[str] | None = None,
    ):
        self.orchestrator = RAGOrchestrator(openai_client, persist_directory)
        self.obs_manager = ObservabilityManager(observability_providers)
        self.client = self.orchestrator.client

    def ingest_pdf(self, pdf_path: Path) -> tuple[Document, int]:
        """Ingest a PDF with tracing."""
        with self.obs_manager.trace("pdf_ingestion", inputs={"pdf_path": str(pdf_path)}) as trace:
            with self.obs_manager.span("parse_pdf", SpanType.TOOL, trace, inputs={"path": str(pdf_path)}) as parse_span:
                doc, chunks = self.orchestrator.ingestor.ingest_pdf(pdf_path)
                parse_span.provider_spans = {
                    k: self._update_outputs(v, {"doc_id": doc.id, "chunk_count": len(chunks)})
                    for k, v in parse_span.provider_spans.items()
                }

            with self.obs_manager.span("store_embeddings", SpanType.EMBEDDING, trace) as embed_span:
                self.orchestrator.retriever.add_chunks(chunks)
                embed_span.provider_spans = {
                    k: self._update_outputs(v, {"stored_chunks": len(chunks)})
                    for k, v in embed_span.provider_spans.items()
                }

        return doc, len(chunks)

    def query(
        self,
        query: str,
        filter_doc_ids: list[str] | None = None,
        stream: bool = False,
        skip_graph_extraction: bool = False,
    ) -> PipelineResult | GenType[str, None, PipelineResult]:
        """
        Execute the full RAG pipeline with tracing.

        All steps are traced and sent to all active observability providers.
        """
        total_start = time.time()
        step_latencies = {}

        with self.obs_manager.trace("rag_query", inputs={"query": query}) as trace:
            # Step 1: Query Analysis
            with self.obs_manager.span("query_analyzer", SpanType.LLM, trace, inputs={"query": query}) as qa_span:
                step_start = time.time()
                query_analysis = self.orchestrator.query_analyzer.analyze(query)
                step_latencies["query_analysis"] = (time.time() - step_start) * 1000

                # Log the LLM call
                self.obs_manager.log_llm_call(
                    qa_span,
                    model=config.llm_model,
                    messages=[{"role": "user", "content": query}],
                    response=type("Response", (), {
                        "choices": [type("Choice", (), {"message": type("Msg", (), {"content": str(query_analysis)})()})()],
                        "usage": type("Usage", (), {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150})(),
                    })(),
                )

            # Step 2: Retrieval
            with self.obs_manager.span("retriever", SpanType.RETRIEVER, trace, inputs={"query": query_analysis.expanded_query or query}) as ret_span:
                step_start = time.time()
                retrieved_chunks = self.orchestrator.retriever.retrieve(
                    query_analysis, filter_doc_ids=filter_doc_ids
                )
                step_latencies["retrieval"] = (time.time() - step_start) * 1000

                # Log retrieval
                self.obs_manager.log_retrieval(
                    ret_span,
                    query=query_analysis.expanded_query or query,
                    documents=[
                        {
                            "id": c.chunk.id,
                            "text": c.chunk.text[:200],
                            "page": c.chunk.page_number,
                        }
                        for c in retrieved_chunks
                    ],
                    scores=[c.score for c in retrieved_chunks],
                )

            # Step 3: Reranking
            with self.obs_manager.span("reranker", SpanType.LLM, trace, inputs={"chunk_count": len(retrieved_chunks)}) as rerank_span:
                step_start = time.time()
                reranked_chunks = self.orchestrator.reranker.rerank(query_analysis, retrieved_chunks)
                step_latencies["reranking"] = (time.time() - step_start) * 1000

            # Step 4: Generation
            with self.obs_manager.span("generator", SpanType.LLM, trace, inputs={"context_chunks": len(reranked_chunks)}) as gen_span:
                step_start = time.time()

                if stream:
                    return self._stream_response(
                        query,
                        query_analysis,
                        retrieved_chunks,
                        reranked_chunks,
                        step_latencies,
                        total_start,
                        skip_graph_extraction,
                        trace,
                        gen_span,
                    )

                response = self.orchestrator.generator.generate(query_analysis, reranked_chunks)
                step_latencies["generation"] = (time.time() - step_start) * 1000

                # Log the generation
                self.obs_manager.log_llm_call(
                    gen_span,
                    model=config.llm_model,
                    messages=[{"role": "user", "content": query}],
                    response=type("Response", (), {
                        "choices": [type("Choice", (), {"message": type("Msg", (), {"content": response.answer})()})()],
                        "usage": type("Usage", (), {
                            "prompt_tokens": response.token_usage.get("prompt_tokens", 0),
                            "completion_tokens": response.token_usage.get("completion_tokens", 0),
                            "total_tokens": response.token_usage.get("total_tokens", 0),
                        })(),
                    })(),
                )

            # Step 5: Graph Extraction
            concepts = []
            if not skip_graph_extraction:
                with self.obs_manager.span("graph_extractor", SpanType.LLM, trace, inputs={"response_length": len(response.answer)}) as graph_span:
                    step_start = time.time()
                    concepts = self.orchestrator.graph_extractor.extract(response)
                    step_latencies["graph_extraction"] = (time.time() - step_start) * 1000

            total_latency = (time.time() - total_start) * 1000

            # Get trace URLs for all providers
            trace_urls = self.obs_manager.get_trace_urls(trace)

        return PipelineResult(
            query=query,
            query_analysis=query_analysis,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
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
        step_latencies: dict[str, float],
        total_start: float,
        skip_graph_extraction: bool,
        trace: Any,
        gen_span: Any,
    ) -> GenType[str, None, PipelineResult]:
        """Stream response with tracing."""
        step_start = time.time()
        response_gen = self.orchestrator.generator.generate(query_analysis, reranked_chunks, stream=True)

        full_response = ""
        for chunk in response_gen:
            if isinstance(chunk, str):
                full_response += chunk
                yield chunk
            else:
                response = chunk

        step_latencies["generation"] = (time.time() - step_start) * 1000

        # Log the streaming response
        self.obs_manager.log_llm_call(
            gen_span,
            model=config.llm_model,
            messages=[{"role": "user", "content": query}],
            response=type("Response", (), {
                "choices": [type("Choice", (), {"message": type("Msg", (), {"content": response.answer})()})()],
                "usage": type("Usage", (), {
                    "prompt_tokens": response.token_usage.get("prompt_tokens", 0),
                    "completion_tokens": response.token_usage.get("completion_tokens", 0),
                    "total_tokens": response.token_usage.get("total_tokens", 0),
                })(),
            })(),
        )

        # Graph extraction after streaming
        concepts = []
        if not skip_graph_extraction:
            with self.obs_manager.span("graph_extractor", SpanType.LLM, trace) as graph_span:
                step_start = time.time()
                concepts = self.orchestrator.graph_extractor.extract(response)
                step_latencies["graph_extraction"] = (time.time() - step_start) * 1000

        total_latency = (time.time() - total_start) * 1000

        return PipelineResult(
            query=query,
            query_analysis=query_analysis,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            response=response,
            concepts=concepts,
            total_latency_ms=total_latency,
            step_latencies=step_latencies,
        )

    def _update_outputs(self, span: Any, outputs: dict) -> Any:
        """Helper to update span outputs."""
        if hasattr(span, 'outputs'):
            span.outputs = outputs
        return span

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline and observability statistics."""
        return {
            **self.orchestrator.get_stats(),
            "observability": self.obs_manager.get_provider_status(),
        }

    def clear_data(self) -> None:
        """Clear all stored data."""
        self.orchestrator.clear_data()

    def shutdown(self) -> None:
        """Shutdown observability providers."""
        self.obs_manager.shutdown()

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
from src.evaluations.cost import CostTracker
from src.evaluations.ragas_eval import RagasEvaluator


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
        self.cost_tracker = CostTracker()
        self.ragas_evaluator = RagasEvaluator()

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
        run_evals: bool = False,
        retrieval_only: bool = False,
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
                
                # 2.1 Vector Search
                with self.obs_manager.span("vector_search", SpanType.RETRIEVER, trace, inputs={"type": "vector"}) as vec_span:
                    vector_results = self.orchestrator.retriever.retrieve(
                        query_analysis, filter_doc_ids=filter_doc_ids, top_k=(config.top_k * 2)
                    )
                    self.obs_manager.log_retrieval(
                        vec_span,
                        query=query_analysis.expanded_query or query,
                        documents=[{"id": c.chunk.id, "text": c.chunk.text[:50]} for c in vector_results],
                        scores=[c.score for c in vector_results]
                    )

                # 2.2 BM25 Search
                with self.obs_manager.span("bm25_search", SpanType.RETRIEVER, trace, inputs={"type": "bm25"}) as bm25_span:
                    bm25_results = self.orchestrator.retriever.retrieve_bm25(
                        query_analysis, filter_doc_ids=filter_doc_ids, top_k=(config.top_k * 2)
                    )
                    self.obs_manager.log_retrieval(
                        bm25_span,
                        query=query_analysis.expanded_query or query,
                        documents=[{"id": c.chunk.id, "text": c.chunk.text[:50]} for c in bm25_results],
                        scores=[c.score for c in bm25_results]
                    )

                # 2.3 RRF Merge
                with self.obs_manager.span("rrf_merge", SpanType.RETRIEVER, trace) as rrf_span:
                    retrieved_chunks = self.orchestrator.retriever.rrf_merge(vector_results, bm25_results)

                step_latencies["retrieval"] = (time.time() - step_start) * 1000

                # Log final retrieval
                self.obs_manager.log_retrieval(
                    ret_span,
                    query=query_analysis.expanded_query or query,
                    documents=[
                        {
                            "id": c.chunk.id,
                            "text": c.chunk.text[:200],
                            "page": c.chunk.page_number,
                            "source": "vector" if any(v.chunk.id == c.chunk.id for v in vector_results) else "bm25"
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

            # Step 4: Reasoning
            with self.obs_manager.span("reasoning", SpanType.CHAIN, trace, inputs={"context_chunks": len(reranked_chunks)}) as reason_span:
                step_start = time.time()
                reasoning_steps = self.orchestrator.reasoning_engine.run(query_analysis, reranked_chunks)
                
                # Log tool calls if any
                reason_span.outputs = {"tool_calls": len(reasoning_steps), "steps": reasoning_steps}
                step_latencies["reasoning"] = (time.time() - step_start) * 1000

            # Step 5: Generation
            if retrieval_only:
                response = GeneratedResponse(
                    answer="**Retrieval Only Mode**: Generation skipped.",
                    citations=[],
                    concepts=[],
                    token_usage={},
                    latency_ms=0.0,
                    tool_calls=[]
                )
                step_latencies["generation"] = 0.0

                if stream:
                    def _dummy_stream():
                        yield response.answer
                        total_latency = (time.time() - total_start) * 1000
                        yield PipelineResult(
                            query=query,
                            query_analysis=query_analysis,
                            retrieved_chunks=retrieved_chunks,
                            reranked_chunks=reranked_chunks,
                            reasoning_steps=reasoning_steps,
                            response=response,
                            concepts=[],
                            total_latency_ms=total_latency,
                            step_latencies=step_latencies,
                            cost_estimate_usd=0.0,
                            eval_metrics={}
                        )
                    return _dummy_stream()

            else:
                with self.obs_manager.span("generator", SpanType.LLM, trace, inputs={"context_chunks": len(reranked_chunks), "reasoning_steps": len(reasoning_steps)}) as gen_span:
                    step_start = time.time()
    
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
                            trace,
                            gen_span,
                        )
    
                    response = self.orchestrator.generator.generate(
                        query_analysis, reranked_chunks, reasoning_steps=reasoning_steps
                    )
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

            # Step 6: Post-processing (Cost & Evals)
            
            # Cost
            total_tokens = sum(s.get("total_tokens", 0) for s in [response.token_usage])
            # Note: We should ideally sum up tokens from all steps (query analysis, reranking, etc)
            # For now, we mainly have access to generation usage easily.
            # Let's estimate cost based on generation usage primarily + query analysis estimate
            cost = self.cost_tracker.estimate_cost(
                config.llm_model, 
                response.token_usage.get("prompt_tokens", 0), 
                response.token_usage.get("completion_tokens", 0)
            )
            # Add embedding cost estimate (rough)
            # Add reranking cost estimate (rough)
            
            # Evals
            eval_metrics = {}
            if run_evals:
                with self.obs_manager.span("evaluation", SpanType.CHAIN, trace) as eval_span:
                    try:
                        eval_metrics = self.ragas_evaluator.evaluate_response(
                            query=query,
                            answer=response.answer,
                            retrieved_contexts=[c.chunk.text for c in reranked_chunks]
                        )
                        eval_span.outputs = eval_metrics
                    except Exception as e:
                        print(f"Eval failed: {e}")
                        self.obs_manager.log_error(eval_span, e)

            total_latency = (time.time() - total_start) * 1000

            # Get trace URLs for all providers
            trace_urls = self.obs_manager.get_trace_urls(trace)

            result = PipelineResult(
                query=query,
                query_analysis=query_analysis,
                retrieved_chunks=retrieved_chunks,
                reranked_chunks=reranked_chunks,
                reasoning_steps=reasoning_steps,
                response=response,
                concepts=concepts,
                total_latency_ms=total_latency,
                step_latencies=step_latencies,
                cost_estimate_usd=cost,
                eval_metrics=eval_metrics,
            )

            # Update trace output with the final answer
            trace.provider_spans = {
                k: self._update_outputs(v, {"answer": result.response.answer if result.response else ""})
                for k, v in trace.provider_spans.items()
            }

            return result

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
        trace: Any,
        gen_span: Any,
    ) -> GenType[str, None, PipelineResult]:
        """Stream response with tracing."""
        step_start = time.time()
        response_gen = self.orchestrator.generator.generate(
            query_analysis, 
            reranked_chunks, 
            reasoning_steps=reasoning_steps, 
            stream=True
        )

        full_response = ""
        response = None  # Initialize response
        
        for chunk in response_gen:
            if isinstance(chunk, str):
                full_response += chunk
                yield chunk
            else:
                response = chunk

        step_latencies["generation"] = (time.time() - step_start) * 1000

        # If no response object was yielded, create one from accumulated text
        if response is None:
            response = GeneratedResponse(
                answer=full_response,
                citations=[],
                concepts=[],
                token_usage={},
                latency_ms=step_latencies["generation"],
                tool_calls=[]
            )

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

        yield PipelineResult(
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

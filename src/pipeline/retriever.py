"""Vector retrieval module - Step 2 of the RAG pipeline."""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from src.config import config
from src.models import Chunk, QueryAnalysis, RetrievedChunk
from rank_bm25 import BM25Okapi
import re

def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


class Retriever:
    """Handles vector storage and retrieval using ChromaDB."""

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        persist_directory: Path | None = None,
    ):
        self.client = openai_client or OpenAI(api_key=config.openai_api_key)
        self.persist_dir = persist_directory or config.chroma_dir

        # Initialize ChromaDB
        if config.chroma_host:
             self.chroma_client = chromadb.HttpClient(
                 host=config.chroma_host,
                 port=config.chroma_port,
                 settings=Settings(anonymized_telemetry=False)
             )
        else:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize BM25 index
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self._build_bm25_index()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks with embeddings to the vector store."""
        if not chunks:
            return

        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks if c.embedding],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    **c.metadata,
                }
                for c in chunks
            ],
        )
        self._build_bm25_index()

    def retrieve(
        self,
        query_analysis: QueryAnalysis,
        top_k: int | None = None,
        filter_doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query_analysis: Analyzed query with expanded form
            top_k: Number of results to return
            filter_doc_ids: Optional list of document IDs to filter by

        Returns:
            List of RetrievedChunk objects with scores
        """
        k = top_k or config.top_k
        query_text = query_analysis.expanded_query or query_analysis.original_query

        # Generate query embedding
        response = self.client.embeddings.create(
            model=config.embedding_model,
            input=[query_text],
        )
        query_embedding = response.data[0].embedding

        # Build where filter
        where_filter = None
        if filter_doc_ids:
            where_filter = {"document_id": {"$in": filter_doc_ids}}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to RetrievedChunk objects
        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score (cosine distance -> similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # For cosine distance

                chunk = Chunk(
                    id=chunk_id,
                    document_id=results["metadatas"][0][i].get("document_id", ""),
                    text=results["documents"][0][i],
                    page_number=results["metadatas"][0][i].get("page_number", 0),
                    chunk_index=results["metadatas"][0][i].get("chunk_index", 0),
                    metadata=results["metadatas"][0][i],
                )

                retrieved.append(RetrievedChunk(chunk=chunk, score=score, rank=i + 1))

        return retrieved

    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
        }

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all chunks in ChromaDB."""
        # Note: In a production system, we wouldn't fetch all documents at once.
        # This is fine for a demo/benchmark setup.
        result = self.collection.get(include=["documents"])
        if result and result["ids"]:
            self.chunk_ids = result["ids"]
            tokenized_corpus = [tokenize(doc) for doc in result["documents"] if doc]
            # Ensure corpus length matches chunk_ids in case of None documents (unlikely with upsert)
            if len(tokenized_corpus) == len(self.chunk_ids):
                self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                 # Fallback if there's a mismatch or empty docs
                 self.bm25 = None
        else:
            self.bm25 = None
            self.chunk_ids = []

    def retrieve_bm25(
        self,
        query_analysis: QueryAnalysis,
        top_k: int | None = None,
        filter_doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform BM25 search."""
        if not self.bm25:
             return []

        k = top_k or config.top_k
        query_text = query_analysis.expanded_query or query_analysis.original_query
        tokenized_query = tokenize(query_text)
        
        scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k * 2]
        
        bm25_results = []
        
        # Prepare candidate IDs to fetch in batch
        candidate_indices = [i for i, _ in top_n]
        candidate_ids = [self.chunk_ids[i] for i in candidate_indices if i < len(self.chunk_ids)]

        if not candidate_ids:
            return []

        try:
            chunks_data = self.collection.get(
                ids=candidate_ids,
                include=["metadatas", "documents"]
            )
        except Exception:
            # Handle case where IDs might be missing or other DB errors
            return []

        chunk_map = {
            id_: (meta, doc) 
            for id_, meta, doc in zip(chunks_data["ids"], chunks_data["metadatas"], chunks_data["documents"])
        }

        for i, score in top_n:
            if i >= len(self.chunk_ids): continue
            chunk_id = self.chunk_ids[i]
            
            if chunk_id in chunk_map:
                meta, doc_text = chunk_map[chunk_id]
                
                # Check filter
                if filter_doc_ids and meta.get("document_id") not in filter_doc_ids:
                    continue

                chunk = Chunk(
                    id=chunk_id,
                    document_id=meta.get("document_id", ""),
                    text=doc_text,
                    page_number=meta.get("page_number", 0),
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta
                )
                bm25_results.append(RetrievedChunk(chunk=chunk, score=score, rank=0))

        return bm25_results[:k]

    def rrf_merge(
        self,
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        k: int | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """Merge results using Reciprocal Rank Fusion."""
        k = k or config.top_k
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        # Process Vector Results
        for i, res in enumerate(vector_results):
            rrf_scores[res.chunk.id] = rrf_scores.get(res.chunk.id, 0) + (1 / (rrf_k + i + 1))
            chunk_map[res.chunk.id] = res

        # Process BM25 Results
        for i, res in enumerate(bm25_results):
            rrf_scores[res.chunk.id] = rrf_scores.get(res.chunk.id, 0) + (1 / (rrf_k + i + 1))
            if res.chunk.id not in chunk_map:
                chunk_map[res.chunk.id] = res

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        
        final_results = []
        for i, chunk_id in enumerate(sorted_ids):
            res = chunk_map[chunk_id]
            res.score = rrf_scores[chunk_id]
            res.rank = i + 1
            final_results.append(res)
            
        return final_results

    def hybrid_retrieve(
        self,
        query_analysis: QueryAnalysis,
        top_k: int | None = None,
        filter_doc_ids: list[str] | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Retrieve using both vector search and BM25, merging with RRF.
        """
        k = top_k or config.top_k
        
        # 1. Vector Search
        vector_results = self.retrieve(query_analysis, top_k=k * 2, filter_doc_ids=filter_doc_ids)
        
        # 2. BM25 Search
        bm25_results = self.retrieve_bm25(query_analysis, top_k=k * 2, filter_doc_ids=filter_doc_ids)

        # 3. Reciprocal Rank Fusion
        return self.rrf_merge(vector_results, bm25_results, k=k, rrf_k=rrf_k)


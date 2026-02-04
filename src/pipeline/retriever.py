"""Vector retrieval module - Step 2 of the RAG pipeline."""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from src.config import config
from src.models import Chunk, QueryAnalysis, RetrievedChunk


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
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"},
        )

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

    def clear(self) -> None:
        """Clear all data from the collection."""
        self.chroma_client.delete_collection("pdf_chunks")
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"},
        )

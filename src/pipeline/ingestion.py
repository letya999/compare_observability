"""PDF ingestion and chunking module."""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI

from src.config import config
from src.models import Chunk, Document


class PDFIngestor:
    """Handles PDF parsing, chunking, and embedding generation."""

    def __init__(self, openai_client: OpenAI | None = None):
        self.client = openai_client or OpenAI(api_key=config.openai_api_key)
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def ingest_pdf(self, pdf_path: Path) -> tuple[Document, list[Chunk]]:
        """
        Ingest a PDF file: parse, chunk, and generate embeddings.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (Document, list of Chunks with embeddings)
        """
        # Generate document ID from file hash
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        doc_id = f"doc_{file_hash}"

        # Parse PDF
        pdf_doc = fitz.open(pdf_path)
        title = pdf_doc.metadata.get("title") or pdf_path.stem

        document = Document(
            id=doc_id,
            filename=pdf_path.name,
            title=title,
            page_count=len(pdf_doc),
            ingested_at=datetime.now(),
        )

        # Extract text by page
        pages_text = []
        for page_num, page in enumerate(pdf_doc):
            text = page.get_text()
            pages_text.append((page_num + 1, text))

        pdf_doc.close()

        # Chunk the text
        chunks = self._create_chunks(doc_id, pages_text)

        # Generate embeddings
        chunks_with_embeddings = self._generate_embeddings(chunks)

        return document, chunks_with_embeddings

    def _create_chunks(
        self, doc_id: str, pages_text: list[tuple[int, str]]
    ) -> list[Chunk]:
        """Split text into overlapping chunks."""
        chunks = []
        chunk_index = 0

        for page_num, text in pages_text:
            # Simple chunking by character count with overlap
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunk = Chunk(
                        id=f"{doc_id}_chunk_{chunk_index}",
                        document_id=doc_id,
                        text=chunk_text,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        metadata={
                            "char_start": start,
                            "char_end": min(end, len(text)),
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                start = end - self.chunk_overlap
                if start >= len(text):
                    break

        return chunks

    def _generate_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embeddings for all chunks using OpenAI."""
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]

        # Batch embedding generation
        response = self.client.embeddings.create(
            model=config.embedding_model,
            input=texts,
        )

        for i, embedding_data in enumerate(response.data):
            chunks[i].embedding = embedding_data.embedding

        return chunks

    def ingest_directory(self, directory: Path) -> list[tuple[Document, list[Chunk]]]:
        """Ingest all PDFs in a directory."""
        results = []
        for pdf_path in directory.glob("*.pdf"):
            doc, chunks = self.ingest_pdf(pdf_path)
            results.append((doc, chunks))
        return results

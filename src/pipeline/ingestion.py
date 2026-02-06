"""PDF ingestion and chunking module."""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import re
from typing import Any, Dict, List, Tuple
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
        # Generate document ID from file hash
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        doc_id = f"doc_{file_hash}"

        # Open content
        pdf_doc = fitz.open(pdf_path)
        title = pdf_doc.metadata.get("title") or pdf_path.stem

        # 1. Map Chapters
        chapter_map = self._get_chapter_map(pdf_doc)

        # 2. Extract & Clean Content (Blocks)
        content_items = self._extract_clean_content(pdf_doc, chapter_map, title)
        
        page_count = len(pdf_doc)
        pdf_doc.close()

        # 3. Create Smart Chunks
        chunks = self._create_smart_chunks(doc_id, content_items)

        # Generate embeddings
        chunks_with_embeddings = self._generate_embeddings(chunks)

        document = Document(
            id=doc_id,
            filename=pdf_path.name,
            title=title,
            page_count=page_count,
            ingested_at=datetime.now()
        )

        return document, chunks_with_embeddings

    def _get_chapter_map(self, doc: fitz.Document) -> Dict[int, str]:
        """Build a map of page number to chapter title."""
        chapter_map = {}
        try:
            toc = doc.get_toc()
            # toc format: [lvl, title, page_num, ...]
            for i in range(len(toc)):
                lvl, title, page = toc[i]
                if lvl == 1: # Top level chapters
                    start_page = page - 1 # 0-indexed
                    # Determine end page if possible, otherwise map distinct pages
                    # Simple approach: Identify which chapter starts at valid pages
                    chapter_map[start_page] = title
        except Exception:
            pass # Fallback if no TOC
        return chapter_map

    def _extract_clean_content(
        self, doc: fitz.Document, chapter_map: Dict[int, str], doc_title: str
    ) -> List[Dict[str, Any]]:
        """Extract text blocks, filtering headers/footers and noise."""
        items = []
        current_chapter = "Unknown"
        doc_author = doc.metadata.get("author", "")

        sorted_chapters = sorted(chapter_map.keys())
        
        for page_num, page in enumerate(doc):
            # Update chapter
            if page_num in chapter_map:
                current_chapter = chapter_map[page_num]
            elif sorted_chapters:
                # Find the last chapter start <= page_num
                valid_chaps = [p for p in sorted_chapters if p <= page_num]
                if valid_chaps:
                    current_chapter = chapter_map[valid_chaps[-1]]

            # Get structured blocks: (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            page_height = page.rect.height
            
            # Margins for header/footer detection (approx 8% top/bottom)
            top_margin = page_height * 0.08
            bottom_margin = page_height * 0.92

            for b in blocks:
                x0, y0, x1, y1, text, _, _ = b
                text = text.strip()
                if not text:
                    continue

                # --- 1. Geometry Filter (Header/Footer) ---
                is_header_footer = y1 < top_margin or y0 > bottom_margin
                
                # --- 2. Content Heuristics ---
                # Detect page numbers (digits only, or "Page X")
                is_page_num = re.match(r'^(page\s?)?\d+$', text.lower())
                
                # Detect Title/Author repetition (simple header check)
                is_title_noise = (len(text) < 50) and (
                    doc_title.lower() in text.lower() or 
                    (doc_author and doc_author.lower() in text.lower())
                )

                if is_header_footer and (is_page_num or is_title_noise):
                    continue

                items.append({
                    "text": text,
                    "page": page_num + 1,
                    "chapter": current_chapter,
                    "bbox": (x0, y0, x1, y1)
                })
        
        return items

    def _create_smart_chunks(
        self, doc_id: str, items: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """Group blocks into chunks respecting boundaries and size."""
        chunks = []
        chunk_idx = 0
        
        current_buffer = []
        current_len = 0
        
        # Helper to flush buffer
        def flush_buffer(buffer, idx):
            if not buffer: return None
            
            full_text = " ".join([item["text"] for item in buffer]) # Join blocks with space to preserve flow
            
            # Metadata from the first/majority item
            first = buffer[0]
            last = buffer[-1]
            
            meta = {
                "chapter": first["chapter"],
                "start_page": first["page"],
                "end_page": last["page"],
                "bbox": f"{first['bbox'][0]:.1f},{first['bbox'][1]:.1f},{last['bbox'][2]:.1f},{last['bbox'][3]:.1f}"
            }
            
            return Chunk(
                id=f"{doc_id}_chunk_{idx}",
                document_id=doc_id,
                text=full_text,
                page_number=first["page"],
                chunk_index=idx,
                metadata=meta
            )

        i = 0
        while i < len(items):
            item = items[i]
            item_len = len(item["text"])
            
            if current_len + item_len > self.chunk_size:
                # Buffer full
                if current_buffer:
                    chunks.append(flush_buffer(current_buffer, chunk_idx))
                    chunk_idx += 1
                    
                    # --- Overlap Logic ---
                    # Keep last N items that fit within overlap size
                    overlap_acc = 0
                    new_buffer = []
                    for existing in reversed(current_buffer):
                        if overlap_acc + len(existing["text"]) < self.chunk_overlap:
                            new_buffer.insert(0, existing)
                            overlap_acc += len(existing["text"])
                        else:
                            break
                    current_buffer = new_buffer
                    current_len = overlap_acc
                
            current_buffer.append(item)
            current_len += item_len
            i += 1

        # Final flush
        if current_buffer:
            chunks.append(flush_buffer(current_buffer, chunk_idx))

        return chunks

    def _generate_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embeddings for all chunks using OpenAI in batches."""
        if not chunks:
            return chunks

        # OpenAI has a limit of ~8192 tokens per input and 30k tokens per request (varies by model)
        # We'll use a conservative batch size of 20 chunks to be safe
        batch_size = 20
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            texts = [chunk.text for chunk in batch_chunks]
            
            try:
                response = self.client.embeddings.create(
                    model=config.embedding_model,
                    input=texts,
                )
                
                for j, embedding_data in enumerate(response.data):
                    # Validate index to avoid index out of bounds if response doesn't match
                    if j < len(batch_chunks):
                        batch_chunks[j].embedding = embedding_data.embedding
                        
            except Exception as e:
                print(f"Error generating embeddings for batch starting at {i}: {e}")
                raise e

        return chunks

    def ingest_directory(self, directory: Path) -> list[tuple[Document, list[Chunk]]]:
        """Ingest all PDFs in a directory."""
        results = []
        for pdf_path in directory.glob("*.pdf"):
            doc, chunks = self.ingest_pdf(pdf_path)
            results.append((doc, chunks))
        return results

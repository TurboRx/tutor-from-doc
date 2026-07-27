"""
loader.py — PDF/Document Ingestion and Chunking

Provides utilities for reading PDF, text, and markdown files, extracting text content,
and splitting documents into overlapping chunks with precise metadata (source file, page numbers).
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    import pypdf
except ImportError:
    pypdf = None


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_name: str
    page_number: int
    chunk_index: int
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentLoader:
    """Ingests documents (PDF, TXT, MD) and splits them into clean, structured chunks."""

    def __init__(self, default_chunk_size: int = 500, default_chunk_overlap: int = 50):
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text page by page from a PDF file."""
        if pypdf is None:
            raise ImportError(
                "pypdf package is not installed. Install via `pip install pypdf` to read PDF files."
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        pages_data = []
        reader = pypdf.PdfReader(file_path)
        source_name = os.path.basename(file_path)

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages_data.append({
                    "source": source_name,
                    "page_number": page_idx,
                    "text": text
                })

        return pages_data

    def load_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from plain text or markdown file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found at path: {file_path}")

        source_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()

        if not content:
            return []

        return [{
            "source": source_name,
            "page_number": 1,
            "text": content
        }]

    def chunk_text(
        self,
        text: str,
        source_name: str = "document",
        page_number: int = 1,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Split a block of text into sliding window chunks with overlap.
        Preserves paragraph and sentence boundaries where possible.
        """
        size = chunk_size or self.default_chunk_size
        overlap = chunk_overlap or self.default_chunk_overlap

        if not text or not text.strip():
            return []

        # Simple sliding character window with fallback to line/paragraph breaks
        chunks = []
        start = 0
        text_length = len(text)
        chunk_idx = 0

        while start < text_length:
            end = start + size

            # If we are not at the end of the text, try to break cleanly on a space/newline
            if end < text_length:
                # Look for last space or newline before limit
                last_space = text.rfind(' ', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_space, last_newline)
                
                # Use breakpoint if it gives a chunk of reasonable size (> 50% target size)
                if break_point > start + (size // 2):
                    end = break_point + 1

            chunk_content = text[start:end].strip()

            if chunk_content:
                chunk_id = f"{source_name}_p{page_number}_c{chunk_idx}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    source_name=source_name,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    char_count=len(chunk_content)
                ))
                chunk_idx += 1

            # Advance start pointer by (end - overlap)
            start_next = end - overlap
            if start_next <= start:  # Prevent infinite loop
                start_next = end
            start = start_next

        return chunks

    def load_and_chunk(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Automatically detects file format, extracts text, and returns document chunks."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            pages = self.load_pdf(file_path)
        else:
            pages = self.load_text(file_path)

        all_chunks = []
        for page in pages:
            page_chunks = self.chunk_text(
                text=page["text"],
                source_name=page["source"],
                page_number=page["page_number"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            all_chunks.extend(page_chunks)

        return all_chunks

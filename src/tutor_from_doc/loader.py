"""
loader.py — PDF/Document Ingestion and Chunking

Provides utilities for reading PDF, text, and markdown files, extracting text content,
and splitting documents into overlapping chunks with precise metadata (source file, page numbers).
"""

import os
import re
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
        self.default_chunk_size = max(50, default_chunk_size)
        self.default_chunk_overlap = max(0, min(default_chunk_overlap, self.default_chunk_size - 10))

    def clean_text(self, text: str) -> str:
        """Clean control characters and normalize excessive whitespace."""
        if not text:
            return ""
        # Strip non-printable control characters (except newline/tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize multiple blank spaces while preserving line breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text page by page from a PDF file with per-page exception safety."""
        if pypdf is None:
            raise ImportError(
                "pypdf package is not installed. Install via `pip install pypdf` to read PDF files."
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        pages_data = []
        try:
            reader = pypdf.PdfReader(file_path)
            source_name = os.path.basename(file_path)

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    cleaned = self.clean_text(text)
                    if cleaned:
                        pages_data.append({
                            "source": source_name,
                            "page_number": page_idx,
                            "text": cleaned
                        })
                except Exception:
                    # Skip corrupt page gracefully
                    continue
        except Exception as e:
            raise ValueError(f"Failed to read PDF file '{file_path}': {str(e)}")

        return pages_data

    def load_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from plain text or markdown file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found at path: {file_path}")

        source_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        cleaned = self.clean_text(content)
        if not cleaned:
            return []

        return [{
            "source": source_name,
            "page_number": 1,
            "text": cleaned
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
        size = max(50, chunk_size or self.default_chunk_size)
        overlap = max(0, min(chunk_overlap if chunk_overlap is not None else self.default_chunk_overlap, size - 10))

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        chunks = []
        start = 0
        text_length = len(cleaned_text)
        chunk_idx = 0

        while start < text_length:
            end = start + size

            if end < text_length:
                last_space = cleaned_text.rfind(' ', start, end)
                last_newline = cleaned_text.rfind('\n', start, end)
                break_point = max(last_space, last_newline)
                
                if break_point > start + (size // 2):
                    end = break_point + 1

            chunk_content = cleaned_text[start:end].strip()

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

            start_next = end - overlap
            if start_next <= start:
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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

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

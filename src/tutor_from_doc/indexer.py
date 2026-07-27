"""
indexer.py — Embedding Generation and Vector Search Storage

Provides embedding generation and local vector search.
Uses sentence-transformers if available, with a fast zero-dependency TF-IDF/Cosine vector store
as fallback so the library works out-of-the-box in low-resource environments without forcing heavy ML installs.
"""

import os
import math
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from tutor_from_doc.loader import DocumentChunk

# Try optional imports
try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None


class BaseEmbedder:
    """Interface for text embedding generation."""
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class FastTFIDFEmbedder(BaseEmbedder):
    """
    Lightweight zero-dependency TF-IDF / N-gram term frequency embedder.
    Requires no GPU, zero neural network downloads, and runs instantly on any device.
    """
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if len(w) > 2]

    def fit_and_embed(self, texts: List[str]) -> List[List[float]]:
        # Build vocabulary from corpus
        word_counts: Dict[str, int] = {}
        for text in texts:
            tokens = set(self._tokenize(text))
            for t in tokens:
                word_counts[t] = word_counts.get(t, 0) + 1

        # Sort top vocabulary
        sorted_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size]
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}

        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        vocab_len = len(self.vocab)
        if vocab_len == 0:
            return [[0.0] for _ in texts]

        for text in texts:
            vec = [0.0] * vocab_len
            tokens = self._tokenize(text)
            for t in tokens:
                if t in self.vocab:
                    vec[self.vocab[t]] += 1.0

            # L2 normalization
            sq_sum = sum(v * v for v in vec)
            if sq_sum > 0:
                norm = math.sqrt(sq_sum)
                vec = [v / norm for v in vec]

            embeddings.append(vec)
        return embeddings


class SentenceTransformerEmbedder(BaseEmbedder):
    """Neural embedding generator using HuggingFace sentence-transformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Install via `pip install sentence-transformers`."
            )
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()


class VectorIndexer:
    """
    Local Vector Store for indexing document chunks and running top-K similarity search.
    Supports both neural embeddings and lightweight TF-IDF fallback vectors.
    """
    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self.embedder = embedder or FastTFIDFEmbedder()
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Add and index document chunks."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]

        # If using FastTFIDFEmbedder, fit vocabulary first
        if isinstance(self.embedder, FastTFIDFEmbedder):
            new_embeddings = self.embedder.fit_and_embed(texts)
        else:
            new_embeddings = self.embedder.embed_texts(texts)

        self.chunks.extend(chunks)
        self.embeddings.extend(new_embeddings)

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        min_len = min(len(vec_a), len(vec_b))
        if min_len == 0:
            return 0.0

        dot = sum(vec_a[i] * vec_b[i] for i in range(min_len))
        norm_a = math.sqrt(sum(vec_a[i] * vec_a[i] for i in range(min_len)))
        norm_b = math.sqrt(sum(vec_b[i] * vec_b[i] for i in range(min_len)))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for top-K most relevant chunks for a given text query.
        Returns list of (DocumentChunk, similarity_score) sorted by score descending.
        """
        if not self.chunks or not self.embeddings:
            return []

        query_embeddings = self.embedder.embed_texts([query])
        query_vec = query_embeddings[0]

        scored_results = []
        for chunk, emb in zip(self.chunks, self.embeddings):
            score = self._cosine_similarity(query_vec, emb)
            scored_results.append((chunk, score))

        # Sort by similarity score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def save(self, file_path: str):
        """Save vector store state to a local JSON file."""
        data = {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "embeddings": self.embeddings,
            "vocab": getattr(self.embedder, "vocab", {})
        }
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, file_path: str):
        """Load vector store state from a local JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vector store file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [DocumentChunk(**item) for item in data["chunks"]]
        self.embeddings = data["embeddings"]
        if isinstance(self.embedder, FastTFIDFEmbedder) and "vocab" in data:
            self.embedder.vocab = data["vocab"]

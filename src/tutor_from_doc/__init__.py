"""
tutor-from-doc package initialization.
"""

from tutor_from_doc.loader import DocumentLoader, DocumentChunk
from tutor_from_doc.indexer import VectorIndexer, FastTFIDFEmbedder, SentenceTransformerEmbedder
from tutor_from_doc.tutor import TutorEngine, BaseLLM, GroundedLocalLLM, OpenAILLM, OllamaLLM

__version__ = "0.1.0"
__all__ = [
    "DocumentLoader",
    "DocumentChunk",
    "VectorIndexer",
    "FastTFIDFEmbedder",
    "SentenceTransformerEmbedder",
    "TutorEngine",
    "BaseLLM",
    "GroundedLocalLLM",
    "OpenAILLM",
    "OllamaLLM"
]

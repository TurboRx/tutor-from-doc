import pytest
from tutor_from_doc.loader import DocumentChunk
from tutor_from_doc.indexer import VectorIndexer, FastTFIDFEmbedder


def test_vector_indexer_search():
    indexer = VectorIndexer(embedder=FastTFIDFEmbedder())
    
    chunk1 = DocumentChunk(
        chunk_id="c1",
        text="Photosynthesis converts solar energy into sugar glucose in chloroplasts.",
        source_name="bio.txt",
        page_number=1,
        chunk_index=0,
        char_count=70
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        text="Newton's laws of motion explain inertia, force, acceleration, and action-reaction.",
        source_name="physics.txt",
        page_number=2,
        chunk_index=0,
        char_count=80
    )

    indexer.index_chunks([chunk1, chunk2])
    
    results = indexer.search("Tell me about photosynthesis in chloroplasts", top_k=1)
    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "c1"
    assert score > 0.0

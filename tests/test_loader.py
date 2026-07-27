import os
import tempfile
import pytest
from tutor_from_doc.loader import DocumentLoader, DocumentChunk


def test_chunk_text_basic():
    loader = DocumentLoader(default_chunk_size=100, default_chunk_overlap=20)
    sample_text = (
        "Photosynthesis is the process used by plants, algae and certain bacteria to harness "
        "energy from sunlight and turn it into chemical energy. Sunlight powers the reaction."
    )
    chunks = loader.chunk_text(sample_text, source_name="biology_notes.txt", page_number=1)

    assert len(chunks) > 0
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].source_name == "biology_notes.txt"
    assert chunks[0].page_number == 1
    assert "Photosynthesis" in chunks[0].text


def test_load_text_file():
    loader = DocumentLoader()
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as temp:
        temp.write("Cellular respiration produces ATP energy for cells to function efficiently.")
        temp_path = temp.name

    try:
        chunks = loader.load_and_chunk(temp_path)
        assert len(chunks) == 1
        assert "Cellular respiration" in chunks[0].text
    finally:
        os.remove(temp_path)

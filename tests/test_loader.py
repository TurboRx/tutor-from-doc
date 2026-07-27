import os
import tempfile
import unittest
from tutor_from_doc.loader import DocumentLoader, DocumentChunk


class TestDocumentLoader(unittest.TestCase):
    def test_chunk_text_basic(self):
        loader = DocumentLoader(default_chunk_size=100, default_chunk_overlap=20)
        sample_text = (
            "Photosynthesis is the process used by plants, algae and certain bacteria to harness "
            "energy from sunlight and turn it into chemical energy. Sunlight powers the reaction."
        )
        chunks = loader.chunk_text(sample_text, source_name="biology_notes.txt", page_number=1)

        self.assertTrue(len(chunks) > 0)
        self.assertIsInstance(chunks[0], DocumentChunk)
        self.assertEqual(chunks[0].source_name, "biology_notes.txt")
        self.assertEqual(chunks[0].page_number, 1)
        self.assertIn("Photosynthesis", chunks[0].text)

    def test_load_text_file(self):
        loader = DocumentLoader()
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as temp:
            temp.write("Cellular respiration produces ATP energy for cells to function efficiently.")
            temp_path = temp.name

        try:
            chunks = loader.load_and_chunk(temp_path)
            self.assertEqual(len(chunks), 1)
            self.assertIn("Cellular respiration", chunks[0].text)
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()

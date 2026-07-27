import os
import tempfile
import unittest
from tutor_from_doc.tutor import TutorEngine, GroundedLocalLLM


class TestTutorEngine(unittest.TestCase):
    def test_grounded_answering_and_refusal(self):
        engine = TutorEngine(llm=GroundedLocalLLM())
        
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as temp:
            temp.write(
                "The mitochondria is known as the powerhouse of the cell. "
                "It generates most of the chemical energy needed to power the cell's biochemical reactions."
            )
            temp_path = temp.name

        try:
            engine.ingest_document(temp_path)

            # Question present in context
            res1 = engine.ask("What is the powerhouse of the cell?")
            self.assertTrue(res1["grounded"])
            self.assertIn("mitochondria", res1["answer"].lower())
            self.assertGreater(len(res1["citations"]), 0)

            # Question absent from context (testing strict non-hallucination)
            res2 = engine.ask("Who was the 16th president of the United States?")
            self.assertIn("cannot find this information", res2["answer"].lower())
            self.assertEqual(len(res2["citations"]), 0)
        finally:
            os.remove(temp_path)

    def test_quiz_generation(self):
        engine = TutorEngine()
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as temp:
            temp.write(
                "Water boils at 100 degrees Celsius under standard atmospheric pressure. "
                "Freezing point of pure water is 0 degrees Celsius."
            )
            temp_path = temp.name

        try:
            engine.ingest_document(temp_path)
            quiz_res = engine.quiz(num_questions=2)
            self.assertIn("questions", quiz_res)
            self.assertGreater(len(quiz_res["questions"]), 0)
            self.assertIn("page_reference", quiz_res["questions"][0])
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import pytest
from tutor_from_doc.tutor import TutorEngine, GroundedLocalLLM


def test_grounded_answering_and_refusal():
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
        assert res1["grounded"] is True
        assert "mitochondria" in res1["answer"].lower()
        assert len(res1["citations"]) > 0

        # Question absent from context (testing strict non-hallucination)
        res2 = engine.ask("Who was the 16th president of the United States?")
        assert "cannot find this information" in res2["answer"].lower() or len(res2["citations"]) == 0
    finally:
        os.remove(temp_path)


def test_quiz_generation():
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
        assert "questions" in quiz_res
        assert len(quiz_res["questions"]) > 0
        assert "page_reference" in quiz_res["questions"][0]
    finally:
        os.remove(temp_path)

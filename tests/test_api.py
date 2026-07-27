import unittest
import io
from fastapi.testclient import TestClient
from tutor_from_doc.api import app


class TestTutorAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("active_sessions", data)

    def test_upload_and_ask_flow(self):
        file_content = (
            "Photosynthesis happens inside chloroplasts. Chlorophyll pigment absorbs sunlight "
            "to convert water and carbon dioxide into oxygen and glucose."
        )
        file_obj = io.BytesIO(file_content.encode("utf-8"))

        # Upload document
        upload_res = self.client.post(
            "/upload",
            files={"file": ("biology.txt", file_obj, "text/plain")},
            data={"doc_id": "test_bio"}
        )
        self.assertEqual(upload_res.status_code, 200)
        self.assertEqual(upload_res.json()["status"], "success")

        # Ask question
        ask_res = self.client.post(
            "/ask",
            json={"doc_id": "test_bio", "question": "Where does photosynthesis happen?", "top_k": 2}
        )
        self.assertEqual(ask_res.status_code, 200)
        self.assertIn("chloroplasts", ask_res.json()["answer"].lower())

        # Generate quiz
        quiz_res = self.client.post(
            "/quiz",
            json={"doc_id": "test_bio", "num_questions": 1}
        )
        self.assertEqual(quiz_res.status_code, 200)
        self.assertGreater(len(quiz_res.json()["questions"]), 0)

    def test_ask_missing_doc_id_404(self):
        res = self.client.post("/ask", json={"doc_id": "nonexistent_id", "question": "Test?"})
        self.assertEqual(res.status_code, 404)

    def test_ask_empty_question_400(self):
        res = self.client.post("/ask", json={"doc_id": "default", "question": ""})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()

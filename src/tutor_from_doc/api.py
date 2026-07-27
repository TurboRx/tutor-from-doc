"""
api.py — FastAPI REST Service for Tutor-From-Doc

Provides HTTP endpoints:
- POST /upload: Upload & ingest PDF or text document
- POST /ask: Ask grounded question & receive answer + citations
- POST /quiz: Generate grounded practice quiz questions
"""

import os
import shutil
import tempfile
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field

from tutor_from_doc.tutor import TutorEngine

app = FastAPI(
    title="TutorFromDoc REST API",
    description="Grounded AI Tutor RAG service for low-resource education access",
    version="0.1.0"
)

# Global engine registry mapping session/document IDs to TutorEngine instances
engines: Dict[str, TutorEngine] = {}
DEFAULT_SESSION = "default"


class AskRequest(BaseModel):
    doc_id: Optional[str] = Field(default="default", description="Document ID returned from /upload")
    question: str = Field(..., description="Student question grounded in document content")
    top_k: Optional[int] = Field(default=3, description="Number of context chunks to retrieve")


class QuizRequest(BaseModel):
    doc_id: Optional[str] = Field(default="default", description="Document ID returned from /upload")
    num_questions: Optional[int] = Field(default=3, description="Number of practice quiz questions to generate")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(engines),
        "version": "0.1.0"
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(default="default")
):
    """
    Ingest a PDF, TXT, or MD document into the vector search index.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail="Supported file formats are .pdf, .txt, .md")

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize engine for this doc_id
        engine = TutorEngine()
        result = engine.ingest_document(temp_path)
        engines[doc_id] = engine

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": filename,
            "total_chunks": result["total_chunks"],
            "message": f"Successfully indexed '{filename}'. You can now ask questions via /ask or generate a quiz via /quiz."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Ask a question grounded ONLY in the uploaded document.
    """
    engine = engines.get(request.doc_id)
    if not engine:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for doc_id '{request.doc_id}'. Please call POST /upload first."
        )

    response = engine.ask(question=request.question, top_k=request.top_k)
    return response


@app.post("/quiz")
def generate_quiz(request: QuizRequest):
    """
    Generate grounded practice quiz questions from the uploaded document.
    """
    engine = engines.get(request.doc_id)
    if not engine:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for doc_id '{request.doc_id}'. Please call POST /upload first."
        )

    response = engine.quiz(num_questions=request.num_questions)
    return response

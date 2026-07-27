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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tutor_from_doc.tutor import TutorEngine

app = FastAPI(
    title="TutorFromDoc REST API",
    description="Grounded AI Tutor RAG service for low-resource education access",
    version="0.1.0"
)

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    session_id = (doc_id or DEFAULT_SESSION).strip()

    # Create temporary directory and save uploaded file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    try:
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        engine = TutorEngine()
        result = engine.ingest_document(temp_path)
        engines[session_id] = engine

        return {
            "status": "success",
            "doc_id": session_id,
            "filename": filename,
            "total_chunks": result["total_chunks"],
            "message": f"Successfully indexed '{filename}'. You can now ask questions via /ask or generate a quiz via /quiz."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Ask a question grounded ONLY in the uploaded document.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    doc_id = (request.doc_id or DEFAULT_SESSION).strip()
    engine = engines.get(doc_id)

    if not engine:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for doc_id '{doc_id}'. Please call POST /upload first."
        )

    response = engine.ask(question=request.question.strip(), top_k=request.top_k or 3)
    return response


@app.post("/quiz")
def generate_quiz(request: QuizRequest):
    """
    Generate grounded practice quiz questions from the uploaded document.
    """
    doc_id = (request.doc_id or DEFAULT_SESSION).strip()
    engine = engines.get(doc_id)

    if not engine:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for doc_id '{request.doc_id}'. Please call POST /upload first."
        )

    response = engine.quiz(num_questions=request.num_questions or 3)
    return response

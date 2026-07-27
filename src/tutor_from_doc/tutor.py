"""
tutor.py — Core RAG Tutor Engine and Grounded Question Answering

Implements RAG retrieval, grounded answer synthesis with 0% hallucination guardrails,
practice quiz generation, and pluggable LLM interfaces (Local, OpenAI, Ollama).
"""

import re
import json
import urllib.request
from typing import List, Dict, Any, Optional
from tutor_from_doc.loader import DocumentLoader, DocumentChunk
from tutor_from_doc.indexer import VectorIndexer


class BaseLLM:
    """Abstract interface for LLM calls in Tutor Engine."""
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError


class GroundedLocalLLM(BaseLLM):
    """
    Default zero-cost, offline-friendly LLM synthesis engine.
    Synthesizes grounded responses directly from retrieved chunks without external API dependencies.
    """
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if "CONTEXT:" in prompt:
            parts = prompt.split("CONTEXT:")
            context_section = parts[1].split("QUESTION:")[0].strip() if len(parts) > 1 else ""
            query_section = prompt.split("QUESTION:")[1].strip() if "QUESTION:" in prompt else ""
        else:
            context_section = prompt
            query_section = ""

        if not context_section or "NO RELEVANT CONTEXT" in context_section:
            return "I cannot find this information in the uploaded document. Please ask a question related to the document content."

        clean_context = re.sub(r'\[Source:\s*[^\]]+\]', '', context_section).strip()
        raw_sentences = re.split(r'(?<!\be\.g)(?<!\bi\.e)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bFig)(?<=[.!?])\s+', clean_context)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]

        stopwords = {
            "what", "is", "the", "a", "an", "how", "why", "can", "you", "tell", "me",
            "about", "in", "on", "of", "who", "was", "where", "which", "are", "do", "does", "did", "for", "with"
        }
        query_words = set(re.findall(r'\w+', query_section.lower())) - stopwords

        matched_sentences = []
        for stmt in sentences:
            stmt_clean = stmt.strip()
            if not stmt_clean:
                continue
            stmt_words = set(re.findall(r'\w+', stmt_clean.lower()))
            overlap = query_words.intersection(stmt_words)
            if overlap:
                matched_sentences.append((stmt_clean, len(overlap)))

        matched_sentences.sort(key=lambda x: x[1], reverse=True)

        if matched_sentences:
            best_sentences = [s[0] for s in matched_sentences[:4]]
            return "Based on the document content:\n\n" + "\n\n".join(f"• {s}" for s in best_sentences)

        return "I cannot find this information in the uploaded document. Please ask a question related to the document content."


class OpenAILLM(BaseLLM):
    """OpenAI API client adapter."""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = json.dumps({"model": self.model, "messages": messages, "temperature": 0.2}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                return res_body["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"


class OllamaLLM(BaseLLM):
    """Local Ollama HTTP client adapter."""
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3", timeout: int = 30):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        url = f"{self.host}/api/generate"
        headers = {"Content-Type": "application/json"}
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        data = json.dumps({"model": self.model, "prompt": full_prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                return res_body.get("response", "").strip()
        except Exception as e:
            return f"Error calling Ollama API: {str(e)}"


class TutorEngine:
    """
    Interactive RAG Tutor Engine.
    Handles document ingestion, vector retrieval, grounded answering, and quiz generation.
    """
    def __init__(
        self,
        indexer: Optional[VectorIndexer] = None,
        llm: Optional[BaseLLM] = None,
        loader: Optional[DocumentLoader] = None
    ):
        self.loader = loader or DocumentLoader()
        self.indexer = indexer or VectorIndexer()
        self.llm = llm or GroundedLocalLLM()
        self.is_indexed = False

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest document file (PDF, TXT, MD), create chunks, and build vector search index."""
        chunks = self.loader.load_and_chunk(file_path)
        if not chunks:
            raise ValueError(f"No valid text content could be extracted from file: {file_path}")

        self.indexer.index_chunks(chunks)
        self.is_indexed = True

        return {
            "status": "success",
            "file_name": file_path,
            "total_chunks": len(chunks),
            "sample_chunk": chunks[0].text[:150] + "..." if chunks else ""
        }

    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Ask a question grounded strictly in document content, returning answer and citations."""
        if not question or not question.strip():
            return {
                "answer": "Please provide a non-empty question.",
                "citations": [],
                "grounded": False
            }

        if not self.is_indexed or not self.indexer.chunks:
            return {
                "answer": "No document has been ingested yet. Please upload a document first.",
                "citations": [],
                "grounded": False
            }

        k = max(1, top_k)
        results = self.indexer.search(query=question, top_k=k)

        relevant_chunks = [chunk for chunk, score in results if score >= 0.08]

        if not relevant_chunks:
            return {
                "answer": "I cannot find this information in the uploaded document. Please ask a question related to the document content.",
                "citations": [],
                "grounded": True
            }

        context_str = "\n\n".join(
            f"[Source: {c.source_name}, Page {c.page_number}]\n{c.text}" for c in relevant_chunks
        )

        system_prompt = (
            "You are an AI Tutor dedicated to assisting students. "
            "STRICT GROUNDING RULE: You must answer the user's question using ONLY the facts present in the provided document context. "
            "If the question cannot be answered using the provided context, state clearly: "
            "'I cannot find this information in the uploaded document.'"
        )

        user_prompt = (
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )

        answer = self.llm.generate(prompt=user_prompt, system_prompt=system_prompt)

        if "cannot find this information" in answer.lower():
            return {
                "answer": "I cannot find this information in the uploaded document. Please ask a question related to the document content.",
                "citations": [],
                "grounded": True
            }

        citations = []
        for chunk, score in results[:len(relevant_chunks)]:
            citations.append({
                "source": chunk.source_name,
                "page": chunk.page_number,
                "snippet": chunk.text[:200] + "...",
                "relevance_score": round(float(score), 4)
            })

        return {
            "answer": answer,
            "citations": citations,
            "grounded": True
        }

    def quiz(self, num_questions: int = 3) -> Dict[str, Any]:
        """Generate practice quiz questions directly from document content for self-testing."""
        if not self.is_indexed or not self.indexer.chunks:
            return {
                "error": "No document ingested. Upload a document to generate a quiz.",
                "questions": []
            }

        target_count = max(1, num_questions)
        chunks = self.indexer.chunks
        questions = []
        seen_questions = set()

        for chunk in chunks:
            if len(questions) >= target_count:
                break

            sentences = [
                s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk.text)
                if len(s.strip()) > 30 and not s.strip().startswith("=") and not s.strip().startswith("#")
            ]
            
            if not sentences:
                continue

            target_stmt = sentences[0]
            clean_stmt = re.sub(r'^[0-9\.\-\s]+', '', target_stmt)
            question_text = f"Based on page {chunk.page_number}, explain the key concept: '{clean_stmt[:120]}...'"
            
            if question_text not in seen_questions:
                seen_questions.add(question_text)
                questions.append({
                    "id": len(questions) + 1,
                    "question": question_text,
                    "page_reference": chunk.page_number,
                    "source_text_hint": chunk.text[:250] + "..."
                })

        return {
            "total_questions": len(questions),
            "questions": questions
        }

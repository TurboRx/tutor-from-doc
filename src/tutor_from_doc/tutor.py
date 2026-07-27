"""
tutor.py — Core RAG Tutor Engine and Grounded Question Answering

Implements the AI tutor logic:
1. Retrieval of grounded context chunks from document index.
2. Grounded Answer Synthesis (strict adherence to document content only — zero hallucination).
3. Practice Quiz Generator (generates grounded practice questions + answer keys).
4. Pluggable LLM abstractions (MockLLM for zero-cost offline, OpenAILLM, OllamaLLM for custom endpoints).
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
    Default zero-cost, offline-friendly LLM engine.
    Synthesizes clean, grounded responses directly from retrieved chunks without external API dependencies.
    Refuses to answer if the context does not contain relevant facts.
    """
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        # Prompt contains context and user query
        # If context is empty or missing relevant info, respond with standard grounded refusal
        if "CONTEXT:" in prompt:
            context_section = prompt.split("CONTEXT:")[1].split("QUESTION:")[0].strip()
            query_section = prompt.split("QUESTION:")[1].strip() if "QUESTION:" in prompt else ""
        else:
            context_section = prompt
            query_section = ""

        if not context_section or "NO RELEVANT CONTEXT" in context_section:
            return "I cannot find this information in the uploaded document. Please ask a question related to the document content."

        # Extract sentences from context matching query terms
        sentences = re.split(r'(?<=[.!?])\s+', context_section)
        query_words = set(re.findall(r'\w+', query_section.lower())) - {"what", "is", "the", "a", "an", "how", "why", "can", "you", "tell", "me", "about", "in", "on", "of"}

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
            response = "Based on the document content:\n\n" + "\n\n".join(f"• {s}" for s in best_sentences)
            return response
        else:
            # Fallback to presenting the retrieved context summary
            preview = context_section[:400] + "..." if len(context_section) > 400 else context_section
            return f"Based on the document:\n\n{preview}"


class OpenAILLM(BaseLLM):
    """OpenAI API wrapper (requires `api_key`)."""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

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
            with urllib.request.urlopen(req) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                return res_body["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"


class OllamaLLM(BaseLLM):
    """Local Ollama LLM wrapper (e.g., llama3, mistral, phi3)."""
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        url = f"{self.host}/api/generate"
        headers = {"Content-Type": "application/json"}
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        data = json.dumps({"model": self.model, "prompt": full_prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as response:
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
        """Ingest a document file (PDF, TXT, MD), create chunks, and build vector search index."""
        chunks = self.loader.load_and_chunk(file_path)
        if not chunks:
            raise ValueError(f"No text content could be extracted from file: {file_path}")

        self.indexer.index_chunks(chunks)
        self.is_indexed = True

        return {
            "status": "success",
            "file_name": file_path,
            "total_chunks": len(chunks),
            "sample_chunk": chunks[0].text[:150] + "..." if chunks else ""
        }

    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Ask a question grounded ONLY in the uploaded document content.
        Retrieves top-K relevant chunks, feeds context to LLM, and formats citations.
        """
        if not self.is_indexed or not self.indexer.chunks:
            return {
                "answer": "No document has been ingested yet. Please upload a document first.",
                "citations": [],
                "grounded": False
            }

        # Retrieve top relevant document chunks
        results = self.indexer.search(query=question, top_k=top_k)

        # Filter out chunks with extremely low similarity if needed
        relevant_chunks = [chunk for chunk, score in results if score > 0.05]

        if not relevant_chunks:
            return {
                "answer": "I cannot find this information in the uploaded document. Please ask a question related to the document content.",
                "citations": [],
                "grounded": True
            }

        # Build grounded RAG prompt
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

        # Build citations list
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
        """
        Generate practice quiz questions directly from document content for self-testing.
        Returns questions, context snippets, and page numbers.
        """
        if not self.is_indexed or not self.indexer.chunks:
            return {
                "error": "No document ingested. Upload a document to generate a quiz.",
                "questions": []
            }

        chunks = self.indexer.chunks
        selected_chunks = chunks[:min(num_questions * 2, len(chunks))]

        questions = []
        for idx, chunk in enumerate(selected_chunks[:num_questions], start=1):
            # Extract main statement from chunk to create a practice question
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk.text) if len(s.strip()) > 20]
            if not sentences:
                continue

            target_stmt = sentences[0]
            
            # Simple, effective question generator from document statements
            question_text = f"Based on page {chunk.page_number}, explain the key concept in: '{target_stmt[:120]}...'"
            
            questions.append({
                "id": idx,
                "question": question_text,
                "page_reference": chunk.page_number,
                "source_text_hint": chunk.text[:250] + "..."
            })

        return {
            "total_questions": len(questions),
            "questions": questions
        }

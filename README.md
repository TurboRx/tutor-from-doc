# 🎓 TutorFromDoc (`tutor-from-doc`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Education Access](https://img.shields.io/badge/Mission-Education%20Access-brightgreen.svg)](#-mission--education-access-motivation)

> **Turn any PDF or document into an interactive, grounded AI tutor for students — with 0% hallucination and zero mandatory paid API dependencies.**

---

## 🌟 Mission & Education-Access Motivation

Quality tutoring and personalized learning remain out of reach for millions of students worldwide due to paywalls, expensive AI subscriptions, and unreliable internet infrastructure. Furthermore, generic LLMs often **hallucinate** facts when asked about complex textbook content, leading to student confusion.

**TutorFromDoc** is an open-source Python library and REST API built to democratize education access. 

### Core Principles:
1. **100% Grounded in Context**: Answers are generated **ONLY** using facts present in the uploaded document. If a topic is not in the text, the tutor explicitly declines to answer rather than making up information.
2. **Zero Paid API Barrier**: Runs out-of-the-box using local, lightweight TF-IDF or open-source embeddings and local zero-cost models.
3. **Pluggable Architecture**: Easily attach custom LLM backends (OpenAI, Ollama, HuggingFace) when available.
4. **Interactive Learning**: Includes an automated **Quiz Generator** (`POST /quiz`) that extracts practice questions directly from document pages for student revision.

---

## 🏗️ Architecture Overview

```
 ┌──────────────────────┐      ┌───────────────────────┐
 │ Student Document     │ ───► │  loader.py            │
 │ (PDF, TXT, Markdown) │      │  (Ingestion & Chunk)  │
 └──────────────────────┘      └───────────┬───────────┘
                                           │
                                           ▼
 ┌──────────────────────┐      ┌───────────────────────┐
 │ Vector Search Index  │ ◄─── │  indexer.py           │
 │ (FAISS / Local Store)│      │  (Embeddings Index)   │
 └──────────┬───────────┘      └───────────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────────────────┐
 │  tutor.py & api.py (RAG Tutor & Practice Quiz Engine│
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────┐
 │  Grounded Answer + Page Citations + Practice Quiz   │
 └─────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install:

```bash
git clone https://github.com/TurboRx/tutor-from-doc.git
cd tutor-from-doc
pip install -e .
```

*Optional ML extensions for neural embeddings:*
```bash
pip install -e ".[ml]"
```

### 2. Python API Usage

```python
from tutor_from_doc import TutorEngine

# 1. Initialize the engine
tutor = TutorEngine()

# 2. Ingest any PDF or study document
tutor.ingest_document("biology_chapter4.pdf")

# 3. Ask grounded questions
response = tutor.ask("What is the role of RuBisCO in the Calvin Cycle?")

print("Tutor Answer:\n", response["answer"])
print("Citations:\n", response["citations"])

# 4. Generate practice quiz questions for self-testing
quiz = tutor.quiz(num_questions=3)
for q in quiz["questions"]:
    print(f"Quiz Q{q['id']}: {q['question']}")
```

---

## 🌐 FastAPI REST API

Start the REST API server:

```bash
uvicorn tutor_from_doc.api:app --reload --port 8000
```

### Endpoints:

#### 1. Ingest Document (`POST /upload`)
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample_biology_guide.txt" \
  -F "doc_id=biology_101"
```

#### 2. Ask Grounded Question (`POST /ask`)
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "biology_101",
    "question": "What process splits water molecules during photosynthesis?",
    "top_k": 3
  }'
```

#### 3. Generate Practice Quiz (`POST /quiz`)
```bash
curl -X POST "http://localhost:8000/quiz" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "biology_101",
    "num_questions": 3
  }'
```

---

## 🔌 Bringing Your Own LLM (Pluggable Backends)

By default, `TutorFromDoc` uses an offline, zero-dependency `GroundedLocalLLM`. You can easily plug in OpenAI or local Ollama instances:

### Using OpenAI GPT:
```python
from tutor_from_doc import TutorEngine, OpenAILLM

llm = OpenAILLM(api_key="your-openai-api-key", model="gpt-4o-mini")
tutor = TutorEngine(llm=llm)
tutor.ingest_document("physics_guide.pdf")
```

### Using Local Ollama (e.g. Llama 3 / Phi 3):
```python
from tutor_from_doc import TutorEngine, OllamaLLM

llm = OllamaLLM(host="http://localhost:11434", model="llama3")
tutor = TutorEngine(llm=llm)
tutor.ingest_document("history_textbook.pdf")
```

---

## 🧪 Running Unit Tests

Run the test suite using `pytest`:

```bash
pytest tests/
```

Or execute the end-to-end example:

```bash
python examples/demo.py
```

---

## 📄 License

Distributed under the [MIT License](LICENSE). Open-source for education everywhere.

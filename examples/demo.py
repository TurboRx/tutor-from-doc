"""
demo.py — End-to-End Demo Script for TutorFromDoc

Demonstrates:
1. Ingesting a sample study guide document.
2. Building local vector embeddings & search index.
3. Asking grounded questions (with exact citations and page references).
4. Demonstrating strict non-hallucination refusal for out-of-bounds questions.
5. Generating grounded practice quiz questions for student revision.
"""

import os
import sys

# Ensure src/ is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tutor_from_doc.tutor import TutorEngine


def main():
    print("=" * 70)
    print(" 🎓 TUTOR-FROM-DOC: Grounded AI Tutor Demo")
    print("=" * 70)

    sample_doc_path = os.path.join(os.path.dirname(__file__), "sample_biology_guide.txt")

    if not os.path.exists(sample_doc_path):
        print(f"Error: Sample document not found at {sample_doc_path}")
        sys.exit(1)

    print(f"\n📂 Step 1: Ingesting study document '{os.path.basename(sample_doc_path)}'...")
    engine = TutorEngine()
    ingest_result = engine.ingest_document(sample_doc_path)

    print(f"✓ Ingested successfully!")
    print(f"  - Total text chunks created: {ingest_result['total_chunks']}")

    # Test Question 1: In-scope question
    q1 = "What enzyme catalyzes carbon fixation in the Calvin Cycle?"
    print(f"\n❓ Step 2: Asking Grounded Question:")
    print(f"   Query: '{q1}'")
    
    ans1 = engine.ask(q1)
    print("\n🤖 Tutor Answer:")
    print(ans1["answer"])
    print("\n📚 Grounded Citations:")
    for cite in ans1["citations"]:
        print(f"  • Source: {cite['source']} (Page {cite['page']}) | Snippet: {cite['snippet']}")

    # Test Question 2: In-scope environmental factors
    q2 = "What three environmental factors affect the photosynthetic rate?"
    print(f"\n❓ Step 3: Asking Second Grounded Question:")
    print(f"   Query: '{q2}'")
    
    ans2 = engine.ask(q2)
    print("\n🤖 Tutor Answer:")
    print(ans2["answer"])

    # Test Question 3: Out-of-scope question (testing strict non-hallucination guardrail)
    q3 = "What is the capital of France?"
    print(f"\n❓ Step 4: Testing Non-Hallucination Guardrail (Out-of-bound Query):")
    print(f"   Query: '{q3}'")
    
    ans3 = engine.ask(q3)
    print("\n🛡️ Tutor Guardrail Response:")
    print(ans3["answer"])

    # Test Quiz Generation
    print(f"\n📝 Step 5: Generating Practice Quiz Questions...")
    quiz = engine.quiz(num_questions=3)
    print(f"✓ Generated {quiz['total_questions']} practice questions:")
    for q in quiz["questions"]:
        print(f"  [{q['id']}] {q['question']}")
        print(f"      Hint: {q['source_text_hint']}")

    print("\n" + "=" * 70)
    print(" 🎉 Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

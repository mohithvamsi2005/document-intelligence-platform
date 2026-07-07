"""
Test script for Phase 7 — RAG Pipeline
Run: python test_phase7.py

This is the BIG test — the full pipeline from question to answer.
"""
import time
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.vector_store import (
    create_vector_store_from_chunks,
    delete_collection
)
from src.generation.rag_engine import run_rag_pipeline

PDF_PATH = "data/sample.pdf"

# ── Setup: rebuild vector store ──────────────────────────────
print("=" * 60)
print("SETUP: Building Vector Store")
print("=" * 60)
delete_collection()
time.sleep(1)

documents = extract_text_from_pdf(PDF_PATH)
chunks    = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
create_vector_store_from_chunks(chunks)
print(f"✅ Vector store ready with {len(chunks)} chunks\n")

# ── Test 1: Basic RAG question ───────────────────────────────
print("=" * 60)
print("TEST 1: Basic RAG Question")
print("=" * 60)
result = run_rag_pipeline(
    question="What is this document about?",
    k=3
)

print(f"Question : What is this document about?")
print(f"Answer   : {result['answer']}")
print()
print("Citations:")
for c in result["citations"]:
    print(f"  [{c['citation_number']}] {c['source']} — Page {c['page_number']}")

# ── Test 2: Token usage stats ────────────────────────────────
print()
print("=" * 60)
print("TEST 2: Token Usage & Cost")
print("=" * 60)
stats = result["usage_stats"]
print(f"  Provider        : {stats['provider']}")
print(f"  Model           : {stats['model']}")
print(f"  Input tokens    : {stats['input_tokens']}")
print(f"  Output tokens   : {stats['output_tokens']}")
print(f"  Total tokens    : {stats['total_tokens']}")
print(f"  Cost            : ${stats['estimated_cost_usd']}")
print(f"  Latency         : {stats['latency_seconds']}s")

# ── Test 3: Multi-turn conversation (chat history) ───────────
print()
print("=" * 60)
print("TEST 3: Multi-Turn Conversation")
print("=" * 60)

# First turn
r1 = run_rag_pipeline(
    question="What is the main topic?",
    k=3
)
print(f"Turn 1 Q : What is the main topic?")
print(f"Turn 1 A : {r1['answer'][:200]}...")

# Build history from turn 1
history = [
    {"role": "user",      "content": "What is the main topic?"},
    {"role": "assistant", "content": r1["answer"]}
]

# Second turn — follow-up question using history
r2 = run_rag_pipeline(
    question="Can you elaborate more on that?",
    chat_history=history,
    k=3
)
print()
print(f"Turn 2 Q          : Can you elaborate more on that?")
print(f"Rewritten query   : {r2['rewritten_query']}")
print(f"Turn 2 A          : {r2['answer'][:200]}...")

# ── Test 4: Question not in document ────────────────────────
print()
print("=" * 60)
print("TEST 4: Question Outside Document Scope")
print("=" * 60)
r3 = run_rag_pipeline(
    question="What is the population of Mars?",
    k=3
)
print(f"Question : What is the population of Mars?")
print(f"Answer   : {r3['answer']}")
print("(Should say it cannot find this info — not hallucinate)")

# ── Sanity check ─────────────────────────────────────────────
print()
print("=" * 60)
print("FINAL SANITY CHECK")
print("=" * 60)
print(f"  Answer generated      : {'✅' if result['answer'] else '❌'}")
print(f"  Citations returned    : {'✅' if result['citations'] else '❌'}")
print(f"  Token stats tracked   : {'✅' if result['usage_stats']['total_tokens'] > 0 else '❌'}")
print(f"  Multi-turn works      : {'✅' if r2['answer'] else '❌'}")
print(f"  Query rewriting works : {'✅' if r2['rewritten_query'] != 'Can you elaborate more on that?' else '⚠️ (no rewrite needed)'}")
print(f"  Hallucination check   : {'✅' if 'not' in r3['answer'].lower() or 'cannot' in r3['answer'].lower() else '⚠️ check manually'}")
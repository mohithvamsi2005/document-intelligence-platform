"""
Test script for Phase 9 — Advanced Retrieval
Run: python test_phase9.py
"""
import time
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.vector_store import (
    create_vector_store_from_chunks,
    delete_collection
)
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import hybrid_search, hybrid_search_stats
from src.retrieval.reranker import rerank_documents
from src.generation.rag_engine import run_rag_pipeline

PDF_PATH = "data/sample.pdf"
QUERY    = "What is this document about?"

# ── Setup ────────────────────────────────────────────────────
print("=" * 60)
print("SETUP: Building Vector Store")
print("=" * 60)
delete_collection()
time.sleep(1)

documents = extract_text_from_pdf(PDF_PATH)
chunks    = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
create_vector_store_from_chunks(chunks)
print(f"✅ Ready — {len(chunks)} chunks\n")

# ── Test 1: BM25 Search ──────────────────────────────────────
print("=" * 60)
print("TEST 1: BM25 Keyword Search")
print("=" * 60)
bm25     = BM25Retriever(chunks)
bm25_res = bm25.search(QUERY, k=3)
print(f"Query  : '{QUERY}'")
print(f"Results: {len(bm25_res)} chunks")
for i, doc in enumerate(bm25_res):
    print(f"  [{i+1}] {doc.metadata.get('source')} p{doc.metadata.get('page_number')} | {doc.page_content[:80]}...")

# ── Test 2: Hybrid Search ────────────────────────────────────
print()
print("=" * 60)
print("TEST 2: Hybrid Search (BM25 + Vector)")
print("=" * 60)
hybrid_res = hybrid_search(
    query=QUERY,
    chunks=chunks,
    k=3
)
print(f"Results: {len(hybrid_res)} chunks")
for i, doc in enumerate(hybrid_res):
    print(f"  [{i+1}] {doc.metadata.get('source')} p{doc.metadata.get('page_number')} | {doc.page_content[:80]}...")

# ── Test 3: Hybrid Stats ─────────────────────────────────────
print()
print("=" * 60)
print("TEST 3: Hybrid Search Stats")
print("=" * 60)
stats = hybrid_search_stats(QUERY, chunks, k=3)
for key, val in stats.items():
    print(f"  {key}: {val}")

# ── Test 4: Reranking ────────────────────────────────────────
print()
print("=" * 60)
print("TEST 4: Reranking")
print("=" * 60)
scored = rerank_documents(QUERY, hybrid_res)
print(f"Reranked {len(scored)} chunks:")
for i, (doc, score) in enumerate(scored):
    print(f"  [{i+1}] Score: {round(score,4)} | {doc.page_content[:80]}...")

# ── Test 5: Full RAG with Hybrid + Reranking ─────────────────
print()
print("=" * 60)
print("TEST 5: Full RAG Pipeline — Hybrid + Reranking")
print("=" * 60)
result = run_rag_pipeline(
    question=QUERY,
    k=3,
    use_hybrid=True,
    use_reranking=True,
    all_chunks=chunks
)
print(f"Answer : {result['answer'][:300]}...")
print(f"Sources: {len(result['citations'])}")

# ── Sanity Check ─────────────────────────────────────────────
print()
print("=" * 60)
print("FINAL SANITY CHECK")
print("=" * 60)
print(f"  BM25 works          : {'✅' if len(bm25_res) > 0 else '❌'}")
print(f"  Hybrid works        : {'✅' if len(hybrid_res) > 0 else '❌'}")
print(f"  Reranking works     : {'✅' if len(scored) > 0 else '❌'}")
print(f"  Full pipeline works : {'✅' if result['answer'] else '❌'}")
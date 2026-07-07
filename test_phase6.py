"""
Test script for Phase 6 — Retrieval Layer
Run: python test_phase6.py

NOTE: Requires Phase 5 vector store to exist.
      If chroma_db/ is missing, run test_phase5.py first.
"""
import os
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.vector_store import (
    create_vector_store_from_chunks,
    delete_collection
)
from src.retrieval.retriever import (
    retrieve_relevant_chunks,
    retrieve_with_scores,
    format_chunks_as_context,
    build_citations,
    run_retrieval_pipeline,
    deduplicate_chunks
)

PDF_PATH = "data/sample.pdf"

# ── Setup: rebuild vector store fresh ───────────────────────
print("=" * 60)
print("SETUP: Rebuilding Vector Store")
print("=" * 60)
delete_collection()

import time
time.sleep(1)

documents = extract_text_from_pdf(PDF_PATH)
chunks    = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
create_vector_store_from_chunks(chunks)
print(f"✅ Vector store ready with {len(chunks)} chunks")

# ── Step 1: Basic retrieval ──────────────────────────────────
print()
print("=" * 60)
print("STEP 1: Basic Retrieval")
print("=" * 60)
query   = "What is this document about?"
results = retrieve_relevant_chunks(query=query, k=2)
print(f"Query  : '{query}'")
print(f"Results: {len(results)} chunks retrieved")

# ── Step 2: Retrieval with scores ────────────────────────────
print()
print("=" * 60)
print("STEP 2: Retrieval With Scores")
print("=" * 60)
scored = retrieve_with_scores(query=query, k=3)
for i, (doc, score) in enumerate(scored):
    print(f"  Rank {i+1} | Score: {round(score,4)} | "
          f"{doc.page_content[:80]}...")

# ── Step 3: Deduplication ────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Deduplication")
print("=" * 60)
before = retrieve_relevant_chunks(query=query, k=4)
after  = deduplicate_chunks(before)
print(f"  Chunks before dedup : {len(before)}")
print(f"  Chunks after dedup  : {len(after)}")
print(f"  Duplicates removed  : {len(before) - len(after)}")

# ── Step 4: Context formatting ───────────────────────────────
print()
print("=" * 60)
print("STEP 4: Context Formatted for LLM")
print("=" * 60)
context = format_chunks_as_context(results)
print(context[:600])
print("...")

# ── Step 5: Citations ────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Source Citations")
print("=" * 60)
citations = build_citations(results)
for c in citations:
    print(f"  [{c['citation_number']}] {c['source']} — Page {c['page_number']}")
    print(f"      Preview: {c['preview'][:100]}...")

# ── Step 6: Full pipeline ────────────────────────────────────
print()
print("=" * 60)
print("STEP 6: Full Retrieval Pipeline")
print("=" * 60)
result = run_retrieval_pipeline(
    query=query,
    k=3,
    deduplicate=True
)
print("Pipeline Stats:")
for key, value in result["stats"].items():
    print(f"  {key}: {value}")

# ── Step 7: Sanity check ─────────────────────────────────────
print()
print("=" * 60)
print("STEP 7: Sanity Check")
print("=" * 60)
print(f"  Retrieval works  : {'✅' if len(results) > 0 else '❌'}")
print(f"  Scores returned  : {'✅' if len(scored) > 0 else '❌'}")
print(f"  Context formatted: {'✅' if len(context) > 0 else '❌'}")
print(f"  Citations built  : {'✅' if len(citations) > 0 else '❌'}")
print(f"  Pipeline runs    : {'✅' if result['chunks'] is not None else '❌'}")
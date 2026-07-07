"""
Test script for Phase 5 — ChromaDB Vector Store
Run: python test_phase5.py
"""
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.vector_store import (
    create_vector_store_from_chunks,
    get_vector_store,
    similarity_search,
    similarity_search_with_scores,
    get_collection_stats,
    delete_collection
)

PDF_PATH = "data/sample.pdf"

# ── Step 1: Full ingestion pipeline ─────────────────────────
print("=" * 60)
print("STEP 1: Run Full Ingestion Pipeline")
print("=" * 60)
documents = extract_text_from_pdf(PDF_PATH)
chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
print(f"✅ Loaded {len(documents)} pages → {len(chunks)} chunks")

# ── Step 2: Create vector store ──────────────────────────────
# ── Step 2: Create vector store ──────────────────────────────
print()
print("=" * 60)
print("STEP 2: Create Vector Store")
print("=" * 60)

# Clean slate — delete any old test data first
# If this fails on Windows, manually delete chroma_db/ folder
delete_collection()

# Small pause after deletion before creating new store
import time
time.sleep(1)

vector_store = create_vector_store_from_chunks(
    chunks=chunks,
    collection_name="documents"
)

# ── Step 3: Collection stats ─────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Collection Stats")
print("=" * 60)
stats = get_collection_stats(vector_store)
for key, value in stats.items():
    print(f"  {key}: {value}")

# ── Step 4: Test persistence ─────────────────────────────────
print()
print("=" * 60)
print("STEP 4: Test Persistence (reload from disk)")
print("=" * 60)
# Simulate app restart — load from disk instead of rebuilding
reloaded_store = get_vector_store(collection_name="documents")
reloaded_stats = get_collection_stats(reloaded_store)
print(f"  Chunks after reload: {reloaded_stats['total_chunks_stored']}")
print(f"  Persistence working: {'✅ YES' if reloaded_stats['total_chunks_stored'] > 0 else '❌ NO'}")

# ── Step 5: Similarity search ────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Similarity Search")
print("=" * 60)
query = "What is this document about?"
print(f"Query: '{query}'")
print()

results = similarity_search(reloaded_store, query=query, k=2)

for i, doc in enumerate(results):
    print(f"── Result {i+1} ──────────────────────────")
    print(f"  Source  : {doc.metadata.get('source')}")
    print(f"  Page    : {doc.metadata.get('page_number')}")
    print(f"  Preview : {doc.page_content[:200]}...")
    print()

# ── Step 6: Search with scores ───────────────────────────────
print("=" * 60)
print("STEP 6: Search With Relevance Scores")
print("=" * 60)
scored_results = similarity_search_with_scores(
    reloaded_store,
    query=query,
    k=2
)

for i, (doc, score) in enumerate(scored_results):
    print(f"  Result {i+1} | Score: {round(score, 4)} | "
          f"Preview: {doc.page_content[:100]}...")
print()
print("  (Lower score = more similar = more relevant)")

# ── Step 7: Final sanity check ───────────────────────────────
print()
print("=" * 60)
print("STEP 7: Final Sanity Check")
print("=" * 60)
print(f"  Chunks stored    : {'✅' if stats['total_chunks_stored'] == len(chunks) else '❌'} {stats['total_chunks_stored']}/{len(chunks)}")
print(f"  Search returned  : {'✅' if len(results) > 0 else '❌'} {len(results)} results")
print(f"  Scores returned  : {'✅' if len(scored_results) > 0 else '❌'} {len(scored_results)} scored results")
print(f"  Persisted to disk: ✅ chroma_db/")
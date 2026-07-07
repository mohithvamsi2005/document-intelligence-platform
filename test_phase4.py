"""
Test script for Phase 4 — Free Local Embeddings
Run: python test_phase4.py
"""
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.text_chunker import chunk_documents
from src.ingestion.embeddings import (
    generate_embeddings_for_chunks,
    generate_query_embedding,
    get_embedding_stats,
    EMBEDDING_DIMENSIONS
)

PDF_PATH = "data/sample.pdf"

# ── Step 1: Load + Chunk ─────────────────────────────────────
print("=" * 60)
print("STEP 1: Load and Chunk PDF")
print("=" * 60)
documents = extract_text_from_pdf(PDF_PATH)
chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
print(f"Chunks ready for embedding: {len(chunks)}")

# ── Step 2: Generate embeddings (FREE, local) ────────────────
print()
print("=" * 60)
print("STEP 2: Generate Embeddings (Local, Free)")
print("=" * 60)
chunks, embeddings = generate_embeddings_for_chunks(chunks)

# ── Step 3: Inspect the embedding vector ─────────────────────
print()
print("=" * 60)
print("STEP 3: Inspect Embedding Vector")
print("=" * 60)
first_embedding = embeddings[0]
print(f"Type           : {type(first_embedding)}")
print(f"Dimensions     : {len(first_embedding)}")
print(f"First 5 values : {[round(x, 6) for x in first_embedding[:5]]}")
print(f"Last 5 values  : {[round(x, 6) for x in first_embedding[-5:]]}")

# ── Step 4: Stats ────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4: Embedding Statistics")
print("=" * 60)
stats = get_embedding_stats(embeddings)
for key, value in stats.items():
    print(f"  {key}: {value}")

# ── Step 5: Query embedding ──────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Query Embedding Test")
print("=" * 60)
query = "What is the main topic of this document?"
query_embedding = generate_query_embedding(query)
print(f"Query          : '{query}'")
print(f"Dimensions     : {len(query_embedding)}")
print(f"First 5 values : {[round(x, 6) for x in query_embedding[:5]]}")

# ── Step 6: Sanity Check ─────────────────────────────────────
print()
print("=" * 60)
print("STEP 6: Sanity Check")
print("=" * 60)
dims_correct = len(embeddings[0]) == EMBEDDING_DIMENSIONS
counts_match = len(chunks) == len(embeddings)

print(f"  Chunks count     : {len(chunks)}")
print(f"  Embeddings count : {len(embeddings)}")
print(f"  Counts match?    : {'✅ YES' if counts_match else '❌ NO'}")
print(f"  Vector dims      : {'✅ 384' if dims_correct else '❌ Wrong size'}")
print(f"  Cost             : ✅ $0.00 (local model)")
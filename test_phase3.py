"""
Test script for Phase 3 — Text Chunking
Run: python test_phase3.py
"""
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.text_chunker import chunk_documents, get_chunking_stats

PDF_PATH = "data/sample.pdf"

# ── Step 1: Load PDF (Phase 2) ──────────────────────────────
print("=" * 60)
print("STEP 1: Loading PDF")
print("=" * 60)
documents = extract_text_from_pdf(PDF_PATH)
print(f"Pages loaded: {len(documents)}")

# ── Step 2: Chunk the documents ─────────────────────────────
print()
print("=" * 60)
print("STEP 2: Chunking Documents")
print("=" * 60)
chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
print(f"Total chunks created: {len(chunks)}")

# ── Step 3: Show stats ──────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Chunking Statistics")
print("=" * 60)
stats = get_chunking_stats(documents, chunks)
for key, value in stats.items():
    print(f"  {key}: {value}")

# ── Step 4: Inspect first 3 chunks ─────────────────────────
print()
print("=" * 60)
print("STEP 4: Inspecting First 3 Chunks")
print("=" * 60)
for i, chunk in enumerate(chunks[:3]):
    print(f"\n── CHUNK {i+1} ──────────────────")
    print(f"  Characters : {len(chunk.page_content)}")
    print(f"  Metadata   : {chunk.metadata}")
    print(f"  Preview    : {chunk.page_content[:150]}...")

# ── Step 5: Verify overlap ──────────────────────────────────
if len(chunks) >= 2:
    print()
    print("=" * 60)
    print("STEP 5: Verifying Overlap Between Chunk 1 and Chunk 2")
    print("=" * 60)
    end_of_chunk1 = chunks[0].page_content[-100:]
    start_of_chunk2 = chunks[1].page_content[:100]
    print(f"  Last 100 chars of Chunk 1 : ...{end_of_chunk1}")
    print(f"  First 100 chars of Chunk 2: {start_of_chunk2}...")
    print()
    print("  👆 You should see overlapping text above")
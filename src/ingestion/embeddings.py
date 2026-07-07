"""
Embeddings Module
------------------
Converts text chunks into numerical vectors using a FREE local model:
    sentence-transformers/all-MiniLM-L6-v2

WHY this model:
- Completely free, runs on your machine
- No API key needed, no internet after first download
- 384-dimensional vectors (vs OpenAI's 1536 — smaller but very capable)
- Most popular open-source embedding model in the world
- First run: downloads ~80MB model weights (one time only)
- Every run after: instant, fully offline

This is STEP 3 of the RAG ingestion pipeline:
PDF → Text → Chunks → Embeddings (THIS FILE) → Vector DB
"""

from typing import List, Tuple
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ── Model Configuration ───────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads and returns the local HuggingFace embedding model.

    First call: downloads ~80MB model weights from HuggingFace Hub
                and caches them on your machine
    Every call after: loads instantly from local cache

    Returns:
        HuggingFaceEmbeddings instance ready to use
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(First run downloads ~80MB — subsequent runs are instant)")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},   # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}
        # normalize_embeddings=True → vectors have length 1
        # This makes cosine similarity = dot product (faster math later)
    )

    print(f"✅ Model loaded — {EMBEDDING_DIMENSIONS} dimensions per vector")
    return embedding_model


def generate_embeddings_for_chunks(
    chunks: List[Document],
    batch_size: int = 64
) -> Tuple[List[Document], List[List[float]]]:
    """
    Generates embeddings for a list of Document chunks.

    Runs entirely on your local CPU — no API calls, no cost.

    Args:
        chunks     : List of Document chunks from text_chunker.py
        batch_size : Chunks processed at once (64 is good for CPU)

    Returns:
        Tuple of:
          - The same chunks (unchanged)
          - List of embedding vectors (one 384-dim vector per chunk)
    """
    model = get_embedding_model()

    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    print(f"Model     : {EMBEDDING_MODEL_NAME}")
    print(f"Dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Device    : CPU")
    print("-" * 50)

    # Extract just the text from each chunk
    all_texts = [chunk.page_content for chunk in chunks]

    # embed_documents handles batching internally
    # No need for manual batch loop like OpenAI — HuggingFace does it
    all_embeddings = model.embed_documents(all_texts)

    print(f"✅ Done! Total embeddings generated: {len(all_embeddings)}")

    return chunks, all_embeddings


def generate_query_embedding(query: str) -> List[float]:
    """
    Generates an embedding for a single search query.
    Called at retrieval time when the user asks a question.

    Args:
        query: The user's question as a string

    Returns:
        A single embedding vector (list of 384 floats)
    """
    model = get_embedding_model()
    embedding = model.embed_query(query)
    return embedding


def get_embedding_stats(embeddings: List[List[float]]) -> dict:
    """
    Returns statistics about generated embeddings.
    Cost is always $0.00 — we're running locally!

    Args:
        embeddings: List of embedding vectors

    Returns:
        Dictionary of stats
    """
    if not embeddings:
        return {"error": "No embeddings provided"}

    stats = {
        "total_embeddings"    : len(embeddings),
        "embedding_dimensions": len(embeddings[0]),
        "model_used"          : EMBEDDING_MODEL_NAME,
        "runs_locally"        : True,
        "estimated_cost_usd"  : 0.00     # FREE 🎉
    }

    return stats
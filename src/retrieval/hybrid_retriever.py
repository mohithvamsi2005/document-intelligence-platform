"""
Hybrid Search Module
---------------------
Combines BM25 (keyword) and Vector (semantic) search results
using Reciprocal Rank Fusion (RRF).

Feature #11 — Hybrid Search

HOW RRF works:
  Each result gets a score based on its rank in each system:
  RRF score = 1/(k + rank_in_bm25) + 1/(k + rank_in_vector)

  A document ranked #1 in both systems gets the highest score.
  A document ranked #1 in one and missing from other still scores well.

WHY RRF over simple score averaging:
  BM25 scores and cosine similarity scores are on different scales.
  BM25 might give scores of 0-50, cosine gives 0-1.
  You can't average them directly.
  RRF uses only the RANK (position) not the raw score,
  making it scale-independent and robust.
"""

from typing import List, Dict, Optional
from langchain.schema import Document
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import (
    get_vector_store,
    similarity_search_with_scores
)


def reciprocal_rank_fusion(
    bm25_results: List[Document],
    vector_results: List[Document],
    k: int = 60,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5
) -> List[Document]:
    """
    Merges BM25 and vector search results using
    Reciprocal Rank Fusion (RRF).

    Args:
        bm25_results   : Ranked list from BM25
        vector_results : Ranked list from vector search
        k              : RRF constant (60 is standard)
        bm25_weight    : Weight for BM25 scores (0-1)
        vector_weight  : Weight for vector scores (0-1)

    Returns:
        Merged and re-ranked list of Documents
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    # Score BM25 results by rank
    for rank, doc in enumerate(bm25_results):
        # Use first 100 chars as unique key
        doc_id = doc.page_content[:100]
        rrf_score = bm25_weight * (1.0 / (k + rank + 1))
        scores[doc_id]  = scores.get(doc_id, 0) + rrf_score
        doc_map[doc_id] = doc

    # Score vector results by rank
    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content[:100]
        rrf_score = vector_weight * (1.0 / (k + rank + 1))
        scores[doc_id]  = scores.get(doc_id, 0) + rrf_score
        doc_map[doc_id] = doc

    # Sort by combined RRF score (descending)
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [doc_map[doc_id] for doc_id in sorted_ids]


def hybrid_search(
    query: str,
    chunks: List[Document],
    collection_name: str = "documents",
    k: int = 4,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5
) -> List[Document]:
    """
    Full hybrid search pipeline.
    Runs BM25 + Vector search in parallel and merges results.

    Args:
        query           : User's search query
        chunks          : All document chunks (needed for BM25 index)
        collection_name : ChromaDB collection for vector search
        k               : Number of final results to return
        bm25_weight     : How much to weight keyword results (0-1)
        vector_weight   : How much to weight semantic results (0-1)

    Returns:
        Merged top-k Documents from both search methods
    """
    # ── BM25 Search ───────────────────────────────────────────
    bm25 = BM25Retriever(chunks)
    bm25_results = bm25.search(query, k=k)

    # ── Vector Search ─────────────────────────────────────────
    vs = get_vector_store(collection_name=collection_name)
    vector_results = vs.similarity_search(query, k=k)

    # ── Merge with RRF ────────────────────────────────────────
    merged = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vector_results,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight
    )

    # Return top k of merged results
    return merged[:k]


def hybrid_search_stats(
    query: str,
    chunks: List[Document],
    collection_name: str = "documents",
    k: int = 4
) -> Dict:
    """
    Returns detailed stats comparing BM25 vs Vector results.
    Used in the Evaluation Dashboard (Phase 10).

    Args:
        query           : Search query
        chunks          : All document chunks
        collection_name : ChromaDB collection name
        k               : Results per method

    Returns:
        Dictionary with per-method results and overlap stats
    """
    # BM25 results
    bm25 = BM25Retriever(chunks)
    bm25_results = bm25.search(query, k=k)
    bm25_ids     = set(d.page_content[:100] for d in bm25_results)

    # Vector results
    vs             = get_vector_store(collection_name=collection_name)
    vector_results = vs.similarity_search(query, k=k)
    vector_ids     = set(d.page_content[:100] for d in vector_results)

    # Overlap
    overlap = bm25_ids & vector_ids

    stats = {
        "bm25_results_count"  : len(bm25_results),
        "vector_results_count": len(vector_results),
        "overlap_count"       : len(overlap),
        "unique_to_bm25"      : len(bm25_ids - vector_ids),
        "unique_to_vector"    : len(vector_ids - bm25_ids),
        "bm25_top_source"     : bm25_results[0].metadata.get("source", "?") if bm25_results else "none",
        "vector_top_source"   : vector_results[0].metadata.get("source", "?") if vector_results else "none"
    }

    return stats
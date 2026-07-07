"""
Reranking Module
-----------------
Feature #12 — Reranking Layer

After retrieval (BM25 + Vector), we have top-k chunks.
Reranking uses a Cross-Encoder to more accurately score
which chunks BEST answer the query.

Cross-Encoder vs Bi-Encoder:
- Bi-encoder (retrieval): encodes query and doc SEPARATELY → fast
- Cross-encoder (reranking): encodes query + doc TOGETHER → accurate
"""

# ── Imports at the TOP — always ───────────────────────────────
from typing import List, Tuple, Optional
from langchain.schema import Document

# Lazy load — only downloads model when reranking is first used
_cross_encoder = None


def get_cross_encoder():
    """
    Lazily loads the cross-encoder model.
    Downloads ~80MB on first call, cached locally after.

    Returns:
        CrossEncoder model instance
    """
    global _cross_encoder

    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("Loading reranking model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("(First run downloads ~80MB — subsequent runs instant)")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Reranking model loaded")

    return _cross_encoder


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: Optional[int] = None
) -> List[Tuple[Document, float]]:
    """
    Reranks retrieved documents using a Cross-Encoder model.

    Args:
        query     : The user's question
        documents : Retrieved documents to rerank
        top_n     : How many to return (None = return all)

    Returns:
        List of (Document, score) tuples, best first
        Higher score = more relevant
    """
    if not documents:
        return []

    if top_n is None:
        top_n = len(documents)

    model = get_cross_encoder()

    # Build (query, document_text) pairs
    pairs = [(query, doc.page_content) for doc in documents]

    # Cross-encoder scores all pairs
    scores = model.predict(pairs)

    # Sort by score descending
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return scored_docs[:top_n]


def rerank_and_return_docs(
    query: str,
    documents: List[Document],
    top_n: Optional[int] = None
) -> List[Document]:
    """
    Reranks and returns only Documents (without scores).

    Args:
        query     : The user's question
        documents : Retrieved documents
        top_n     : How many to return

    Returns:
        Reranked list of Documents, best first
    """
    scored = rerank_documents(query, documents, top_n)
    return [doc for doc, score in scored]
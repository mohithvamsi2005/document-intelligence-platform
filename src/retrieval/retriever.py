"""
Retriever Module
-----------------
Sits between the vector store and the LLM.
Takes a user query, retrieves the most relevant chunks,
formats them for the LLM, and builds source citations.

This is the RETRIEVAL step of RAG:
User Query → [THIS FILE] → Relevant Chunks → LLM → Answer

Key responsibilities:
1. Semantic search against ChromaDB
2. Metadata filtering (by source file, page number)
3. Deduplication (remove near-identical chunks)
4. Context formatting (turn chunks into LLM-ready text)
5. Citation building (track exactly where each answer came from)
"""

from typing import List, Optional, Dict
from langchain.schema import Document
from src.retrieval.vector_store import (
    similarity_search,
    similarity_search_with_scores,
    get_vector_store
)


# ── Core Retrieval ────────────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    collection_name: str = "documents",
    k: int = 4,
    score_threshold: Optional[float] = None
) -> List[Document]:
    """
    Main retrieval function. Given a user query, returns the
    most relevant document chunks from ChromaDB.

    Args:
        query           : User's question
        collection_name : Which ChromaDB collection to search
        k               : Number of chunks to retrieve
        score_threshold : Optional max distance score to filter
                          low-quality results (lower = stricter)
                          ChromaDB scores: 0.0 (identical) to 2.0 (opposite)
                          Typical good results are below 1.0

    Returns:
        List of relevant Document chunks, best match first
    """
    vector_store = get_vector_store(collection_name=collection_name)

    if score_threshold is not None:
        # Get results with scores so we can filter by quality
        scored_results = similarity_search_with_scores(
            vector_store=vector_store,
            query=query,
            k=k
        )
        # Keep only results below the score threshold
        results = [
            doc for doc, score in scored_results
            if score <= score_threshold
        ]
    else:
        results = similarity_search(
            vector_store=vector_store,
            query=query,
            k=k
        )

    return results


def retrieve_with_scores(
    query: str,
    collection_name: str = "documents",
    k: int = 4
) -> List[tuple]:
    """
    Retrieves chunks AND their relevance scores.
    Used by the Evaluation Dashboard (Phase 10).

    Args:
        query           : User's question
        collection_name : Which ChromaDB collection to search
        k               : Number of results

    Returns:
        List of (Document, float) tuples — score is cosine distance
        Lower score = more relevant
    """
    vector_store = get_vector_store(collection_name=collection_name)

    scored_results = similarity_search_with_scores(
        vector_store=vector_store,
        query=query,
        k=k
    )

    return scored_results


# ── Metadata Filtering ────────────────────────────────────────────────────────

def retrieve_from_source(
    query: str,
    source_filename: str,
    collection_name: str = "documents",
    k: int = 4
) -> List[Document]:
    """
    Retrieves chunks from ONE specific PDF file only.
    This is Document Metadata Filtering — Feature #14.

    Example use case:
        User uploads 3 PDFs, but asks "search only in contract.pdf"

    Args:
        query           : User's question
        source_filename : Exact filename to filter by (e.g. "contract.pdf")
        collection_name : ChromaDB collection name
        k               : Number of results

    Returns:
        Relevant chunks from that specific file only
    """
    from langchain_chroma import Chroma
    from src.ingestion.embeddings import get_embedding_model

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory="chroma_db"
    )

    # ChromaDB's where filter — only return chunks where
    # metadata["source"] matches our filename
    results = vector_store.similarity_search(
        query=query,
        k=k,
        filter={"source": source_filename}
    )

    return results


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_chunks(
    chunks: List[Document],
    similarity_threshold: int = 50
) -> List[Document]:
    """
    Removes near-duplicate chunks from retrieval results.

    WHY: Sometimes two consecutive chunks in a document are very
    similar due to overlap. Sending both to the LLM wastes tokens
    and confuses the model. We deduplicate by checking if any two
    chunks share more than X% of their content.

    Args:
        chunks               : Retrieved chunks (may have duplicates)
        similarity_threshold : If two chunks share this % of words,
                               drop the second one (default 50%)

    Returns:
        Deduplicated list of chunks
    """
    if not chunks:
        return chunks

    unique_chunks = [chunks[0]]

    for candidate in chunks[1:]:
        is_duplicate = False

        candidate_words = set(candidate.page_content.lower().split())

        for existing in unique_chunks:
            existing_words = set(existing.page_content.lower().split())

            # Jaccard similarity: intersection / union
            if not candidate_words or not existing_words:
                continue

            intersection = len(candidate_words & existing_words)
            union = len(candidate_words | existing_words)
            jaccard = (intersection / union) * 100

            if jaccard >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_chunks.append(candidate)

    return unique_chunks


# ── Context Formatting ────────────────────────────────────────────────────────

def format_chunks_as_context(chunks: List[Document]) -> str:
    """
    Converts retrieved chunks into a single formatted string
    ready to be inserted into an LLM prompt.

    WHY formatting matters:
    The LLM needs clearly separated, labeled context blocks so it
    can distinguish between different sources and cite them properly.

    Args:
        chunks : Retrieved Document chunks

    Returns:
        A formatted string like:
        ---
        [Source 1] report.pdf | Page 2
        <text content here>
        ---
        [Source 2] contract.pdf | Page 5
        <text content here>
        ---
    """
    if not chunks:
        return "No relevant context found."

    context_parts = []

    for i, chunk in enumerate(chunks):
        source   = chunk.metadata.get("source", "Unknown")
        page_num = chunk.metadata.get("page_number", "?")
        content  = chunk.page_content.strip()

        context_parts.append(
            f"[Source {i+1}] {source} | Page {page_num}\n{content}"
        )

    formatted = "\n\n---\n\n".join(context_parts)
    return formatted


# ── Citation Builder ──────────────────────────────────────────────────────────

def build_citations(chunks: List[Document]) -> List[Dict]:
    """
    Builds a structured list of citations from retrieved chunks.
    This powers Feature #8 (Source Citations) in the UI.

    Args:
        chunks : Retrieved Document chunks

    Returns:
        List of citation dictionaries, e.g.:
        [
            {
                "citation_number" : 1,
                "source"          : "report.pdf",
                "page_number"     : 3,
                "preview"         : "First 150 chars of the chunk..."
            },
            ...
        ]
    """
    citations = []

    for i, chunk in enumerate(chunks):
        citation = {
            "citation_number" : i + 1,
            "source"          : chunk.metadata.get("source", "Unknown"),
            "page_number"     : chunk.metadata.get("page_number", "?"),
            "preview"         : chunk.page_content[:150].strip() + "..."
        }
        citations.append(citation)

    return citations


# ── Full Retrieval Pipeline ───────────────────────────────────────────────────

def run_retrieval_pipeline(
    query: str,
    collection_name: str = "documents",
    k: int = 4,
    source_filter: Optional[str] = None,
    deduplicate: bool = True,
    score_threshold: Optional[float] = None
) -> Dict:
    """
    The complete retrieval pipeline in one function.
    This is what Phase 7 (RAG) will call.

    Args:
        query           : User's question
        collection_name : ChromaDB collection to search
        k               : Number of chunks to retrieve
        source_filter   : Optional filename to restrict search to
        deduplicate     : Whether to remove near-duplicate chunks
        score_threshold : Optional quality filter on results

    Returns:
        Dictionary with:
          - chunks     : The retrieved Document objects
          - context    : Formatted string for LLM prompt
          - citations  : Structured list for UI display
          - stats      : Retrieval metadata
    """
    # Step 1: Retrieve
    if source_filter:
        chunks = retrieve_from_source(
            query=query,
            source_filename=source_filter,
            collection_name=collection_name,
            k=k
        )
    else:
        chunks = retrieve_relevant_chunks(
            query=query,
            collection_name=collection_name,
            k=k,
            score_threshold=score_threshold
        )

    # Step 2: Deduplicate
    if deduplicate:
        chunks_before = len(chunks)
        chunks = deduplicate_chunks(chunks)
        chunks_after = len(chunks)
    else:
        chunks_before = chunks_after = len(chunks)

    # Step 3: Format for LLM
    context   = format_chunks_as_context(chunks)
    citations = build_citations(chunks)

    # Step 4: Build stats
    stats = {
        "query"               : query,
        "chunks_retrieved"    : chunks_before,
        "chunks_after_dedup"  : chunks_after,
        "duplicates_removed"  : chunks_before - chunks_after,
        "source_filter"       : source_filter,
        "collection"          : collection_name
    }

    return {
        "chunks"    : chunks,
        "context"   : context,
        "citations" : citations,
        "stats"     : stats
    }
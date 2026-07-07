"""
Text Chunker Module
--------------------
Takes LangChain Document objects (output of pdf_loader.py) and
splits them into smaller overlapping chunks.

This is STEP 2 of the RAG ingestion pipeline:
PDF → Raw Text → Chunks (THIS FILE) → Embeddings → Vector DB

WHY chunking matters:
- Embedding models have token limits (~8191 for text-embedding-3-small)
- Smaller chunks = more precise retrieval
- Overlap prevents context loss at chunk boundaries
"""

from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """
    Creates and returns a configured text splitter.

    RecursiveCharacterTextSplitter is the recommended splitter for
    general-purpose RAG. It tries to split on natural boundaries
    in this priority order:
        1. Paragraphs  ("\n\n")
        2. Lines       ("\n")
        3. Sentences   (". ")
        4. Words       (" ")
        5. Characters  ("")   ← last resort

    This means it tries to keep paragraphs together first,
    and only splits mid-sentence if absolutely necessary.

    Args:
        chunk_size    : Max characters per chunk (default 1000)
        chunk_overlap : Characters shared between consecutive chunks (default 200)

    Returns:
        A configured RecursiveCharacterTextSplitter instance
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,          # use character count, not token count
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.
    Preserves and enriches metadata on every chunk so we always
    know which source document and page a chunk came from.

    Args:
        documents     : List of Documents from pdf_loader.py
        chunk_size    : Max characters per chunk
        chunk_overlap : Characters of overlap between chunks

    Returns:
        List of chunked Documents — more items, smaller page_content each
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)

    chunked_documents = []

    for doc_index, document in enumerate(documents):
        # Split this single document into chunks
        chunks = splitter.split_documents([document])

        # Enrich each chunk with extra metadata
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["total_chunks_in_page"] = len(chunks)
            chunk.metadata["doc_index"] = doc_index
            chunk.metadata["chunk_size_used"] = chunk_size
            chunk.metadata["chunk_overlap_used"] = chunk_overlap

            chunked_documents.append(chunk)

    return chunked_documents


def get_chunking_stats(
    original_docs: List[Document],
    chunked_docs: List[Document]
) -> dict:
    """
    Returns statistics about the chunking operation.
    Useful for the Evaluation Dashboard (Feature #16) later.

    Args:
        original_docs : The documents BEFORE chunking
        chunked_docs  : The documents AFTER chunking

    Returns:
        A dictionary of stats
    """
    total_chars_original = sum(len(d.page_content) for d in original_docs)
    total_chars_chunked = sum(len(d.page_content) for d in chunked_docs)

    chunk_lengths = [len(d.page_content) for d in chunked_docs]

    stats = {
        "original_page_count": len(original_docs),
        "total_chunks_created": len(chunked_docs),
        "avg_chunk_length_chars": round(
            total_chars_chunked / len(chunked_docs), 1
        ) if chunked_docs else 0,
        "min_chunk_length_chars": min(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_length_chars": max(chunk_lengths) if chunk_lengths else 0,
        "total_chars_original": total_chars_original,
        "expansion_ratio": round(
            len(chunked_docs) / len(original_docs), 2
        ) if original_docs else 0
    }
    return stats
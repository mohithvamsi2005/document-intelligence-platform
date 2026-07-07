"""
Vector Store Module
--------------------
Stores document chunk embeddings in ChromaDB and provides
similarity search functionality.

This is STEP 4 of the RAG ingestion pipeline:
PDF → Text → Chunks → Embeddings → Vector DB (THIS FILE)

And STEP 1 of the retrieval pipeline:
User Query → [VECTOR SEARCH] → Top K Chunks → LLM → Answer

WHY ChromaDB:
- Runs locally (no server needed, no cost)
- Persists to disk (survives app restarts)
- Simple Python API
- Supports metadata filtering
- Production-ready for small-to-medium scale

"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"
from typing import List, Optional
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.ingestion.embeddings import get_embedding_model


# ── Constants ─────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION  = "documents"


# ── Core Functions ────────────────────────────────────────────────────────────

def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str = CHROMA_PERSIST_DIR
) -> Chroma:
    """
    Loads an existing ChromaDB collection from disk.
    Use this when the collection already exists (app restarts).

    Args:
        collection_name : Name of the ChromaDB collection
        persist_dir     : Folder where ChromaDB saves data

    Returns:
        A Chroma vector store instance connected to existing data
    """
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )

    return vector_store


def create_vector_store_from_chunks(
    chunks: List[Document],
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str = CHROMA_PERSIST_DIR
) -> Chroma:
    """
    Creates a NEW ChromaDB collection from document chunks.
    Embeds all chunks and stores them with their metadata.

    Use this when ingesting documents for the first time.

    Args:
        chunks          : List of Document chunks from text_chunker.py
        collection_name : Name for this ChromaDB collection
        persist_dir     : Folder to persist data on disk

    Returns:
        A Chroma vector store instance with all chunks stored
    """
    os.makedirs(persist_dir, exist_ok=True)

    embedding_model = get_embedding_model()

    print(f"Creating vector store...")
    print(f"  Collection : '{collection_name}'")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  Persist dir: {persist_dir}")
    print("-" * 50)

    # Chroma.from_documents does 3 things in one call:
    #   1. Embeds all chunks using embedding_model
    #   2. Stores vectors + metadata in ChromaDB
    #   3. Persists everything to disk automatically
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir
    )

    print(f"✅ Vector store created and persisted to '{persist_dir}/'")
    return vector_store


def add_chunks_to_vector_store(
    vector_store: Chroma,
    new_chunks: List[Document]
) -> Chroma:
    """
    Adds NEW chunks to an EXISTING vector store.
    Used when a user uploads additional PDFs after the first one.

    Args:
        vector_store : Existing Chroma instance
        new_chunks   : New Document chunks to add

    Returns:
        The same vector store (now with more documents)
    """
    vector_store.add_documents(new_chunks)
    print(f"✅ Added {len(new_chunks)} new chunks to existing vector store")
    return vector_store


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Finds the K most semantically similar chunks to a query.

    This is the core of RAG retrieval — given a user question,
    find the most relevant document chunks.

    Args:
        vector_store : The Chroma instance to search
        query        : User's question as a string
        k            : Number of results to return (default 4)

    Returns:
        List of the K most relevant Document chunks
    """
    results = vector_store.similarity_search(
        query=query,
        k=k
    )
    return results


def similarity_search_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = 4
) -> List[tuple]:
    """
    Same as similarity_search but also returns relevance scores.
    Score is cosine distance (lower = more similar).

    Used in the Evaluation Dashboard (Phase 10) to show
    how confident the retrieval was.

    Args:
        vector_store : The Chroma instance to search
        query        : User's question as a string
        k            : Number of results to return

    Returns:
        List of (Document, score) tuples, sorted by relevance
    """
    results = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )
    return results


def get_collection_stats(vector_store: Chroma) -> dict:
    """
    Returns statistics about what's stored in the vector store.

    Args:
        vector_store : The Chroma instance to inspect

    Returns:
        Dictionary of stats
    """
    collection = vector_store._collection
    count = collection.count()

    stats = {
        "total_chunks_stored" : count,
        "collection_name"     : collection.name,
        "persist_directory"   : CHROMA_PERSIST_DIR,
        "embedding_model"     : "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimensions": 384
    }

    return stats


def delete_collection(
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str = CHROMA_PERSIST_DIR
) -> None:
    """
    Deletes a ChromaDB collection entirely.
    Handles Windows file-lock issues with SQLite.

    Args:
        collection_name : Name of collection to delete
        persist_dir     : Where ChromaDB data lives on disk
    """
    import shutil
    import time
    import gc

    if not os.path.exists(persist_dir):
        print(f"⚠️  No collection found at '{persist_dir}/' — nothing to delete")
        return

    # Step 1: Force Python garbage collector to release any
    # ChromaDB/SQLite file handles still open in memory.
    # This is the key fix for Windows "Access Denied" errors.
    gc.collect()

    # Step 2: Small delay — gives Windows time to release locks
    time.sleep(0.5)

    # Step 3: Try deletion with retries
    # Windows sometimes needs a moment even after gc.collect()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(persist_dir)
            print(f"✅ Deleted collection '{collection_name}' from '{persist_dir}/'")
            return
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"  ⏳ File lock detected, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(1)  # wait 1 second before retrying
            else:
                # All retries failed — give clear instructions
                print(f"""
❌ Could not delete '{persist_dir}/' automatically.
   Windows is locking the SQLite file inside it.

   MANUAL FIX (do one of these):
   1. Delete the folder manually:
      → Open File Explorer
      → Navigate to your project folder
      → Delete the 'chroma_db' folder
      → Re-run the test

   2. Or run this in PowerShell:
      Remove-Item -Recurse -Force chroma_db
      python test_phase5.py
""")
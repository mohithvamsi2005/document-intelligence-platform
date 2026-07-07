"""
BM25 Retriever Module
----------------------
Implements keyword-based search using the BM25 algorithm.
Used as the keyword component of Hybrid Search (Feature #11).

BM25 = Best Match 25
- Classical information retrieval algorithm
- Used by Elasticsearch, Apache Solr, and many search engines
- Excels at exact keyword matches
- Complements vector search which excels at semantic matches

WHY BM25 alongside vector search:
- Vector search: "What is Goldman's profit?" finds chunks about
  "earnings", "net income", "financial results" (semantic)
- BM25: "Goldman Sachs Q3 2024 revenue" finds chunks with
  those EXACT words (keyword precision)
- Together: best recall possible
"""

import string
from typing import List, Optional
from langchain.schema import Document

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Run: pip install rank-bm25==0.2.2")


class BM25Retriever:
    """
    BM25-based keyword retriever.

    Usage:
        retriever = BM25Retriever(chunks)
        results   = retriever.search("Goldman Sachs revenue", k=4)
    """

    def __init__(self, documents: List[Document]):
        """
        Builds the BM25 index from a list of documents.
        Must be rebuilt whenever new documents are added.

        Args:
            documents: List of LangChain Document chunks
        """
        self.documents = documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"✅ BM25 index built with {len(documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text for BM25 indexing.
        Lowercases and removes punctuation.

        Args:
            text: Raw text string

        Returns:
            List of word tokens
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Split into words
        tokens = text.split()
        return tokens

    def search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Searches for the top-k most relevant documents
        using BM25 keyword scoring.

        Args:
            query: Search query string
            k    : Number of results to return

        Returns:
            List of top-k Document chunks by BM25 score
        """
        # Tokenize the query same way as documents
        tokenized_query = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top-k scores (descending)
        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        # Return the corresponding documents
        results = [self.documents[i] for i in top_k_indices]
        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple]:
        """
        Same as search() but returns (document, score) tuples.
        Used in hybrid search for score normalization.

        Args:
            query: Search query string
            k    : Number of results

        Returns:
            List of (Document, float) tuples
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = [
            (self.documents[i], float(scores[i]))
            for i in top_k_indices
        ]
        return results
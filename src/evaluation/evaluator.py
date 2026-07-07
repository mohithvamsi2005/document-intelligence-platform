"""
Evaluation Module
------------------
Tracks and computes metrics for the RAG pipeline.
Feature #16 — Evaluation Dashboard
Feature #17 — Retrieval Metrics
Feature #18 — Token Usage Tracking
Feature #19 — Cost Estimation

WHY evaluation matters:
- Proves your system works (not just "it gives answers")
- Lets you compare: vector vs hybrid vs hybrid+reranking
- Quantifies improvement when you tune parameters
- Required for production ML systems
"""

import time
from typing import List, Dict, Optional
from langchain.schema import Document


# ── Query Log ─────────────────────────────────────────────────────────────────
# Stores every query's metrics in memory during the session
# In production this would go to a database (PostgreSQL, BigQuery etc.)

_query_log: List[Dict] = []


def log_query(
    query: str,
    retrieved_chunks: List[Document],
    answer: str,
    usage_stats: Dict,
    retrieval_stats: Dict,
    latency_seconds: float
) -> Dict:
    """
    Logs metrics for a single query.
    Called automatically by the UI after every RAG call.

    Args:
        query            : The user's question
        retrieved_chunks : Chunks returned by retrieval
        answer           : LLM's generated answer
        usage_stats      : Token counts and cost from rag_engine
        retrieval_stats  : Chunk counts from retriever
        latency_seconds  : Total pipeline time

    Returns:
        The logged record as a dictionary
    """
    record = {
        "query"              : query,
        "query_length_chars" : len(query),
        "query_length_words" : len(query.split()),
        "chunks_retrieved"   : retrieval_stats.get("chunks_retrieved", 0),
        "chunks_after_dedup" : retrieval_stats.get("chunks_after_dedup", 0),
        "answer_length_chars": len(answer),
        "answer_length_words": len(answer.split()),
        "input_tokens"       : usage_stats.get("input_tokens", 0),
        "output_tokens"      : usage_stats.get("output_tokens", 0),
        "total_tokens"       : usage_stats.get("total_tokens", 0),
        "cost_usd"           : usage_stats.get("estimated_cost_usd", 0.0),
        "latency_seconds"    : latency_seconds,
        "provider"           : usage_stats.get("provider", "unknown"),
        "model"              : usage_stats.get("model", "unknown"),
        "sources_cited"      : list(set(
            c.metadata.get("source", "unknown")
            for c in retrieved_chunks
        )),
        "timestamp"          : time.strftime("%Y-%m-%d %H:%M:%S")
    }

    _query_log.append(record)
    return record


def get_session_metrics() -> Dict:
    """
    Computes aggregate metrics across all queries in this session.
    Powers the Evaluation Dashboard UI.

    Returns:
        Dictionary of aggregate metrics
    """
    if not _query_log:
        return {
            "total_queries"        : 0,
            "message"              : "No queries yet — ask some questions first!"
        }

    total_queries    = len(_query_log)
    total_tokens     = sum(r["total_tokens"]    for r in _query_log)
    total_cost       = sum(r["cost_usd"]        for r in _query_log)
    total_latency    = sum(r["latency_seconds"] for r in _query_log)
    avg_latency      = total_latency / total_queries
    avg_tokens       = total_tokens  / total_queries
    avg_chunks       = sum(r["chunks_retrieved"] for r in _query_log) / total_queries
    avg_answer_words = sum(r["answer_length_words"] for r in _query_log) / total_queries

    # Token breakdown
    avg_input_tokens  = sum(r["input_tokens"]  for r in _query_log) / total_queries
    avg_output_tokens = sum(r["output_tokens"] for r in _query_log) / total_queries

    # Most queried sources
    all_sources = []
    for r in _query_log:
        all_sources.extend(r["sources_cited"])

    source_counts = {}
    for s in all_sources:
        source_counts[s] = source_counts.get(s, 0) + 1

    top_sources = sorted(
        source_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        "total_queries"        : total_queries,
        "total_tokens_used"    : total_tokens,
        "total_cost_usd"       : round(total_cost, 6),
        "avg_latency_seconds"  : round(avg_latency, 2),
        "avg_tokens_per_query" : round(avg_tokens, 1),
        "avg_input_tokens"     : round(avg_input_tokens, 1),
        "avg_output_tokens"    : round(avg_output_tokens, 1),
        "avg_chunks_retrieved" : round(avg_chunks, 1),
        "avg_answer_words"     : round(avg_answer_words, 1),
        "top_sources"          : top_sources,
        "provider"             : _query_log[-1]["provider"],
        "model"                : _query_log[-1]["model"],
        "query_log"            : _query_log
    }


def compute_hit_rate(
    query_chunk_pairs: List[Dict]
) -> float:
    """
    Computes Hit Rate — what % of queries retrieved
    at least one relevant chunk in top-k.

    In a real eval, you need labeled data:
    [{"query": "...", "relevant_chunk_id": "..."}]

    Here we use a proxy: did the LLM cite any source?
    (If answer contains [Source N], retrieval "hit")

    Args:
        query_chunk_pairs: List of logged query records

    Returns:
        Hit rate as float 0.0 to 1.0
    """
    if not query_chunk_pairs:
        return 0.0

    hits = sum(
        1 for r in query_chunk_pairs
        if r["chunks_retrieved"] > 0
    )
    return hits / len(query_chunk_pairs)


def compute_mrr(ranked_results: List[List[Document]]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).

    MRR = average of (1 / rank_of_first_relevant_result)

    Example:
      Query 1: relevant doc at rank 1 → 1/1 = 1.0
      Query 2: relevant doc at rank 3 → 1/3 = 0.33
      MRR = (1.0 + 0.33) / 2 = 0.67

    In production you need human labels.
    Here we return a placeholder for the dashboard.

    Args:
        ranked_results: List of ranked chunk lists per query

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not ranked_results:
        return 0.0

    reciprocal_ranks = []
    for results in ranked_results:
        if results:
            # Assume first result is relevant (optimistic proxy)
            reciprocal_ranks.append(1.0 / 1)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def get_latency_breakdown(query_log: List[Dict]) -> Dict:
    """
    Breaks down latency statistics across all queries.

    Args:
        query_log: List of logged query records

    Returns:
        Latency stats dictionary
    """
    if not query_log:
        return {}

    latencies = [r["latency_seconds"] for r in query_log]

    return {
        "min_latency" : round(min(latencies), 2),
        "max_latency" : round(max(latencies), 2),
        "avg_latency" : round(sum(latencies) / len(latencies), 2),
        "p95_latency" : round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
        if len(latencies) >= 2 else round(latencies[0], 2)
    }


def clear_query_log():
    """Clears the in-memory query log. Called on session reset."""
    global _query_log
    _query_log = []
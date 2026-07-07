"""
RAG Engine Module
------------------
The heart of the system. Orchestrates the full
Retrieval-Augmented Generation pipeline end to end.

Full flow:
1. Receive user question + chat history
2. (Optional) Rewrite question for better retrieval
3. Retrieve relevant chunks from ChromaDB
4. Build LLM prompt with context + history
5. Call LLM (Groq/OpenAI) to generate answer
6. Return answer + citations + stats

This is what the Streamlit UI (Phase 8) will call.
"""

import os
import time
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from src.retrieval.retriever import run_retrieval_pipeline
from src.generation.prompt_builder import (
    build_rag_prompt,
    build_standalone_question_prompt,
    build_prompt_stats
)

load_dotenv()


# ── LLM Client Setup ──────────────────────────────────────────────────────────

def get_llm_client():
    """
    Returns a configured LLM client.
    Tries Groq first (free), falls back to OpenAI.

    Returns:
        Tuple of (client, model_name, provider_name)
    """
    groq_key   = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        return client, "llama-3.3-70b-versatile", "groq"

    elif openai_key:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        return client, "gpt-3.5-turbo", "openai"

    else:
        raise ValueError(
            "No LLM API key found. Add GROQ_API_KEY or "
            "OPENAI_API_KEY to your .env file."
        )


def call_llm(
    messages: List[Dict],
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> Tuple[str, Dict]:
    """
    Sends messages to the LLM and returns the response.

    WHY temperature=0.1:
    Lower temperature = more deterministic, factual output.
    For RAG over documents, we want consistency not creativity.
    Range: 0.0 (robotic/deterministic) to 1.0 (creative/random)

    Args:
        messages    : Message list from build_rag_prompt()
        temperature : Creativity level (0.0–1.0)
        max_tokens  : Max response length

    Returns:
        Tuple of (answer_text, usage_stats)
    """
    client, model, provider = get_llm_client()

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    end_time  = time.time()
    latency   = round(end_time - start_time, 2)

    answer    = response.choices[0].message.content

    # Usage stats for token tracking (Feature #18)
    usage_stats = {
        "provider"       : provider,
        "model"          : model,
        "input_tokens"   : response.usage.prompt_tokens,
        "output_tokens"  : response.usage.completion_tokens,
        "total_tokens"   : response.usage.total_tokens,
        "latency_seconds": latency
    }

    return answer, usage_stats


# ── Query Rewriting ───────────────────────────────────────────────────────────

def rewrite_query(
    question: str,
    chat_history: List[Dict]
) -> str:
    """
    Rewrites a follow-up question into a standalone question.
    Feature #13 — Query Rewriting.

    WHY: "What about the pricing?" is a terrible retrieval query
    if the user was previously asking about subscriptions.
    We rewrite it to "What is the pricing for subscriptions?"
    before sending to ChromaDB.

    Args:
        question     : The raw follow-up question
        chat_history : Previous conversation turns

    Returns:
        A rewritten standalone question string
    """
    # Only rewrite if there's actual history to reference
    if not chat_history or len(chat_history) < 2:
        return question

    prompt = build_standalone_question_prompt(question, chat_history)

    try:
        messages = [{"role": "user", "content": prompt}]
        rewritten, _ = call_llm(messages, temperature=0.0, max_tokens=150)
        return rewritten.strip()
    except Exception:
        # If rewriting fails, just use the original question
        return question


# ── Main RAG Function ─────────────────────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    collection_name: str = "documents",
    chat_history: List[Dict] = None,
    k: int = 4,
    source_filter: Optional[str] = None,
    rewrite_query_flag: bool = True,
    temperature: float = 0.1,
    use_hybrid: bool = False,
    use_reranking: bool = False,
    all_chunks: List = None
) -> Dict:
    """
    The complete RAG pipeline — from question to answer.
    This is the single function the UI calls.

    Args:
        question           : User's question
        collection_name    : ChromaDB collection to search
        chat_history       : Previous conversation turns
        k                  : Number of chunks to retrieve
        source_filter      : Optional filename filter
        rewrite_query_flag : Whether to rewrite follow-up questions
        temperature        : LLM creativity (0.0–1.0)

    Returns:
        Dictionary with everything the UI needs:
        {
            "answer"         : The LLM's answer string,
            "citations"      : List of source citations,
            "context"        : The raw context sent to LLM,
            "retrieval_stats": Stats from retrieval,
            "usage_stats"    : Token usage and cost,
            "rewritten_query": The (possibly rewritten) query,
            "chunks"         : The raw retrieved Document objects
        }
    """
    if chat_history is None:
        chat_history = []

    pipeline_start = time.time()

    # ── Step 1: Query Rewriting ───────────────────────────────
    if rewrite_query_flag and chat_history:
        retrieval_query = rewrite_query(question, chat_history)
    else:
        retrieval_query = question
    # ── Step 2: Retrieval ─────────────────────────────────────
    if use_hybrid and all_chunks:
        # Hybrid search: BM25 + Vector
        from src.retrieval.hybrid_retriever import hybrid_search
        chunks = hybrid_search(
            query=retrieval_query,
            chunks=all_chunks,
            collection_name=collection_name,
            k=k
        )
        # Format manually since we bypassed run_retrieval_pipeline
        from src.retrieval.retriever import (
            deduplicate_chunks,
            format_chunks_as_context,
            build_citations
        )
        chunks    = deduplicate_chunks(chunks)
        context   = format_chunks_as_context(chunks)
        citations = build_citations(chunks)
        retrieval_result = {
            "chunks"   : chunks,
            "context"  : context,
            "citations": citations,
            "stats"    : {
                "query"              : retrieval_query,
                "chunks_retrieved"   : len(chunks),
                "chunks_after_dedup" : len(chunks),
                "duplicates_removed" : 0,
                "source_filter"      : source_filter,
                "collection"         : collection_name
            }
        }
    else:
        # Standard vector-only retrieval
        retrieval_result = run_retrieval_pipeline(
            query=retrieval_query,
            collection_name=collection_name,
            k=k,
            source_filter=source_filter,
            deduplicate=True
        )

    chunks    = retrieval_result["chunks"]
    context   = retrieval_result["context"]
    citations = retrieval_result["citations"]

    # ── Step 2b: Reranking (optional) ─────────────────────────
    if use_reranking and chunks:
        from src.retrieval.reranker import rerank_and_return_docs
        from src.retrieval.retriever import format_chunks_as_context, build_citations
        chunks    = rerank_and_return_docs(question, chunks, top_n=k)
        context   = format_chunks_as_context(chunks)
        citations = build_citations(chunks)

    # ── Step 3: Build Prompt ──────────────────────────────────
    messages = build_rag_prompt(
        question=question,    # original question (not rewritten)
        context=context,
        chat_history=chat_history
    )

    prompt_stats = build_prompt_stats(messages)

    # ── Step 4: Generate Answer ───────────────────────────────
    answer, usage_stats = call_llm(
        messages=messages,
        temperature=temperature
    )

    # ── Step 5: Calculate Cost ────────────────────────────────
    # Groq is free. OpenAI costs approx:
    # gpt-3.5-turbo: $0.0005/1K input, $0.0015/1K output
    if usage_stats["provider"] == "openai":
        cost = (
            (usage_stats["input_tokens"]  / 1000) * 0.0005 +
            (usage_stats["output_tokens"] / 1000) * 0.0015
        )
    else:
        cost = 0.0  # Groq is free

    usage_stats["estimated_cost_usd"] = round(cost, 6)

    pipeline_end     = time.time()
    total_latency    = round(pipeline_end - pipeline_start, 2)
    usage_stats["total_pipeline_latency_seconds"] = total_latency

    return {
        "answer"          : answer,
        "citations"       : citations,
        "context"         : context,
        "retrieval_stats" : retrieval_result["stats"],
        "usage_stats"     : usage_stats,
        "rewritten_query" : retrieval_query,
        "chunks"          : chunks,
        "prompt_stats"    : prompt_stats
    }
"""
Evaluation Dashboard
---------------------
A separate Streamlit page showing RAG system metrics.
Feature #16 — Evaluation Dashboard

Run alongside main app — Streamlit handles multiple pages
automatically when files are in the pages/ folder.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.evaluation.evaluator import (
    get_session_metrics,
    get_latency_breakdown,
    compute_hit_rate
)

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("# 📊 Evaluation Dashboard")
st.markdown("Real-time metrics for your RAG pipeline.")
st.markdown("---")

# Get metrics
metrics = get_session_metrics()

if metrics["total_queries"] == 0:
    st.info(
        "📭 No queries logged yet. "
        "Go to the main page, ask some questions, then come back here."
    )
else:
    # ── Row 1: Key Metrics ────────────────────────────────────
    st.markdown("### 🎯 Session Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Queries",    metrics["total_queries"])
    c2.metric("Total Tokens",     f"{metrics['total_tokens_used']:,}")
    c3.metric("Total Cost",       f"${metrics['total_cost_usd']:.4f}")
    c4.metric("Avg Latency",      f"{metrics['avg_latency_seconds']}s")
    c5.metric("Avg Answer Words", metrics["avg_answer_words"])

    st.markdown("---")

    # ── Row 2: Token Breakdown ────────────────────────────────
    st.markdown("### 🪙 Token Usage Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Per Query Average**")
        st.metric("Input tokens",  metrics["avg_input_tokens"])
        st.metric("Output tokens", metrics["avg_output_tokens"])
        st.metric("Total tokens",  metrics["avg_tokens_per_query"])

    with col2:
        st.markdown("**Model Info**")
        st.info(
            f"**Provider:** {metrics['provider']}\n\n"
            f"**Model:** {metrics['model']}"
        )

    with col3:
        st.markdown("**Cost Projection**")
        cost_per_query = (
            metrics["total_cost_usd"] / metrics["total_queries"]
            if metrics["total_queries"] > 0 else 0
        )
        st.metric("Cost per query",    f"${cost_per_query:.6f}")
        st.metric("100 queries cost",  f"${cost_per_query * 100:.4f}")
        st.metric("1000 queries cost", f"${cost_per_query * 1000:.4f}")

    st.markdown("---")

    # ── Row 3: Retrieval Metrics ──────────────────────────────
    st.markdown("### 🔍 Retrieval Metrics")
    col1, col2 = st.columns(2)

    query_log = metrics.get("query_log", [])

    with col1:
        hit_rate = compute_hit_rate(query_log)
        st.metric(
            "Hit Rate",
            f"{hit_rate * 100:.1f}%",
            help="% of queries where at least 1 chunk was retrieved"
        )
        st.metric(
            "Avg Chunks Retrieved",
            metrics["avg_chunks_retrieved"],
            help="Average number of chunks returned per query"
        )

    with col2:
        latency_stats = get_latency_breakdown(query_log)
        if latency_stats:
            st.metric("Min Latency", f"{latency_stats['min_latency']}s")
            st.metric("Max Latency", f"{latency_stats['max_latency']}s")
            st.metric("P95 Latency", f"{latency_stats['p95_latency']}s")

    st.markdown("---")

    # ── Row 4: Top Sources ────────────────────────────────────
    st.markdown("### 📄 Most Queried Sources")
    if metrics["top_sources"]:
        for source, count in metrics["top_sources"]:
            st.progress(
                count / metrics["total_queries"],
                text=f"{source} — cited in {count} queries"
            )
    else:
        st.write("No source data yet.")

    st.markdown("---")

    # ── Row 5: Query Log Table ────────────────────────────────
    st.markdown("### 📋 Query Log")

    if query_log:
        import pandas as pd

        df = pd.DataFrame([{
            "Query"    : r["query"][:60] + "..." if len(r["query"]) > 60 else r["query"],
            "Chunks"   : r["chunks_retrieved"],
            "Tokens"   : r["total_tokens"],
            "Cost $"   : f"{r['cost_usd']:.6f}",
            "Latency s": r["latency_seconds"],
            "Time"     : r["timestamp"]
        } for r in query_log])

        st.dataframe(df, use_container_width=True)

        # Download query log
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Query Log (CSV)",
            data=csv,
            file_name="query_log.csv",
            mime="text/csv"
        )
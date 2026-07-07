"""
Evaluation Dashboard Page
--------------------------
Streamlit multi-page routing file.
Simply imports and runs the evaluation dashboard.
"""
import sys
import os

# Go up TWO levels: pages/ → app/ → project_root/
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
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

metrics = get_session_metrics()

if metrics["total_queries"] == 0:
    st.info(
        "📭 No queries logged yet.\n\n"
        "Go to the **main page**, ask some questions, "
        "then come back here to see metrics."
    )

else:
    # ── Overview ──────────────────────────────────────────────
    st.markdown("### 🎯 Session Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Queries",          metrics["total_queries"])
    c2.metric("Total Tokens",     f"{metrics['total_tokens_used']:,}")
    c3.metric("Total Cost",       f"${metrics['total_cost_usd']:.4f}")
    c4.metric("Avg Latency",      f"{metrics['avg_latency_seconds']}s")
    c5.metric("Avg Answer Words", metrics["avg_answer_words"])

    st.markdown("---")

    # ── Token Breakdown ───────────────────────────────────────
    st.markdown("### 🪙 Token Usage")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Per Query**")
        st.metric("Input tokens",  metrics["avg_input_tokens"])
        st.metric("Output tokens", metrics["avg_output_tokens"])

    with col2:
        st.markdown("**Model**")
        st.info(
            f"**Provider:** {metrics['provider']}\n\n"
            f"**Model:** {metrics['model']}"
        )

    with col3:
        cost_per_query = (
            metrics["total_cost_usd"] / metrics["total_queries"]
        )
        st.markdown("**Cost Projection**")
        st.metric("Per query",     f"${cost_per_query:.6f}")
        st.metric("Per 1K queries",f"${cost_per_query * 1000:.4f}")

    st.markdown("---")

    # ── Retrieval Metrics ─────────────────────────────────────
    st.markdown("### 🔍 Retrieval Metrics")
    query_log = metrics.get("query_log", [])
    col1, col2 = st.columns(2)

    with col1:
        hit_rate = compute_hit_rate(query_log)
        st.metric("Hit Rate",
            f"{hit_rate * 100:.1f}%",
            help="% queries with at least 1 chunk retrieved"
        )
        st.metric("Avg Chunks Retrieved",
            metrics["avg_chunks_retrieved"]
        )

    with col2:
        latency = get_latency_breakdown(query_log)
        if latency:
            st.metric("Min Latency", f"{latency['min_latency']}s")
            st.metric("Max Latency", f"{latency['max_latency']}s")
            st.metric("P95 Latency", f"{latency['p95_latency']}s")

    st.markdown("---")

    # ── Top Sources ───────────────────────────────────────────
    st.markdown("### 📄 Most Queried Sources")
    for source, count in metrics["top_sources"]:
        st.progress(
            count / metrics["total_queries"],
            text=f"{source} — {count} queries"
        )

    st.markdown("---")

    # ── Query Log Table ───────────────────────────────────────
    st.markdown("### 📋 Full Query Log")
    df = pd.DataFrame([{
        "Query"    : r["query"][:60] + "..." if len(r["query"]) > 60 else r["query"],
        "Chunks"   : r["chunks_retrieved"],
        "Tokens"   : r["total_tokens"],
        "Cost $"   : f"{r['cost_usd']:.6f}",
        "Latency s": r["latency_seconds"],
        "Time"     : r["timestamp"]
    } for r in query_log])

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Query Log (CSV)",
        data=csv,
        file_name="query_log.csv",
        mime="text/csv"
    )
"""
Document Intelligence Platform — Streamlit UI
----------------------------------------------
The main application file. Brings together all phases:
- PDF Upload + Ingestion  (Phases 2, 3, 4, 5)
- Retrieval               (Phase 6)
- RAG Generation          (Phase 7)
- Chat Interface          (Phase 8)

Run with: streamlit run app/main.py
"""

import os
import sys
import time

# Add project root to Python path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.ingestion.text_chunker import chunk_documents
from src.retrieval.vector_store import (
    create_vector_store_from_chunks,
    get_vector_store,
    add_chunks_to_vector_store,
    get_collection_stats
)
from src.generation.rag_engine import run_rag_pipeline
from src.utils.helpers import (
    save_uploaded_file,
    get_file_hash,
    export_chat_history,
    format_file_size
)

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Intelligence Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 2rem;
    }

    /* Chat messages */
    .user-message {
        background: #1A1D27;
        border-left: 3px solid #6C63FF;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .assistant-message {
        background: #1A1D27;
        border-left: 3px solid #00D4AA;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }

    /* Citation cards */
    .citation-card {
        background: #252836;
        border: 1px solid #3D3F4E;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.8rem;
    }

    /* Stats cards */
    .stat-card {
        background: #1A1D27;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #2D2F3E;
    }

    /* Upload area */
    .upload-success {
        color: #00D4AA;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────
# Streamlit re-runs the entire script on every interaction.
# st.session_state persists data across re-runs.
def _full_reset():
    """
    Resets the app using ChromaDB's internal API to delete
    the collection — avoids Windows file lock issues entirely.
    """
    import gc
    import time

    # Step 1: Delete collection using ChromaDB's own API
    # This works even when Windows locks the SQLite file
    try:
        import chromadb
        gc.collect()
        time.sleep(0.3)

        client = chromadb.PersistentClient(path="chroma_db")
        existing = [c.name for c in client.list_collections()]

        if "documents" in existing:
            client.delete_collection("documents")
            print("✅ Collection deleted via ChromaDB API")

        del client
        gc.collect()

    except Exception as e:
        print(f"ChromaDB reset error: {e}")

    # Step 2: Try folder delete as bonus cleanup
    import shutil
    time.sleep(0.5)
    gc.collect()
    if os.path.exists("chroma_db"):
        try:
            shutil.rmtree("chroma_db")
        except Exception:
            pass  # OK — collection already deleted above

    # Step 3: Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
# Clear evaluator log (lazy import)
    try:
        from src.evaluation.evaluator import clear_query_log
        clear_query_log()
    except Exception:
        pass

    st.rerun()
    
def init_session_state():
    """Initialise all session state variables."""
    defaults = {
        "chat_history"       : [],
        "processed_files"    : [],
        "processed_hashes"   : [],
        "vector_store_ready" : False,
        "total_chunks"       : 0,
        "total_tokens_used"  : 0,
        "total_cost_usd"     : 0.0,
        "query_count"        : 0,
        "current_collection" : "documents"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Auto-detect existing ChromaDB data on startup
    if not st.session_state.vector_store_ready:
        try:
            import chromadb
            client   = chromadb.PersistentClient(path="chroma_db")
            existing = [c.name for c in client.list_collections()]

            if "documents" in existing:
                col   = client.get_collection("documents")
                count = col.count()
                if count > 0:
                    st.session_state.vector_store_ready = True
                    st.session_state.total_chunks       = count
                    st.session_state["restored_from_disk"] = True

            del client

        except Exception:
            pass
init_session_state()
# Show notice if we restored data from a previous session
if st.session_state.get("restored_from_disk"):
    st.toast(
        f"📂 Restored previous session — "
        f"{st.session_state.total_chunks} chunks loaded from disk",
        icon="✅"
    )
    st.session_state["restored_from_disk"] = False
# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 Document Intelligence")
    st.markdown("---")

    # ── File Upload Section ───────────────────────────────────
    st.markdown("### 📁 Upload Documents")

    uploaded_files = st.file_uploader(
        label="Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to query against"
    )

    if uploaded_files:
        if st.button("🚀 Process Documents", use_container_width=True):
            new_files_processed = 0

            for uploaded_file in uploaded_files:
                # Save to disk
                file_path = save_uploaded_file(uploaded_file)
                file_hash = get_file_hash(file_path)

                # Skip duplicates
                if file_hash in st.session_state.processed_hashes:
                    st.warning(f"⏭️ '{uploaded_file.name}' already processed")
                    continue

                with st.spinner(f"Processing '{uploaded_file.name}'..."):
                    try:
                        # Full ingestion pipeline
                        docs   = extract_text_from_pdf(file_path)
                        chunks = chunk_documents(
                            docs,
                            chunk_size=1000,
                            chunk_overlap=200
                        )

                        if not st.session_state.vector_store_ready:
                            # First file — create new vector store
                            vs = create_vector_store_from_chunks(
                                chunks=chunks,
                                collection_name=st.session_state.current_collection
                            )
                            st.session_state.vector_store_ready = True
                        else:
                            # Additional files — add to existing store
                            vs = get_vector_store(
                                collection_name=st.session_state.current_collection
                            )
                            add_chunks_to_vector_store(vs, chunks)

                        # Update session state
                        st.session_state.processed_files.append(uploaded_file.name)
                        st.session_state.processed_hashes.append(file_hash)
                        st.session_state.total_chunks += len(chunks)

                        new_files_processed += 1
                        st.success(
                            f"✅ '{uploaded_file.name}' — "
                            f"{len(docs)} pages, {len(chunks)} chunks"
                        )

                    except Exception as e:
                        st.error(f"❌ Failed to process '{uploaded_file.name}': {e}")

            if new_files_processed > 0:
                st.success(f"🎉 {new_files_processed} file(s) ready to query!")
                st.rerun()

    st.markdown("---")

    # ── Processed Files ───────────────────────────────────────
    if st.session_state.processed_files:
        st.markdown("### 📚 Loaded Documents")
        for fname in st.session_state.processed_files:
            st.markdown(f"✅ {fname}")
        st.markdown("---")

    # ── Stats Section ─────────────────────────────────────────
    if st.session_state.vector_store_ready:
        st.markdown("### 📊 Session Statistics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", st.session_state.total_chunks)
            st.metric("Queries", st.session_state.query_count)
        with col2:
            st.metric("Tokens", f"{st.session_state.total_tokens_used:,}")
            st.metric(
                "Cost",
                f"${st.session_state.total_cost_usd:.4f}"
            )
        st.markdown("---")

    # ── Settings Section ──────────────────────────────────────
    st.markdown("### ⚙️ Settings")

    num_chunks = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=10,
        value=4,
        help="More chunks = more context but slower and costlier"
    )

    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="0 = factual/deterministic, 1 = creative/random"
    )

    show_citations = st.toggle("Show source citations", value=True)
    show_context   = st.toggle("Show retrieved context", value=False)
    show_stats     = st.toggle("Show query stats", value=False)

    st.markdown("---")

    # ── Export + Reset ────────────────────────────────────────
    # ── Export & Reset ────────────────────────────────────────
    st.markdown("---")

    if st.session_state.chat_history:
        export_text = export_chat_history(
            st.session_state.chat_history, format="md"
        )
        st.download_button(
            label="💾 Export Conversation",
            data=export_text,
            file_name="conversation.md",
            mime="text/markdown",
            use_container_width=True
        )

    # Clear chat only (keeps documents loaded)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Full reset — clears everything including ChromaDB
    st.markdown("---")
    st.markdown("**⚠️ Danger Zone**")
    if st.button("🔄 New Session (Reset All)", use_container_width=True):
        _full_reset()

# ── Main Chat Area ────────────────────────────────────────────────────────────

# Header
st.markdown(
    '<p class="main-header">📄 Document Intelligence Platform</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Upload PDFs → Ask questions → Get answers with citations</p>',
    unsafe_allow_html=True
)

# Welcome message when no documents loaded
if not st.session_state.vector_store_ready:
    st.info(
        "👈 **Get started:** Upload one or more PDF files in the sidebar, "
        "then click **Process Documents**."
    )

    # Feature overview
    with st.expander("✨ What can this platform do?"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🔍 Smart Search**")
            st.markdown("Semantic search finds relevant content even when exact words differ")
        with col2:
            st.markdown("**📎 Source Citations**")
            st.markdown("Every answer links back to the exact page and document it came from")
        with col3:
            st.markdown("**💬 Chat Memory**")
            st.markdown("Ask follow-up questions — the platform remembers your conversation")

else:
    # ── Chat History Display ──────────────────────────────────
    chat_container = st.container()

    with chat_container:
        # Show all previous messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show citations if this is an assistant message
                # and citations are stored
                if (message["role"] == "assistant"
                        and "citations" in message
                        and show_citations):
                    with st.expander(
                        f"📎 {len(message['citations'])} Source(s)",
                        expanded=False
                    ):
                        for c in message["citations"]:
                            st.markdown(
                                f"""<div class="citation-card">
                                <b>[{c['citation_number']}]</b>
                                {c['source']} — Page {c['page_number']}<br>
                                <small>{c['preview']}</small>
                                </div>""",
                                unsafe_allow_html=True
                            )

    # ── Chat Input ────────────────────────────────────────────
    if question := st.chat_input(
        "Ask a question about your documents...",
        disabled=not st.session_state.vector_store_ready
    ):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Add to history
        st.session_state.chat_history.append({
            "role"   : "user",
            "content": question
        })

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating answer..."):

                # Build history for LLM (exclude current question)
                llm_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]
                ]

                # Run full RAG pipeline
                result = run_rag_pipeline(
                    question=question,
                    collection_name=st.session_state.current_collection,
                    chat_history=llm_history,
                    k=num_chunks,
                    temperature=temperature
                )

            # Display answer
            st.markdown(result["answer"])

            # Citations
            if show_citations and result["citations"]:
                with st.expander(
                    f"📎 {len(result['citations'])} Source(s)",
                    expanded=True
                ):
                    for c in result["citations"]:
                        st.markdown(
                            f"""<div class="citation-card">
                            <b>[{c['citation_number']}]</b>
                            {c['source']} — Page {c['page_number']}<br>
                            <small>{c['preview']}</small>
                            </div>""",
                            unsafe_allow_html=True
                        )

            # Retrieved context (debug view)
            if show_context:
                with st.expander("🔎 Retrieved Context", expanded=False):
                    st.text(result["context"])

            # Query stats
            if show_stats:
                with st.expander("📊 Query Statistics", expanded=False):
                    stats = result["usage_stats"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Provider",  stats["provider"])
                    c2.metric("Tokens",    stats["total_tokens"])
                    c3.metric("Latency",   f"{stats['latency_seconds']}s")
                    c4.metric("Cost",      f"${stats['estimated_cost_usd']}")

                    r_stats = result["retrieval_stats"]
                    st.markdown(
                        f"**Chunks retrieved:** {r_stats['chunks_retrieved']} → "
                        f"**After dedup:** {r_stats['chunks_after_dedup']}"
                    )
                    if r_stats["query"] != question:
                        st.markdown(
                            f"**Rewritten query:** {r_stats['query']}"
                        )

        # Store assistant message WITH citations
        st.session_state.chat_history.append({
            "role"     : "assistant",
            "content"  : result["answer"],
            "citations": result["citations"]
        })

        # Update session stats
        st.session_state.total_tokens_used += result["usage_stats"]["total_tokens"]
        st.session_state.total_cost_usd    += result["usage_stats"]["estimated_cost_usd"]
        st.session_state.query_count       += 1

        # Log to evaluator (lazy import avoids path issues)
        try:
            from src.evaluation.evaluator import log_query
            log_query(
                query=question,
                retrieved_chunks=result["chunks"],
                answer=result["answer"],
                usage_stats=result["usage_stats"],
                retrieval_stats=result["retrieval_stats"],
                latency_seconds=result["usage_stats"]["total_pipeline_latency_seconds"]
            )
        except Exception:
            pass
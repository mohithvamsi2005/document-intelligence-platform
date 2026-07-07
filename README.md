# 📄 Enterprise Document Intelligence Platform

A production-quality RAG (Retrieval-Augmented Generation) system
that lets you upload PDFs and ask questions with source citations.

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone the repo
git clone https://github.com/yourusername/document-intelligence-platform

# Add your API keys
cp .env.example .env
# Edit .env with your keys

# Run
docker-compose up
```
Access at: http://localhost:8501

### Option 2: Local Python
```bash
# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add API keys to .env
# Run
streamlit run app/main.py
```

## 🔑 Environment Variables

Create a `.env` file:
```
GROQ_API_KEY=gsk_your_key_here      # Free at console.groq.com
OPENAI_API_KEY=sk_your_key_here     # Optional fallback
```

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                           ↓
User Query → Query Rewriting → Hybrid Search → Reranking → LLM → Answer + Citations
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | sentence-transformers (free, local) |
| Vector DB | ChromaDB |
| LLM | Groq Llama 3.3 (free) |
| Search | BM25 + Vector (Hybrid) |
| Reranking | Cross-Encoder |

## ✨ Features

- Upload multiple PDFs
- Semantic + keyword hybrid search
- Source citations with page numbers
- Chat memory across turns
- Query rewriting for follow-ups
- Evaluation dashboard with metrics
- Token usage and cost tracking
- Conversation export

## 📊 Performance

- Embedding: ~384 dims, fully local, free
- LLM: Groq Llama 3.3 70B, free tier
- Latency: 3-8 seconds per query
- Cost: $0.00 (fully free stack)
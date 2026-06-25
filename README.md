# 🔍 RAG Pipeline with LangChain + FAISS

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, FAISS vector store, and OpenAI / Anthropic Claude.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange?style=flat-square)

## Features
- 📄 Load and chunk PDF, TXT, and DOCX documents
- 🧠 Generate embeddings with OpenAI or HuggingFace models
- 🔍 FAISS vector store for fast semantic retrieval
- 🤖 LangChain conversational chain with memory
- 💾 Persist and reload vector indexes

## Architecture
```
Documents → Chunking → Embeddings → FAISS Index
                                         ↓
User Query → Embed Query → Similarity Search → Top-K Chunks → LLM → Answer
```

## Quick Start
```bash
git clone https://github.com/OsamaBinShah1/rag-langchain-faiss.git
cd rag-langchain-faiss
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python main.py
```

## Usage
```python
from rag.chain import RAGChain

rag = RAGChain(docs_path="./docs", model="gpt-4o-mini")
answer = rag.query("What are the main points in this document?")
print(answer)
```

## Stack
- **LangChain** — chain orchestration & memory
- **FAISS** — vector similarity search
- **OpenAI GPT-4 / HuggingFace** — LLM & embeddings
- **FastAPI** — REST API serving

## Author
**Muhammad Osama Bin Shah** — AI Engineer, Frankfurt, Germany
[LinkedIn](https://www.linkedin.com/in/muhammad-osama-bin-shah/)

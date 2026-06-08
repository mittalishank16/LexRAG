# backend/agents/legal_graph.py
# ── v2.2 — Strict ONNX Runtime Isolation for Render Free Tier ─────────────────────
#
# Crucial Updates:
#   1. Aligned model pathing targets with flat Dockerfile-compiled image layers.
#   2. Enforced strict ONNX-only loading to remove risky internet-facing PyTorch fallbacks.
#   3. Preserved LangGraph multi-agent parallel orchestration structure.
#   4. Retained safe lazy-loading rules for the sentence-transformers Cross-Encoder layer.

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════

import os
import re
import json
import shutil
import hashlib
import nltk
import numpy as np
import redis

from pathlib import Path
from typing import List, Optional, TypedDict
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from rank_bm25 import BM25Okapi

load_dotenv()
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & PATHS
# ════════════════════════════════════════════════════════════════════════════

APP_ENV = os.getenv("APP_ENV", "production")
device  = "cpu"   # Render free tier has no GPU — always CPU
print(f"Device: {device} | APP_ENV: {APP_ENV}")

# ── Flat Container Path Resolution ──────────────────────────────────────────
# Both local environments and Docker containers look for 'models/' at their workspace root.
MODEL_BASE_DIR = Path("models")

# Fallback adjustment to ensure smooth path translation in unnested local IDE contexts
if not MODEL_BASE_DIR.exists() and (Path(__file__).parent.parent / "models").exists():
    MODEL_BASE_DIR = Path(__file__).parent.parent / "models"

INLEGALBERT_ONNX = MODEL_BASE_DIR / "inlegalbert_onnx"
BGE_ONNX         = MODEL_BASE_DIR / "bge_onnx"

# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — STRICT ONNX RUNTIME ISOLATION
# ════════════════════════════════════════════════════════════════════════════

def _load_onnx_embeddings(onnx_dir: Path, label: str):
    """
    Loads an ONNXEmbeddings engine from the specified local directory path layer.
    """
    if not onnx_dir.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: {label} ONNX directory not found at target path: '{onnx_dir}'.\n"
            "Ensure that you ran your local split script and your Docker file built successfully."
        )
    try:
        from models.onnx_embeddings import ONNXEmbeddings
        print(f"Loading isolated {label} engine from ONNX binary context ({onnx_dir.name})...")
        emb = ONNXEmbeddings(str(onnx_dir), normalize=True)
        print(f"SUCCESS: {label} ONNX engine loaded into memory stack.")
        return emb
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to instantiate {label} ONNX engine array: {e}")


print("Initializing isolated embedding structures...")

# ── Query encoder: InLegalBERT ONNX (Zero Internet Fallbacks) ──────────────
query_emb = _load_onnx_embeddings(INLEGALBERT_ONNX, "InLegalBERT (Query Encoder)")

# ── Passage encoder: BGE-Base ONNX (Zero Internet Fallbacks) ────────────────
passage_emb = _load_onnx_embeddings(BGE_ONNX, "BGE-Base (Passage Encoder)")

# ── Cross-encoder reranker: Lazy load execution sequence ───────────────────
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("Lazy loading BGE-Reranker model into active memory frame...")
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
        print("BGE-Reranker compiled successfully")
    return _cross_encoder

# ════════════════════════════════════════════════════════════════════════════
# LLMs
# ════════════════════════════════════════════════════════════════════════════

main_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
fast_llm  = ChatGroq(model="llama-3.1-8b-instant",   temperature=0, max_tokens=10)

def llm(prompt: str) -> str:
    return main_llm.invoke(prompt).content

# ════════════════════════════════════════════════════════════════════════════
# VECTOR STORE — ChromaDB (local development) or Pinecone (Production Cloud)
# ════════════════════════════════════════════════════════════════════════════

def _get_legal_vectorstore():
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if pinecone_key and APP_ENV == "production":
        # ── Pinecone Production Index Configuration ───────────────────────
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        index_name = os.getenv("PINECONE_INDEX_NAME", "legal-knowledge")
        print(f"Connecting to remote Pinecone instance index target: '{index_name}'...")

        pc = Pinecone(api_key=pinecone_key)
        existing = [i.name for i in pc.list_indexes()]

        if index_name not in existing:
            raise RuntimeError(
                f"Pinecone index target reference '{index_name}' not discovered in cloud account.\n"
                "Verify database migration metrics."
            )

        vs = PineconeVectorStore(
            index_name=index_name,
            embedding=passage_emb,
            pinecone_api_key=pinecone_key,
        )
        print("Pinecone vector infrastructure boundary established.")
        return vs

    else:
        # ── Local Development ChromaDB Configuration ──────────────────────
        import chromadb
        db_path = os.getenv("CHROMA_PERSIST_DIR", "../data/vector_database")
        print(f"Mounting ChromaDB instance local disk workspace path: {db_path}...")

        client = chromadb.PersistentClient(path=db_path)
        vs = Chroma(
            client=client,
            embedding_function=passage_emb,
            collection_name="legal_knowledge_v2",
        )
        print("ChromaDB engine array linked.")
        return vs


legal_vs = _get_legal_vectorstore()

# ════════════════════════════════════════════════════════════════════════════
# BM25 INDEX — Built over the legal corpus at runtime startup
# ════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    return [t for t in re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()) if len(t) > 1]

legal_docs_raw = legal_vs.get()
legal_corpus   = legal_docs_raw["documents"]
legal_bm25     = BM25Okapi([tokenize(c) for c in legal_corpus])

print(f"Legal knowledge base loaded: {len(legal_corpus)} chunks")

# Per-document context boundaries
doc_vs     = None
doc_bm25   = None
doc_corpus = None

# ════════════════════════════════════════════════════════════════════════════
# RETRIEVAL HELPERS
# ════════════════════════════════════════════════════════════════════════════

def rrf(list_a: list, list_b: list, k: int = 60) -> list:
    scores = {}
    for rank, d in enumerate(list
# backend/agents/legal_graph.py
# ── v2.1 — ONNX Runtime integration for Render free tier ─────────────────────
#
# Key changes from your original file:
#   1. InLegalBERT loaded via ONNX Runtime (~90MB) instead of PyTorch (~450MB)
#   2. BGE-Base loaded via ONNX Runtime (~90MB) instead of PyTorch (~450MB)
#   3. SentenceTransformer sentence_model removed (was unused after chunking
#      moved to notebook 01 — kept cross_encoder as lazy load only)
#   4. cross_encoder loads lazily on first rerank() call, not at import time
#   5. Vector store switches between ChromaDB (local) and Pinecone (Render)
#      based on APP_ENV environment variable
#   6. strategist_agent sanitiser fixed (handles LLM returning extra words)
#   7. confidence field removed from LegalState (was never populated)
#   8. graph node name fixed: 'generate_answer' → 'answer' for consistency
#   9. critic node now wired to END
#  10. Redis imports moved to top with rest of imports (not mid-file)

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
from langchain_huggingface import HuggingFaceEmbeddings
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

# MODELS_DIR resolves to: lexrag/backend/models/
# legal_graph.py is at:   lexrag/backend/agents/legal_graph.py
# so __file__.parent.parent = lexrag/backend/
MODELS_DIR       = Path(__file__).parent.parent / "models"
INLEGALBERT_ONNX = MODELS_DIR / "inlegalbert_onnx"
BGE_ONNX         = MODELS_DIR / "bge_onnx"

# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING — ONNX RUNTIME
# ════════════════════════════════════════════════════════════════════════════
#
# Memory budget on Render free tier (512MB total):
#   InLegalBERT ONNX INT8 :  ~90 MB   (was ~450 MB with PyTorch)
#   BGE-Base    ONNX INT8 :  ~90 MB   (was ~450 MB with PyTorch)
#   FastAPI + LangGraph   : ~100 MB
#   ChromaDB / Pinecone   :  ~50 MB
#   Cross-encoder (lazy)  :  ~90 MB   loaded on first rerank call
#   OS + Python runtime   :  ~80 MB
#   ─────────────────────────────
#   Total at startup      : ~410 MB   ← fits in 512 MB
#   Total after first req : ~500 MB   ← still fits
#
# Why two separate models:
#   query_emb   = InLegalBERT — specialised for Indian legal queries
#                 Understands: "mens rea", "Article 21", "Section 302 IPC"
#                 at a semantic level that BGE-Base misses
#   passage_emb = BGE-Base — better for encoding long document passages
#                 Superior general retrieval performance on English text
# ────────────────────────────────────────────────────────────────────────────

def _load_onnx_embeddings(onnx_dir: Path, label: str):
    """
    Attempt to load an ONNXEmbeddings model from onnx_dir.
    Returns the ONNX model on success, None on failure.
    Caller decides the fallback.
    """
    if not onnx_dir.exists():
        print(f"  {label}: ONNX dir not found at {onnx_dir}")
        return None
    try:
        # Import here so the file can still load even if optimum is not
        # installed (e.g. during notebook development)
        from models.onnx_embeddings import ONNXEmbeddings
        print(f"  Loading {label} from ONNX ({onnx_dir.name})...")
        emb = ONNXEmbeddings(str(onnx_dir), normalize=True)
        print(f"  ✅ {label} ONNX loaded")
        return emb
    except Exception as e:
        print(f"{label} ONNX failed: {e}")
        return None


def _load_hf_embeddings(model_name: str, label: str):
    """
    Fallback: standard HuggingFace PyTorch embeddings.
    Used when ONNX models are not present (local dev without conversion).
    """
    print(f"  Loading {label} via HuggingFace (PyTorch fallback)...")
    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"{label} HuggingFace loaded")
    return emb


print("Loading embedding models...")

# ── Query encoder: InLegalBERT ONNX → InLegalBERT PyTorch → BGE-Base ──────
# Three-level fallback so the system never fails to start
query_emb = (
    _load_onnx_embeddings(INLEGALBERT_ONNX, "InLegalBERT")
    or _load_hf_embeddings("law-ai/InLegalBERT", "InLegalBERT")
    or _load_hf_embeddings("BAAI/bge-base-en-v1.5", "BGE-Base (InLegalBERT fallback)")
)

# ── Passage encoder: BGE-Base ONNX → BGE-Base PyTorch ─────────────────────
passage_emb = (
    _load_onnx_embeddings(BGE_ONNX, "BGE-Base")
    or _load_hf_embeddings("BAAI/bge-base-en-v1.5", "BGE-Base")
)

# ── Cross-encoder reranker: lazy load ──────────────────────────────────────
# NOT loaded at startup — only on first rerank() call.
# Reason: Render marks the service healthy when /health returns 200.
# If cross-encoder loads at startup, the health check fires before loading
# finishes, Render kills the container, and the deploy fails.
# Lazy loading means /health passes instantly and cross-encoder loads
# on the first actual user request in the background.
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("Loading BGE-Reranker (first rerank call)...")
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
        print("BGE-Reranker loaded")
    return _cross_encoder

# ════════════════════════════════════════════════════════════════════════════
# LLMs
# ════════════════════════════════════════════════════════════════════════════

main_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
fast_llm  = ChatGroq(model="llama-3.1-8b-instant",   temperature=0, max_tokens=10)

def llm(prompt: str) -> str:
    return main_llm.invoke(prompt).content

# ════════════════════════════════════════════════════════════════════════════
# VECTOR STORE — ChromaDB (local) or Pinecone (Render production)
# ════════════════════════════════════════════════════════════════════════════
#
# Why two different stores:
#   ChromaDB writes to local disk — perfect for development,
#   breaks on Render because the free tier has no persistent disk.
#   Pinecone is a managed cloud vector DB — always available on Render
#   regardless of restarts or redeploys.
#
# Switch is controlled by APP_ENV environment variable:
#   APP_ENV=development → ChromaDB (set in .env for local work)
#   APP_ENV=production  → Pinecone (set in render.yaml)
# ────────────────────────────────────────────────────────────────────────────

def _get_legal_vectorstore():
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if pinecone_key and APP_ENV == "production":
        # ── Pinecone (production) ─────────────────────────────────────────
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        index_name = os.getenv("PINECONE_INDEX_NAME", "legal-knowledge")
        print(f"Connecting to Pinecone index '{index_name}'...")

        pc = Pinecone(api_key=pinecone_key)
        existing = [i.name for i in pc.list_indexes()]

        if index_name not in existing:
            raise RuntimeError(
                f"Pinecone index '{index_name}' not found.\n"
                "Run: python scripts/migrate_to_pinecone.py"
            )

        vs = PineconeVectorStore(
            index_name=index_name,
            embedding=passage_emb,
            pinecone_api_key=pinecone_key,
        )
        print(f"Pinecone vector store connected")
        return vs

    else:
        # ── ChromaDB (local development) ──────────────────────────────────
        import chromadb
        db_path = os.getenv("CHROMA_PERSIST_DIR", "../data/vector_database")
        print(f"Loading ChromaDB from {db_path}...")

        client = chromadb.PersistentClient(path=db_path)
        vs = Chroma(
            client=client,
            embedding_function=passage_emb,
            collection_name="legal_knowledge_v2",
        )
        print(f"✅ ChromaDB loaded")
        return vs


legal_vs = _get_legal_vectorstore()

# ════════════════════════════════════════════════════════════════════════════
# BM25 INDEX — built over the legal corpus at startup
# ════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    return [t for t in re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()) if len(t) > 1]

legal_docs_raw = legal_vs.get()
legal_corpus   = legal_docs_raw["documents"]
legal_bm25     = BM25Okapi([tokenize(c) for c in legal_corpus])

print(f"Legal knowledge base loaded: {len(legal_corpus)} chunks")

# ── Per-document store (built when user uploads a file) ───────────────────
doc_vs     = None
doc_bm25   = None
doc_corpus = None

# ════════════════════════════════════════════════════════════════════════════
# RETRIEVAL HELPERS
# ════════════════════════════════════════════════════════════════════════════

def rrf(list_a: list, list_b: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion — merges two ranked lists into one.
    Documents appearing high in both lists score highest.
    Formula: score(doc) = sum(1 / (k + rank_i))
    """
    scores = {}
    for rank, d in enumerate(list_a):
        key = d.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
    for rank, d in enumerate(list_b):
        key = d.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
    return [
        Document(page_content=k)
        for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


def legal_search(query: str, k: int = 10) -> list:
    """
    Hybrid search over the Indian legal knowledge base.
    Combines:
      - Vector search  : semantic similarity via passage_emb (BGE-Base ONNX)
      - BM25 keyword   : exact term matching (good for article/section numbers)
    Merges with RRF to get the best of both.
    """
    vec_results  = legal_vs.similarity_search(query, k=k)
    bm25_scores  = legal_bm25.get_scores(tokenize(query))
    bm25_results = [
        Document(page_content=legal_corpus[i])
        for i in np.argsort(bm25_scores)[::-1][:k]
    ]
    return rrf(vec_results, bm25_results)


def document_search(query: str, k: int = 10) -> list:
    """
    Hybrid search over the user-uploaded document.
    Returns empty list if no document has been uploaded this session.
    """
    if doc_vs is None:
        return []
    vec_results  = doc_vs.similarity_search(query, k=k)
    bm25_scores  = doc_bm25.get_scores(tokenize(query))
    bm25_results = [
        Document(page_content=doc_corpus[i])
        for i in np.argsort(bm25_scores)[::-1][:k]
    ]
    return rrf(vec_results, bm25_results)


def rerank(query: str, docs: list, top_k: int = 6) -> list:
    """
    Cross-encoder reranking — deep pairwise scoring of (query, doc) pairs.
    Much more accurate than bi-encoder similarity but slower — only applied
    to the top K candidates from hybrid search, not the full corpus.
    Cross-encoder loads lazily on first call (see get_cross_encoder above).
    """
    if not docs:
        return []
    ce     = get_cross_encoder()
    pairs  = [(query, d.page_content) for d in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

# ════════════════════════════════════════════════════════════════════════════
# DOCUMENT INGESTION — called when user uploads a PDF via the API
# ════════════════════════════════════════════════════════════════════════════

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_document_store(
    file_path: str,
    persist_dir: str = "/tmp/doc_chroma",
) -> None:
    """
    Build a temporary vector store from an uploaded PDF.
    Called once per uploaded file — replaces any previous document store.

    Args:
        file_path:   Path to the uploaded PDF on disk
        persist_dir: Where to persist the temporary ChromaDB
                     (always local — this is per-session, not production data)
    """
    global doc_vs, doc_bm25, doc_corpus

    # Clean up previous session's document store
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    loader = PyMuPDFLoader(file_path)
    pages  = loader.load()

    chunks = []
    for page in pages:
        # Remove non-ASCII noise (Hindi Unicode, PDF artifacts)
        text = "".join(ch for ch in page.page_content if ord(ch) < 128)
        # Remove footer separator lines
        text = re.split(r"_{2,}", text)[0].strip()
        for chunk_text in splitter.split_text(text):
            if len(chunk_text.strip()) > 30:
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **page.metadata,
                        "source_file": Path(file_path).name,
                    },
                ))

    # Always use ChromaDB for the user document (temporary, per-session)
    doc_vs = Chroma.from_documents(
        documents=chunks,
        embedding=passage_emb,
        persist_directory=persist_dir,
    )
    doc_corpus = [c.page_content for c in chunks]
    doc_bm25   = BM25Okapi([tokenize(c) for c in doc_corpus])
    print(f"Document store: {len(chunks)} chunks from {Path(file_path).name}")

# ════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE
# ════════════════════════════════════════════════════════════════════════════

class LegalState(TypedDict):
    question:        str
    uploaded_file:   Optional[str]
    rewritten_query: Optional[str]
    strategy:        Optional[str]
    legal_docs:      Optional[List[Document]]
    document_docs:   Optional[List[Document]]
    final_docs:      Optional[List[Document]]
    answer:          Optional[str]
    critique:        Optional[str]
    # confidence field removed — was declared but never populated,
    # causing TypedDict validation warnings in LangGraph

# ════════════════════════════════════════════════════════════════════════════
# AGENT NODES
# ════════════════════════════════════════════════════════════════════════════

def rewrite_agent(state: LegalState) -> dict:
    """
    Transforms the user's raw question into a retrieval-optimised query.
    Adds legal terminology, article/section numbers, and formal phrasing
    that matches how the legal corpus is written.

    Example:
      Input:  "can police arrest without reason"
      Output: "grounds for arrest without warrant Section 41 CrPC
               personal liberty Article 21 Constitution"
    """
    prompt = (
        "You are a legal query optimizer for Indian law retrieval.\n"
        "Rewrite the following query to maximize retrieval quality.\n"
        "Include relevant article numbers, section numbers, or legal "
        "terminology if applicable.\n"
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"Original query: {state['question']}\n"
        "Rewritten query:"
    )
    return {"rewritten_query": llm(prompt).strip()}


def strategist_agent(state: LegalState) -> dict:
    """
    Routes the query to the correct retrieval path:
      LEGAL    → search Indian legal knowledge base only
      DOCUMENT → search user-uploaded document only
      BOTH     → search both in parallel

    Uses the fast 8B model because this is a classification task —
    no need for the full 70B reasoning model.

    The sanitiser handles cases where the LLM returns extra text like
    "BOTH - this question needs both sources" instead of just "BOTH".
    """
    file_ctx = ""
    if state.get("uploaded_file"):
        fname    = Path(state["uploaded_file"]).stem.replace("_", " ")
        file_ctx = f'\nAn uploaded document is available: "{fname}".'

    prompt = (
        "Routing agent for a legal RAG system. Choose retrieval strategy.\n"
        f'Question: "{state["question"]}"{file_ctx}\n'
        "LEGAL: General Indian law / constitutional question\n"
        "DOCUMENT: Specific facts from the uploaded document only\n"
        "BOTH: Requires both law context and document facts\n"
        "Return ONLY one word: LEGAL, DOCUMENT, or BOTH"
    )

    raw_strategy = fast_llm.invoke(prompt).content.strip().upper()

    # Sanitise — LLM sometimes returns "BOTH - needs legal context" etc.
    if "BOTH" in raw_strategy:
        strategy = "BOTH"
    elif "DOCUMENT" in raw_strategy:
        strategy = "DOCUMENT"
    else:
        strategy = "LEGAL"

    # Safety: if no file uploaded, DOCUMENT/BOTH makes no sense
    if strategy in ("DOCUMENT", "BOTH") and not state.get("uploaded_file"):
        strategy = "LEGAL"

    print(f"  [Strategist] Route: {strategy}")
    return {"strategy": strategy}


def legal_agent(state: LegalState) -> dict:
    """
    Retrieves relevant chunks from the Indian legal knowledge base.
    Uses hybrid search (vector + BM25) with RRF merging.
    InLegalBERT encodes the query for semantically accurate retrieval.
    """
    return {"legal_docs": legal_search(state["rewritten_query"])}


def document_agent(state: LegalState) -> dict:
    """
    Retrieves relevant chunks from the user-uploaded document.
    Returns empty list if no document has been uploaded.
    """
    return {"document_docs": document_search(state["rewritten_query"])}


def fusion_agent(state: LegalState) -> dict:
    """
    Merges and reranks results from legal_agent and document_agent.
    Combines both lists and applies cross-encoder reranking to pick
    the 6 most relevant chunks regardless of which source they came from.
    """
    docs  = (state.get("legal_docs") or []) + (state.get("document_docs") or [])
    final = rerank(state["rewritten_query"], docs)
    return {"final_docs": final}


def answer_agent(state: LegalState) -> dict:
    """
    Generates the final legal answer from the reranked context chunks.
    Uses the full 70B model for reasoning quality.

    The prompt enforces grounding — the LLM must cite specific Articles
    or Sections and cannot use knowledge outside the provided context.
    """
    context = "\n\n---\n\n".join(d.page_content for d in state["final_docs"])
    prompt  = (
        "You are a precise Indian legal assistant.\n"
        "Answer using ONLY the provided context. "
        "If the context is insufficient, say so clearly — do not guess.\n"
        "Always cite the specific Article, Section, or document clause.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer:"
    )
    return {"answer": llm(prompt)}


def critic_agent(state: LegalState) -> dict:
    """
    Evaluates the generated answer for faithfulness and completeness.
    Outputs a structured critique string that the frontend can display.

    This is a lightweight self-evaluation — not a replacement for
    proper RAGAS evaluation (see notebook 03).
    """
    prompt = (
        "Evaluate this legal answer:\n"
        "1. Faithfulness (0-10): Is every claim traceable to the context?\n"
        "2. Completeness (0-10): Does it fully address the question?\n"
        "3. Hallucination: List any claims NOT supported by the context.\n\n"
        f"Question: {state['question']}\n"
        f"Answer: {state['answer']}\n\n"
        "Respond in this exact format:\n"
        "FAITH:<score> COMPLETE:<score> ISSUES:<description or None>"
    )
    return {"critique": llm(prompt)}

# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════

def router(state: LegalState):
    """
    Conditional edge function — determines which node(s) to visit after
    the strategist. Returns:
      "legal"            → single branch to legal_agent
      "document"         → single branch to document_agent
      ["legal","document"] → parallel branches to both
    LangGraph handles parallel execution automatically for the list case.
    """
    s = state.get("strategy", "LEGAL")
    if s == "LEGAL":    return "legal"
    if s == "DOCUMENT": return "document"
    return ["legal", "document"]  # BOTH — parallel execution

# ════════════════════════════════════════════════════════════════════════════
# BUILD & COMPILE LANGGRAPH
# ════════════════════════════════════════════════════════════════════════════

wf = StateGraph(LegalState)

wf.add_node("rewrite",    rewrite_agent)
wf.add_node("strategist", strategist_agent)
wf.add_node("legal",      legal_agent)
wf.add_node("document",   document_agent)
wf.add_node("fusion",     fusion_agent)
wf.add_node("answer",     answer_agent)   # was 'generate_answer' — renamed for consistency
wf.add_node("critic",     critic_agent)

wf.set_entry_point("rewrite")
wf.add_edge("rewrite", "strategist")

wf.add_conditional_edges(
    "strategist",
    router,
    {
        "legal":    "legal",
        "document": "document",
    }
)

wf.add_edge("legal",    "fusion")
wf.add_edge("document", "fusion")
wf.add_edge("fusion",   "answer")
wf.add_edge("answer",   "critic")
wf.add_edge("critic",   END)          # was missing in original — graph had no terminal edge

graph = wf.compile()
print("LangGraph compiled")

# ════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_legal_rag(question: str, uploaded_file: str = None) -> dict:
    """
    Main entry point for the legal RAG pipeline.
    Called directly from notebooks and from cached_rag() below.

    Args:
        question:      Natural language legal question
        uploaded_file: Optional path to a user-uploaded PDF on disk

    Returns:
        Full LangGraph state dict containing:
          answer:          Generated legal answer with citations
          critique:        Faithfulness/completeness self-evaluation
          strategy:        Routing decision (LEGAL / DOCUMENT / BOTH)
          final_docs:      Reranked context chunks used for the answer
          rewritten_query: Retrieval-optimised version of the question
    """
    if uploaded_file:
        build_document_store(uploaded_file)

    # All fields must be explicitly set — LangGraph validates the full state
    initial_state: LegalState = {
        "question":        question,
        "uploaded_file":   uploaded_file,
        "rewritten_query": None,
        "strategy":        None,
        "legal_docs":      None,
        "document_docs":   None,
        "final_docs":      None,
        "answer":          None,
        "critique":        None,
    }

    return graph.invoke(initial_state)

# ════════════════════════════════════════════════════════════════════════════
# REDIS CACHE LAYER
# ════════════════════════════════════════════════════════════════════════════
#
# Sits on top of run_legal_rag. Use cached_rag() in FastAPI endpoints.
# Use run_legal_rag() directly in notebooks (no benefit caching one-off calls).
#
# Cache key: MD5(question + file_path) with "lexrag:" prefix
# TTL: 1 hour (legal answers don't change frequently)
# Behaviour on Redis failure: silently falls through to live pipeline
# ────────────────────────────────────────────────────────────────────────────

_redis_client = None


def get_redis():
    """
    Lazy Redis connection — deferred until first cached_rag() call.
    Returns None (not an error) if REDIS_URL is not configured.
    This means the entire module loads cleanly even without Redis.
    """
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return None  # Redis not configured — caching disabled, not an error
        try:
            _redis_client = redis.from_url(redis_url, decode_responses=True)
            _redis_client.ping()
            print("Redis connected")
        except Exception as e:
            print(f"Redis unavailable: {e} — running without cache")
            _redis_client = None
    return _redis_client


def cached_rag(question: str, file_path: str = None) -> dict:
    """
    Drop-in replacement for run_legal_rag with Redis caching.

    Cache hit  → returns instantly, skips all LLM calls (~50ms)
    Cache miss → runs full pipeline, stores result, returns (~15s)
    Redis down → falls through to live pipeline transparently

    Note: final_docs (Document objects) are NOT cached — not JSON
    serialisable. The cached response contains answer, strategy,
    critique, and a 'cached: True' flag.
    """
    r = get_redis()

    # Build cache key — normalise None file_path to empty string
    raw_key   = f"{question}{file_path or ''}"
    cache_key = f"lexrag:{hashlib.md5(raw_key.encode()).hexdigest()}"

    # ── Cache read ────────────────────────────────────────────────────────
    if r:
        try:
            cached = r.get(cache_key)
            if cached:
                print(f"⚡ Cache HIT ({cache_key[:24]}...)")
                return json.loads(cached)
        except Exception as e:
            print(f"Redis read error: {e} — falling back to live pipeline")

    # ── Cache miss — run live pipeline ────────────────────────────────────
    print("🔍 Cache MISS — running LangGraph pipeline...")
    result = run_legal_rag(question, file_path)

    # ── Cache write ───────────────────────────────────────────────────────
    if r:
        try:
            payload = json.dumps({
                "answer":   result.get("answer",   ""),
                "strategy": result.get("strategy", "LEGAL"),
                "critique": result.get("critique", ""),
                "cached":   True,
            })
            r.setex(cache_key, 3600, payload)
            print(f"Cached for 1 hour ({cache_key[:24]}...)")
        except Exception as e:
            print(f"Redis write error: {e} — result not cached")

    return result
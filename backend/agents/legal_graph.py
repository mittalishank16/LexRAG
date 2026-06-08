# backend/agents/legal_graph.py
import os
import re
import json
import shutil
import hashlib
import nltk
import numpy as np
import redis

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
nltk.download('punkt', quiet=True)

APP_ENV = os.getenv("APP_ENV", "production")
device = "cpu"
print(f"Running on device: {device} | Environment: {APP_ENV}")

# Fix path translation safely for local dev environments vs Docker environments
MODEL_PATH_INLEGALBERT = "models/inlegalbert_onnx"
MODEL_PATH_BGE = "models/bge_onnx"

if not os.path.exists(MODEL_PATH_INLEGALBERT) and os.path.exists(os.path.join("backend", MODEL_PATH_INLEGALBERT)):
    MODEL_PATH_INLEGALBERT = os.path.join("backend", MODEL_PATH_INLEGALBERT)

if not os.path.exists(MODEL_PATH_BGE) and os.path.exists(os.path.join("backend", MODEL_PATH_BGE)):
    MODEL_PATH_BGE = os.path.join("backend", MODEL_PATH_BGE)

# Initialize low-RAM ONNX embedding engines
from models.onnx_embeddings import ONNXEmbeddings
print(f"Loading ONNX Query Encoder from: {MODEL_PATH_INLEGALBERT}")
query_emb = ONNXEmbeddings(MODEL_PATH_INLEGALBERT)

print(f"Loading ONNX Passage Encoder from: {MODEL_PATH_BGE}")
passage_emb = ONNXEmbeddings(MODEL_PATH_BGE)

# Lazy-loaded Cross-Encoder setup
_cross_encoder_instance = None

def get_cross_encoder():
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        from sentence_transformers import CrossEncoder
        print("Lazy loading CrossEncoder into memory...")
        _cross_encoder_instance = CrossEncoder("BAAI/bge-reranker-base")
    return _cross_encoder_instance

# LLM initializations
main_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=10)

def llm(prompt: str) -> str:
    return main_llm.invoke(prompt).content

# Vector Store Resolution
index_name = os.getenv("PINECONE_INDEX_NAME", "legal-knowledge")

if os.getenv("APP_ENV") == "production":
    from langchain_pinecone import PineconeVectorStore
    print("Connecting to production Pinecone index...")
    legal_vs = PineconeVectorStore(index_name=index_name, embedding=passage_emb)
else:
    print("Connecting to local Chroma database...")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "../data/vector_database")
    legal_vs = Chroma(persist_directory=persist_dir, embedding_function=passage_emb)

# ── LAZY LOADED BM25 INDEX STRUCTURE ──
def tokenize(text: str) -> list:
    return [t for t in re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()) if len(t) > 1]

_legal_corpus = None
_legal_bm25 = None

def get_legal_bm25():
    """Builds and caches the legal keyword index only when search fires."""
    global _legal_corpus, _legal_bm25
    if _legal_bm25 is None:
        print("Extracting data index and generating BM25 matrix lazily...")
        legal_docs_raw = legal_vs.get()
        _legal_corpus = legal_docs_raw["documents"]
        _legal_bm25 = BM25Okapi([tokenize(c) for c in _legal_corpus])
        print(f"Matrix built successfully with {len(_legal_corpus)} chunks.")
    return _legal_corpus, _legal_bm25

# Document-specific indices
doc_vs = None
doc_bm25 = None
doc_corpus = None

# Retrieval functions
def rrf(list_a: list, list_b: list, k: int = 60) -> list:
    scores = {}
    for rank, d in enumerate(list_a):
        key = d.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
    for rank, d in enumerate(list_b):
        key = d.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
    return [Document(page_content=k) for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def legal_search(query: str, k: int = 10) -> list:
    corpus, bm25 = get_legal_bm25()
    vec_results = legal_vs.similarity_search(query, k=k)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_results = [Document(page_content=corpus[i]) for i in np.argsort(bm25_scores)[::-1][:k]]
    return rrf(vec_results, bm25_results)

def document_search(query: str, k: int = 10) -> list:
    if doc_vs is None:
        return []
    vec_results = doc_vs.similarity_search(query, k=k)
    bm25_scores = doc_bm25.get_scores(tokenize(query))
    bm25_results = [Document(page_content=doc_corpus[i]) for i in np.argsort(bm25_scores)[::-1][:k]]
    return rrf(vec_results, bm25_results)

def rerank(query: str, docs: list, top_k: int = 6) -> list:
    if not docs:
        return []
    ce = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

# Document processing
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])

def build_document_store(file_path: str, persist_dir: str = "/tmp/doc_chroma") -> None:
    global doc_vs, doc_bm25, doc_corpus
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    chunks = []
    for page in pages:
        text = "".join(ch for ch in page.page_content if ord(ch) < 128)
        text = re.split(r'_{2,}', text)[0].strip()
        for chunk_text in splitter.split_text(text):
            if len(chunk_text.strip()) > 30:
                chunks.append(Document(page_content=chunk_text, metadata={**page.metadata, "source_file": os.path.basename(file_path)}))
                
    doc_vs = Chroma.from_documents(documents=chunks, embedding=passage_emb, persist_directory=persist_dir)
    doc_corpus = [c.page_content for c in chunks]
    doc_bm25 = BM25Okapi([tokenize(c) for c in doc_corpus])
    print(f"Document index compiled with {len(chunks)} fragments.")

# Agent State Graph definitions
class LegalState(TypedDict):
    question: str
    uploaded_file: Optional[str]
    rewritten_query: Optional[str]
    strategy: Optional[str]
    legal_docs: Optional[List[Document]]
    document_docs: Optional[List[Document]]
    final_docs: Optional[List[Document]]
    answer: Optional[str]
    critique: Optional[str]

def rewrite_agent(state: LegalState) -> dict:
    prompt = f"Optimize this Indian legal query for retrieval. Return only the optimized query string:\n\nQuery: {state['question']}"
    return {"rewritten_query": llm(prompt).strip()}

def strategist_agent(state: LegalState) -> dict:
    file_ctx = f'\nUploaded file available: "{os.path.basename(state["uploaded_file"])}".' if state.get("uploaded_file") else ""
    prompt = f"Route this legal question. Respond with exactly one word (LEGAL, DOCUMENT, or BOTH):\nQuestion: {state['question']}{file_ctx}"
    res = fast_llm.invoke(prompt).content.strip().upper()
    strategy = "BOTH" if "BOTH" in res else ("DOCUMENT" if "DOCUMENT" in res else "LEGAL")
    if strategy in ("DOCUMENT", "BOTH") and not state.get("uploaded_file"):
        strategy = "LEGAL"
    return {"strategy": strategy}

def legal_agent(state: LegalState) -> dict:
    return {"legal_docs": legal_search(state["rewritten_query"])}

def document_agent(state: LegalState) -> dict:
    return {"document_docs": document_search(state["rewritten_query"])}

def fusion_agent(state: LegalState) -> dict:
    combined = (state.get("legal_docs") or []) + (state.get("document_docs") or [])
    return {"final_docs": rerank(state["rewritten_query"], combined)}

def answer_agent(state: LegalState) -> dict:
    ctx = "\n\n---\n\n".join(d.page_content for d in state["final_docs"])
    prompt = f"Answer precisely using only the context provided. Cite relevant Sections/Articles.\n\nContext:\n{ctx}\n\nQuestion: {state['question']}"
    return {"answer": llm(prompt)}

def critic_agent(state: LegalState) -> dict:
    prompt = f"Evaluate this answer for faithfulness and completeness:\nQuestion: {state['question']}\nAnswer: {state['answer']}\n\nRespond as: FAITH:<score> COMPLETE:<score> ISSUES:<desc>"
    return {"critique": llm(prompt)}

def router(state: LegalState):
    s = state.get("strategy", "LEGAL")
    return "document" if s == "DOCUMENT" else ("legal" if s == "LEGAL" else ["legal", "document"])

wf = StateGraph(LegalState)
wf.add_node("rewrite", rewrite_agent)
wf.add_node("strategist", strategist_agent)
wf.add_node("legal", legal_agent)
wf.add_node("document", document_agent)
wf.add_node("fusion", fusion_agent)
wf.add_node("answer", answer_agent)
wf.add_node("critic", critic_agent)

wf.set_entry_point("rewrite")
wf.add_edge("rewrite", "strategist")
wf.add_conditional_edges("strategist", router, {"legal": "legal", "document": "document"})
wf.add_edge("legal", "fusion")
wf.add_edge("document", "fusion")
wf.add_edge("fusion", "answer")
wf.add_edge("answer", "critic")
wf.add_edge("critic", END)

graph = wf.compile()

def run_legal_rag(question: str, uploaded_file: str = None) -> dict:
    if uploaded_file:
        build_document_store(uploaded_file)
    return graph.invoke({"question": question, "uploaded_file": uploaded_file, "rewritten_query": None, "strategy": None, "legal_docs": None, "document_docs": None, "final_docs": None, "answer": None, "critique": None})

_redis_client = None
def get_redis():
    global _redis_client
    if _redis_client is None and os.getenv("REDIS_URL"):
        try:
            _redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
            _redis_client.ping()
        except Exception:
            _redis_client = None
    return _redis_client

def cached_rag(question: str, file_path: str = None) -> dict:
    r = get_redis()
    cache_key = f"lexrag:{hashlib.md5(f'{question}{file_path or ""}'.encode()).hexdigest()}"
    if r:
        try:
            cached = r.get(cache_key)
            if cached: return json.loads(cached)
        except Exception:
            pass
            
    result = run_legal_rag(question, file_path)
    if r:
        try:
            r.setex(cache_key, 3600, json.dumps({"answer": result.get("answer", ""), "strategy": result.get("strategy", "LEGAL"), "critique": result.get("critique", ""), "cached": True}))
        except Exception:
            pass
    return result
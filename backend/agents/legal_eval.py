# backend/agents/legal_eval.py
"""
RAGAS evaluation pipeline for the Legal Agentic RAG system.
Converted from LegalRAG14_RAGAS_fixed.ipynb.

Usage:
    python legal_eval.py                     # run full eval on built-in dataset
    python legal_eval.py --output results/   # write CSV + JSON to a directory
"""

import os
import re
import shutil
import nltk
import numpy as np
import argparse
from pathlib import Path
from typing import List, Optional, TypedDict, Any

import torch
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langgraph.graph import StateGraph, END
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Ollama Models ─────────────────────────────────────────────────────────────
_OLLAMA_BASE = "http://localhost:11434"
_CHAT_MODEL  = "llama3.1:8b"
_EMBED_MODEL = "nomic-embed-text"

main_llm = ChatOllama(model=_CHAT_MODEL, base_url=_OLLAMA_BASE)
judge_llm_raw = ChatOllama(model=_CHAT_MODEL, format="json", base_url=_OLLAMA_BASE)

# Exact matching model wrapper used in the chunking stage
passage_emb = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5',
    model_kwargs={'device': DEVICE},
    encode_kwargs={
        'normalize_embeddings': True,
        'prompt': "Represent this sentence for searching relevant passages: "
    }
)

# ── Sentence-Transformers (local, no API key needed) ──────────────────────────
bge_sentence_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=DEVICE)
cross_encoder      = CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)

# ── Vector store config ───────────────────────────────────────────────────────
DB_PATH    = os.getenv("CHROMA_PERSIST_DIR", "../data/vector_database")
COLLECTION = "legal_knowledge_v2"


# ── LLM helper ───────────────────────────────────────────────────────────────
def llm_invoke(prompt: str) -> str:
    return main_llm.invoke(prompt).content



# ── Tokenisation helpers ──────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [t for t in tokens if len(t) > 1]


# ── Semantic + adaptive chunking ──────────────────────────────────────────────
_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def semantic_chunk(text: str, threshold: float = 0.78) -> List[str]:
    """Split text into meaning-coherent chunks using BGE cosine similarity."""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences

    prefixed = [f"Represent this sentence for retrieval: {s}" for s in sentences]
    embs     = bge_sentence_model.encode(prefixed, normalize_embeddings=True)

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i], embs[i - 1]))
        if sim < threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i])
    chunks.append(" ".join(current))
    return chunks


def adaptive_chunk(text: str) -> List[str]:
    """Semantic chunking followed by size-based recursive splitting."""
    final: List[str] = []
    for chunk in semantic_chunk(text):
        final.extend(_recursive_splitter.split_text(chunk))
    return final


def hierarchical_split(docs: List[Document]) -> tuple[List[Document], List[Document]]:
    """Return (parent_docs, child_docs) with metadata linking them."""
    parent_docs, child_docs = [], []
    for doc in docs:
        parents = semantic_chunk(doc.page_content)
        for idx, parent_text in enumerate(parents):
            parent_docs.append(Document(
                page_content=parent_text,
                metadata={**doc.metadata, "chunk_type": "parent", "parent_index": idx},
            ))
            for child_text in adaptive_chunk(parent_text):
                child_docs.append(Document(
                    page_content=child_text,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "child",
                        "parent_chunk": parent_text[:200],
                    },
                ))
    return parent_docs, child_docs


# ── Document store (per uploaded file) ───────────────────────────────────────
doc_vectorstore: Optional[Chroma]      = None
doc_bm25:        Optional[BM25Okapi]   = None
doc_corpus:      Optional[List[str]]   = None


def build_document_retriever(
    child_docs: List[Document],
    persist_dir: str = "/tmp/doc_chroma_eval",
) -> None:
    global doc_vectorstore, doc_bm25, doc_corpus

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"  [build_document_retriever] Cleared old store at {persist_dir}")

    doc_vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=passage_emb,
        persist_directory=persist_dir,
    )
    doc_corpus = [d.page_content for d in child_docs]
    doc_bm25   = BM25Okapi([tokenize(c) for c in doc_corpus])
    print(f"  [build_document_retriever] {len(child_docs)} chunks indexed.")


def load_and_index_file(file_path: str) -> None:
    """Load a PDF, split it hierarchically, and build the document retriever."""
    raw_docs = PyMuPDFLoader(file_path).load()
    _, child_docs = hierarchical_split(raw_docs)
    build_document_retriever(child_docs)


# ── Legal knowledge base ──────────────────────────────────────────────────────
legal_vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=passage_emb,
    collection_name=COLLECTION,
)

try:
    _raw         = legal_vectorstore.get()
    legal_corpus = _raw["documents"] if _raw and "documents" in _raw else []
except Exception as e:
    print(f"Note: Could not load legal corpus for BM25: {e}")
    legal_corpus = []

if legal_corpus:
    legal_bm25 = BM25Okapi([tokenize(c) for c in legal_corpus])
else:
    print("Legal knowledge base is empty — BM25 will return no results.")
    class _EmptyBM25:
        def get_scores(self, _):
            return []
    legal_bm25 = _EmptyBM25()

print(f"Legal knowledge base: {len(legal_corpus)} chunks.")


# ── Retrieval helpers ─────────────────────────────────────────────────────────
def _rrf(a: List[Document], b: List[Document], k: int = 60) -> List[Document]:
    scores: dict[str, float] = {}
    for rank, d in enumerate(a): scores[d.page_content] = scores.get(d.page_content, 0) + 1 / (k + rank)
    for rank, d in enumerate(b): scores[d.page_content] = scores.get(d.page_content, 0) + 1 / (k + rank)
    return [Document(page_content=p) for p, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def legal_search(query: str, k: int = 8) -> List[Document]:
    vec = legal_vectorstore.similarity_search(query, k=k)
    sc  = legal_bm25.get_scores(tokenize(query))
    bm  = [Document(page_content=legal_corpus[i]) for i in np.argsort(sc)[::-1][:k]]
    return _rrf(vec, bm)


def document_search(query: str, k: int = 8) -> List[Document]:
    if doc_vectorstore is None:
        return []
    vec = doc_vectorstore.similarity_search(query, k=k)
    sc  = doc_bm25.get_scores(tokenize(query))
    bm  = [Document(page_content=doc_corpus[i]) for i in np.argsort(sc)[::-1][:k]]
    return _rrf(vec, bm)


def rerank(query: str, docs: List[Document], top_k: int = 6) -> List[Document]:
    if not docs:
        return []
    pairs  = [(query, d.page_content) for d in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


# ── LangGraph State ───────────────────────────────────────────────────────────
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


# ── Agent nodes ───────────────────────────────────────────────────────────────
def rewrite_agent(state: LegalState) -> dict:
    prompt = (
        "Rewrite this query for Indian law retrieval. Return ONLY the rewritten query.\n"
        f"Original: {state['question']}\nRewritten:"
    )
    return {"rewritten_query": llm_invoke(prompt).strip()}


def strategist_agent(state: LegalState) -> dict:
    file_ctx = ""
    if state.get("uploaded_file"):
        name = Path(state["uploaded_file"]).stem.replace("_", " ")
        file_ctx = (
            f"\nCRITICAL CONTEXT: An uploaded case document is available regarding '{name}'. "
            "If the user's question mentions this case or asks for specifics not in general law, "
            "route to DOCUMENT or BOTH."
        )
    prompt = (
        f"You are an intelligent routing agent managing a legal RAG system.\n"
        f"Question: \"{state['question']}\"{file_ctx}\n\n"
        "Options:\n"
        "  LEGAL    — general Indian law / constitutional queries only\n"
        "  DOCUMENT — specific facts from the uploaded case document\n"
        "  BOTH     — needs both sources\n\n"
        "Return ONLY one word: LEGAL, DOCUMENT, or BOTH."
    )
    raw = main_llm.invoke(prompt).content.strip().upper()
    if   "BOTH"     in raw: strategy = "BOTH"
    elif "DOCUMENT" in raw: strategy = "DOCUMENT"
    else:                   strategy = "LEGAL"
    if strategy in ("DOCUMENT", "BOTH") and not state.get("uploaded_file"):
        strategy = "LEGAL"
    print(f"  [Strategist] → {strategy}")
    return {"strategy": strategy}


def legal_agent(state: LegalState) -> dict:
    return {"legal_docs": legal_search(state["rewritten_query"])}


def document_agent(state: LegalState) -> dict:
    return {"document_docs": document_search(state["rewritten_query"])}


def fusion_agent(state: LegalState) -> dict:
    docs = (state.get("legal_docs") or []) + (state.get("document_docs") or [])
    return {"final_docs": rerank(state["rewritten_query"], docs)}


def answer_agent(state: LegalState) -> dict:
    ctx    = "\n\n---\n\n".join(d.page_content for d in state["final_docs"])
    is_doc = state.get("strategy") in ("DOCUMENT", "BOTH")
    cite   = (
        "Cite the specific paragraph, date, or party name from the uploaded case document."
        if is_doc else
        "Cite the specific Article or Section that supports your answer."
    )
    prompt = (
        f"You are a precise Indian legal assistant. Answer using ONLY the context below.\n"
        f"If the context is insufficient, say so — do not guess.\n{cite}\n\n"
        f"Context:\n{ctx}\n\nQuestion: {state['question']}\n\nAnswer:"
    )
    return {"answer": llm_invoke(prompt)}


def critic_agent(state: LegalState) -> dict:
    prompt = (
        f"Evaluate this answer.\n\n"
        f"Question:\n{state['question']}\n\n"
        f"Answer:\n{state['answer']}\n\n"
        "Is the answer grounded and factually correct? Explain briefly."
    )
    return {"critique": llm_invoke(prompt)}


# ── Router ────────────────────────────────────────────────────────────────────
def router(state: LegalState):
    s = state.get("strategy", "LEGAL")
    if s == "LEGAL":    return "legal"
    if s == "DOCUMENT": return "document"
    return ["legal", "document"]


# ── Build graph ───────────────────────────────────────────────────────────────
_wf = StateGraph(LegalState)
for _name, _fn in [
    ("rewrite",    rewrite_agent),
    ("strategist", strategist_agent),
    ("legal",      legal_agent),
    ("document",   document_agent),
    ("fusion",     fusion_agent),
    ("answer",     answer_agent),
    ("critic",     critic_agent),
]:
    _wf.add_node(_name, _fn)

_wf.set_entry_point("rewrite")
_wf.add_edge("rewrite", "strategist")
_wf.add_conditional_edges("strategist", router, {"legal": "legal", "document": "document"})
_wf.add_edge("legal",    "fusion")
_wf.add_edge("document", "fusion")
_wf.add_edge("fusion",   "answer")
_wf.add_edge("answer",   "critic")
_wf.add_edge("critic",   END)
graph = _wf.compile()
print("LangGraph (eval) compiled.")


# ── Public entry point ────────────────────────────────────────────────────────
def run_legal_rag(question: str, uploaded_file: Optional[str] = None) -> dict:
    """Run the agentic RAG pipeline and return answer + final_docs."""
    if uploaded_file:
        load_and_index_file(uploaded_file)
    return graph.invoke({
        "question":        question,
        "uploaded_file":   uploaded_file,
        "rewritten_query": None,
        "strategy":        None,
        "legal_docs":      None,
        "document_docs":   None,
        "final_docs":      None,
        "answer":          None,
        "critique":        None,
    })
  


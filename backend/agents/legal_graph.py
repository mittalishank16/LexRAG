# ________________________________importing all the liberaries_________________________________________________________

import os, re, shutil, nltk, numpy as np 
from pathlib import Path 
from typing import List, Optional, TypedDict 
from dotenv import load_dotenv 
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
nltk.download('puntk', quiet=True)
nltk.download('punkt_tab', quiet=True)

#____________________________device____________________________________________________________________________
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# _________________________Models________________________________________________________________________

# initialize all models

# passage embeddings
passage_embedding = HuggingFaceEmbeddings( 
    model_name='BAAI/bge-base-en-v1.5', 
    model_kwargs={'device': device}, 
    encode_kwargs={'normalize_embeddings': True} 
    )

# Query embedding (InLegalBERT)
query_embedding = HuggingFaceEmbeddings(
    model_name='law-ai/InLegalBERT', 
    model_kwargs={'device': device}, 
    encode_kwargs={'normalize_embeddings': True}
)

# Sentence Similarity for chunking

sentence_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)

# Cross encoder reranker
cross_encoder = CrossEncoder('BAAI/bge-reranker-base')

# LLM via Groq
main_llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.2)
fast_llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0, max_tokens=10)

def llm(prompt: str)->str:
    return main_llm.invoke(prompt).content

# __________________________________Hybrid Search Infrastructure____________________________________________________
# retrieval functions

# load pre-built legal vector store
DB_PATH = '../data/vector_database'

legal_vs = Chroma(
    persist_directory= DB_PATH,
    embedding_function=passage_embedding, # Same model used at index time
    collection_name= 'legal_knowledge_v2'
)


# Build BM25 index over legal Corpus
def tokenize(text: str)->str:
    return [t for t in re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()) if len(t) > 1]

legal_docs_raw = legal_vs.get() 
legal_corpus = legal_docs_raw['documents'] 
legal_bm25 = BM25Okapi([tokenize(c) for c in legal_corpus])

print(f'Legal knowledge base loaded: {len(legal_corpus)} chunks')

# Per document store (built at query time)
doc_vs, doc_bm25, doc_corpus = None, None, None

# retrieval helpers
def rrf(list_a, list_b, k = 60):
    scores = {}
    for rank, d in enumerate(list_a):
        key = d.page_content
        scores[key] = scores.get(key, 0) + 1/(k + rank)
    for rank, d in enumerate(list_b): 
        key = d.page_content 
        scores[key] = scores.get(key, 0) + 1/(k + rank) 
    return [Document(page_content=k) for k, _ in 
            sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def legal_search(query: str, k: int = 10) -> list: 
    vec_results = legal_vs.similarity_search(query, k=k) 
    bm25_scores = legal_bm25.get_scores(tokenize(query)) 
    bm25_results = [Document(page_content=legal_corpus[i]) 
                    for i in np.argsort(bm25_scores)[::-1][:k]] 
    return rrf(vec_results, bm25_results)

def document_search(query: str, k: int = 10) -> list: 
    if doc_vs is None: 
        return [] 
    vec_results = doc_vs.similarity_search(query, k=k) 
    bm25_scores = doc_bm25.get_scores(tokenize(query)) 
    bm25_results = [Document(page_content=doc_corpus[i]) for i in 
                    np.argsort(bm25_scores)[::-1][:k]] 
    return rrf(vec_results, bm25_results)


def rerank(query: str, docs: list, top_k: int = 6) -> list: 
    if not docs: 
        return [] 
    
    pairs = [(query, d.page_content) for d in docs] 
    scores = cross_encoder.predict(pairs) 
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True) 
    return [d for d, _ in ranked[:top_k]]
    

    # _______________________________________Document Ingestion___________________________________________________

    # upload document proccessor

splitter = RecursiveCharacterTextSplitter( 
    chunk_size=500, 
    chunk_overlap=100, 
    separators=['\n\n', '\n', '. ', ' ', ''] 
    )

def build_document_store(file_path: str, persist_dir: str = './tmp_doc_chroma'): 
    global doc_vs, doc_bm25, doc_corpus 
    
    if os.path.exists(persist_dir): 
        shutil.rmtree(persist_dir) 
        
    loader = PyMuPDFLoader(file_path) 
    pages = loader.load() 
    
    chunks = [] 
    for page in pages: 
        text = ''.join(ch for ch in page.page_content if ord(ch) < 128) 
        text = re.split(r'_{2,}', text)[0].strip() 
        for chunk_text in splitter.split_text(text): 
            if len(chunk_text.strip()) > 30: 
                chunks.append(Document( 
                    page_content=chunk_text, 
                    metadata={**page.metadata, 'source_file': Path(file_path).name} 
                    ))
                
    doc_vs = Chroma.from_documents( 
        documents=chunks, 
        embedding=passage_embedding, 
        persist_directory=persist_dir 
        )
    
    doc_corpus = [c.page_content for c in chunks] 
    doc_bm25 = BM25Okapi([tokenize(c) for c in doc_corpus]) 
    print(f'Document store: {len(chunks)} chunks from {Path(file_path).name}')

    
    #______________________________________LangGraph Agents_________________________________________________________

    # define all agent functions

class LegalState(TypedDict): 
    question: str 
    uploaded_file: Optional[str] 
    rewritten_query: Optional[str] 
    strategy: Optional[str] 
    legal_docs: Optional[List] 
    document_docs: Optional[List] 
    final_docs: Optional[List] 
    answer: Optional[str] 
    critique: Optional[str] 
    confidence: Optional[float]

    

# Agent Nodes
def rewrite_agent(state: LegalState) -> dict: 
    prompt = f'''You are a legal query optimizer. 
    Rewrite the following query to maximize retrieval quality from an Indian law database. 
    Include relevant article numbers, section numbers, or legal terminology if applicable. 
    Return ONLY the rewritten query, nothing else. 
    
    Original query: {state['question']} 
    Rewritten query:''' 
    
    return {'rewritten_query': llm(prompt).strip()}


def strategist_agent(state: LegalState) -> dict: 
    file_ctx = '' 
    if state.get('uploaded_file'): 
        fname = Path(state['uploaded_file']).stem.replace('_', ' ') 
        file_ctx = f'\nAn uploaded document is available: "{fname}".' 
    prompt = f'''Routing agent for a legal RAG system. Choose retrieval strategy. 
                Question: "{state['question']}"{file_ctx} 
                LEGAL: General Indian law / constitutional question 
                DOCUMENT: Specific facts from the uploaded document
                BOTH: Requires both law context and document facts 
                Return ONLY: LEGAL, DOCUMENT, or BOTH''' 
    strategy = fast_llm.invoke(prompt).content.strip().upper() 
    if strategy not in ('LEGAL', 'DOCUMENT', 'BOTH'): 
        strategy = 'BOTH' if state.get('uploaded_file') else 'LEGAL' 
    
    print(f' [Strategist] Route: {strategy}') 
    return {'strategy': strategy}


def legal_agent(state: LegalState) -> dict: 
    return {'legal_docs': legal_search(state['rewritten_query'])}



def document_agent(state: LegalState) -> dict: 
    return {'document_docs': document_search(state['rewritten_query'])}



def fusion_agent(state: LegalState) -> dict: 
    docs = (state.get('legal_docs') or []) + (state.get('document_docs') or []) 
    final = rerank(state['rewritten_query'], docs) 
    return {'final_docs': final}



def answer_agent(state: LegalState) -> dict: 
    context = '\n\n---\n\n'.join(d.page_content for d in state['final_docs']) 
    prompt = f'''You are a precise Indian legal assistant. Answer using ONLY the provided context. 
                If insufficient, say so clearly. Always cite the specific Article, Section, or document clause. 
                
            Context: {context} 
            Question: {state['question']} 
            Answer:''' 
    
    return {'answer': llm(prompt)}

def critic_agent(state: LegalState) -> dict: 
    prompt = f'''Evaluate this legal answer for: 
                1. Faithfulness (0-10): Is every claim supported by the context below? 
                2. Completeness (0-10): Does it address the full question? 
                3. Hallucination: List any unsupported claims. 
                
                Question: {state['question']} 
                Answer: {state['answer']} 
                
                Respond in format: FAITH:X COMPLETE:Y ISSUES:... ''' 
    
    critique = llm(prompt) 
    return {'critique': critique}

#__________________________________________Build and Compile the Graph__________________________________________

# compile langgraph

# router
def router(state: LegalState) -> list | str: 
    s = state.get('strategy', 'LEGAL') 
    if s == 'LEGAL': return 'legal' 
    if s == 'DOCUMENT': return 'document' 
    return ['legal', 'document'] # BOTH — parallel


# Build and compile graph
wf = StateGraph(LegalState) 
wf.add_node('rewrite', rewrite_agent) 
wf.add_node('strategist', strategist_agent) 
wf.add_node('legal', legal_agent) 
wf.add_node('document', document_agent) 
wf.add_node('fusion', fusion_agent) 
wf.add_node('generate_answer', answer_agent) 
wf.add_node('critic', critic_agent) 

wf.set_entry_point('rewrite') 
wf.add_edge('rewrite', 'strategist') 
wf.add_conditional_edges('strategist', router, ['legal', 'document']) 
wf.add_edge('legal', 'fusion') 
wf.add_edge('document', 'fusion') 
wf.add_edge('fusion', 'generate_answer') 
wf.add_edge('generate_answer', 'critic') 

graph = wf.compile() 
print('LangGraph compiled') 


# Main entry point
def run_legal_rag(question: str, uploaded_file: str = None) -> dict:
    """
    Main entry point for the legal RAG pipeline.

    Args:
        question:      Natural language legal question
        uploaded_file: Optional path to a user-uploaded PDF

    Returns:
        Full LangGraph state dict containing:
        - answer:          Generated legal answer
        - critique:        Faithfulness/completeness evaluation
        - strategy:        Routing decision (LEGAL / DOCUMENT / BOTH)
        - final_docs:      Reranked context chunks used for the answer
        - rewritten_query: Optimised retrieval query
    """
    if uploaded_file:
        build_document_store(uploaded_file)

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

# ── Redis cache layer (sits on top of run_legal_rag) ──────────────────────────
# Add this block immediately after run_legal_rag

import redis
import json
import hashlib

_redis_client = None

def get_redis():
    """
    Lazy initialisation — only connects to Redis when first needed.
    This prevents the entire module from crashing at import time
    if REDIS_URL is not set (e.g. during local dev without Redis).
    """
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return None          # Redis not configured — silently skip caching
        try:
            _redis_client = redis.from_url(redis_url, decode_responses=True)
            _redis_client.ping()  # verify connection is alive
            print("Redis connected")
        except Exception as e:
            print(f"Redis unavailable: {e} — running without cache")
            _redis_client = None
    return _redis_client


def cached_rag(question: str, file_path: str = None) -> dict:
    """
    Drop-in replacement for run_legal_rag with Redis caching.

    - Cache key: MD5 hash of (question + file_path)
    - TTL: 1 hour (3600 seconds)
    - Cache miss: runs the full LangGraph pipeline and caches the result
    - Cache hit: returns instantly from Redis, skipping all LLM calls

    Use this in FastAPI endpoints instead of run_legal_rag directly.
    In notebooks, use run_legal_rag directly (no benefit caching one-off calls).
    """
    r = get_redis()

    # ── Generate cache key ────────────────────────────────────────────────────
    # MD5 is fine here — this is not security-sensitive, just a lookup key
    # Combining question + file_path means same question on different files
    # correctly gets different cache entries
    raw_key = f"{question}{file_path or ''}"
    cache_key = f"lexrag:{hashlib.md5(raw_key.encode()).hexdigest()}"

    # ── Check cache first ─────────────────────────────────────────────────────
    if r:
        try:
            cached = r.get(cache_key)
            if cached:
                print(f"Cache HIT — returning cached answer (key: {cache_key[:20]}...)")
                return json.loads(cached)
        except Exception as e:
            # Redis read failed — fall through to live pipeline
            # Never let a cache error break the actual RAG response
            print(f"Redis read error: {e} — falling back to live pipeline")

    # ── Cache miss — run the full pipeline ────────────────────────────────────
    print(f"Cache MISS — running LangGraph pipeline...")
    result = run_legal_rag(question, file_path)

    # ── Store result in cache ─────────────────────────────────────────────────
    # Only cache answer and strategy — final_docs contains Document objects
    # which are not JSON serialisable, so we don't cache them
    if r:
        try:
            payload = json.dumps({
                "answer":   result.get("answer", ""),
                "strategy": result.get("strategy", "LEGAL"),
                "critique": result.get("critique", ""),
                "cached":   True,    # flag so the caller knows this came from cache
            })
            r.setex(cache_key, 3600, payload)   # expire after 1 hour
            print(f"Result cached for 1 hour (key: {cache_key[:20]}...)")
        except Exception as e:
            # Redis write failed — result still returned to caller normally
            print(f"Redis write error: {e} — result not cached")

    return result


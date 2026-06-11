# backend/agents/legal_graph.py
import os, re, shutil, nltk, numpy as np, json, hashlib, requests
from pathlib import Path
from typing import List, Optional, TypedDict
from dotenv import load_dotenv


from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import StateGraph, END
from rank_bm25 import BM25Okapi


load_dotenv()
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


HF_API_KEY  = os.getenv('HUGGINGFACEHUB_API_TOKEN')          # required
HF_EMB_URL  = os.getenv(                                 # embedding model endpoint
    'HF_EMBEDDING_URL',
    'https://api-inference.huggingface.co/models/BAAI/bge-base-en-v1.5',
)
HF_RERANK_URL = os.getenv(                               # reranker endpoint
    'HF_RERANKER_URL',
    'https://api-inference.huggingface.co/models/BAAI/bge-reranker-base',
)

_HF_HEADERS = {'Authorization': f'Bearer {HF_API_KEY}', 'Content-Type': 'application/json'}


# ── HuggingFace Inference API helpers ────────────────────────────────────

def _hf_embed(texts: list[str]) -> list[list[float]]:
    """Call HF Inference API to get embeddings. Returns list of float vectors."""
    # The feature-extraction pipeline returns nested lists; handle both shapes.
    payload = {'inputs': texts, 'options': {'wait_for_model': True}}
    resp = requests.post(HF_EMB_URL, headers=_HF_HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # data shape: [[vec], [vec], ...] or [vec, vec, ...]
    if isinstance(data[0][0], list):          # shape (n, 1, dim) — squeeze
        data = [row[0] for row in data]
    # L2-normalise (BGE models recommend this)
    vecs = np.array(data, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (vecs / norms).tolist()


def _hf_rerank(query: str, passages: list[str]) -> list[float]:
    """Call HF Inference API text-ranking endpoint. Returns relevance scores."""
    payload = {
        'inputs': {
            'source_sentence': query,
            'sentences': passages,
        },
        'options': {'wait_for_model': True},
    }
    resp = requests.post(HF_RERANK_URL, headers=_HF_HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # sentence-similarity endpoint returns a flat list of scores
    return data if isinstance(data, list) else [item['score'] for item in data]


# ── LangChain-compatible embedding wrapper ───────────────────────────────

class HFInferenceEmbeddings(Embeddings):
    """Thin LangChain Embeddings wrapper around the HF Inference API."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch in chunks of 64 to stay within request-size limits
        results = []
        for i in range(0, len(texts), 64):
            results.extend(_hf_embed(texts[i:i + 64]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return _hf_embed([text])[0]


passage_emb = HFInferenceEmbeddings()


# ── LLMs ─────────────────────────────────────────────────────────────────
main_llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.2)
fast_llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0, max_tokens=10)


def llm_invoke(prompt): return main_llm.invoke(prompt).content


# ── Vector store ─────────────────────────────────────────────────────────
DB_PATH    = os.getenv('CHROMA_PERSIST_DIR', '../data/vector_database')
COLLECTION = 'legal_knowledge_v2'


legal_vs = Chroma(
    persist_directory=DB_PATH,
    embedding_function=passage_emb,
    collection_name=COLLECTION,
)


def tokenize(text):
    return [t for t in re.findall(r'\b[a-zA-Z0-9]+\b', text.lower()) if len(t) > 1]


_raw         = legal_vs.get()
legal_corpus = _raw['documents']
legal_bm25   = BM25Okapi([tokenize(c) for c in legal_corpus])
print(f'Legal knowledge base: {len(legal_corpus)} chunks')


# ── Document store (per uploaded file) ───────────────────────────────────
doc_vs = doc_bm25 = doc_corpus = None


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=['\n\n', '\n', '. ', ' ', ''],
)


def build_document_store(file_path, persist_dir='/tmp/doc_chroma'):
    global doc_vs, doc_bm25, doc_corpus
    if os.path.exists(persist_dir): shutil.rmtree(persist_dir)
    pages  = PyMuPDFLoader(file_path).load()
    chunks = []
    for page in pages:
        text = ''.join(ch for ch in page.page_content if ord(ch) < 128)
        text = re.split(r'_{2,}', text)[0].strip()
        for ct in splitter.split_text(text):
            if len(ct.strip()) > 30:
                chunks.append(Document(
                    page_content=ct,
                    metadata={**page.metadata, 'source_file': Path(file_path).name},
                ))
    doc_vs     = Chroma.from_documents(chunks, passage_emb, persist_directory=persist_dir)
    doc_corpus = [c.page_content for c in chunks]
    doc_bm25   = BM25Okapi([tokenize(c) for c in doc_corpus])
    print(f'Document store: {len(chunks)} chunks')


# ── Retrieval helpers ─────────────────────────────────────────────────────
def rrf(a, b, k=60):
    scores = {}
    for rank, d in enumerate(a): scores[d.page_content] = scores.get(d.page_content, 0) + 1 / (k + rank)
    for rank, d in enumerate(b): scores[d.page_content] = scores.get(d.page_content, 0) + 1 / (k + rank)
    return [Document(page_content=k) for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def legal_search(query, k=10):
    vec = legal_vs.similarity_search(query, k=k)
    sc  = legal_bm25.get_scores(tokenize(query))
    bm  = [Document(page_content=legal_corpus[i]) for i in np.argsort(sc)[::-1][:k]]
    return rrf(vec, bm)


def document_search(query, k=10):
    if doc_vs is None: return []
    vec = doc_vs.similarity_search(query, k=k)
    sc  = doc_bm25.get_scores(tokenize(query))
    bm  = [Document(page_content=doc_corpus[i]) for i in np.argsort(sc)[::-1][:k]]
    return rrf(vec, bm)


def rerank(query, docs, top_k=6):
    """Re-rank docs using HF Inference API instead of a local CrossEncoder."""
    if not docs: return []
    passages = [d.page_content for d in docs]
    scores   = _hf_rerank(query, passages)
    ranked   = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


# ── State ────────────────────────────────────────────────────────────────
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


# ── Agents ───────────────────────────────────────────────────────────────
def rewrite_agent(state):
    p = (f'Rewrite this query for Indian law retrieval. Return ONLY the rewritten query.\n'
         f'Original: {state["question"]}\nRewritten:')
    return {'rewritten_query': llm_invoke(p).strip()}


def strategist_agent(state):
    fc = f'\nUploaded doc: "{Path(state["uploaded_file"]).stem}"' if state.get('uploaded_file') else ''
    p  = (f'Route this question. Return ONLY: LEGAL, DOCUMENT, or BOTH\n'
          f'Question: "{state["question"]}"{fc}\n'
          f'LEGAL=Indian law, DOCUMENT=uploaded file, BOTH=needs both')
    s  = fast_llm.invoke(p).content.strip().upper()
    if   'BOTH'     in s: s = 'BOTH'
    elif 'DOCUMENT' in s: s = 'DOCUMENT'
    else:                 s = 'LEGAL'
    if s in ('DOCUMENT', 'BOTH') and not state.get('uploaded_file'): s = 'LEGAL'
    print(f'  [Strategist] {s}')
    return {'strategy': s}


def legal_agent(state):    return {'legal_docs':    legal_search(state['rewritten_query'])}
def document_agent(state): return {'document_docs': document_search(state['rewritten_query'])}


def fusion_agent(state):
    docs = (state.get('legal_docs') or []) + (state.get('document_docs') or [])
    return {'final_docs': rerank(state['rewritten_query'], docs)}


def answer_agent(state):
    ctx = '\n\n---\n\n'.join(d.page_content for d in state['final_docs'])
    p   = (f'You are a precise Indian legal assistant. Answer using ONLY the context. '
           f'Cite the specific Article or Section.\n\nContext:\n{ctx}\n\n'
           f'Question: {state["question"]}\n\nAnswer:')
    return {'answer': llm_invoke(p)}


def critic_agent(state):
    p = (f'Rate this answer: FAITH:<0-10> COMPLETE:<0-10> ISSUES:<description or None>\n'
         f'Question: {state["question"]}\nAnswer: {state["answer"]}')
    return {'critique': llm_invoke(p)}


# ── Router ────────────────────────────────────────────────────────────────
def router(state):
    s = state.get('strategy', 'LEGAL')
    if s == 'LEGAL':    return 'legal'
    if s == 'DOCUMENT': return 'document'
    return ['legal', 'document']


# ── Build graph ───────────────────────────────────────────────────────────
wf = StateGraph(LegalState)
for name, fn in [
    ('rewrite',    rewrite_agent),
    ('strategist', strategist_agent),
    ('legal',      legal_agent),
    ('document',   document_agent),
    ('fusion',     fusion_agent),
    ('answer_node',     answer_agent),
    ('critic',     critic_agent),
]:
    wf.add_node(name, fn)

wf.set_entry_point('rewrite')
wf.add_edge('rewrite', 'strategist')
wf.add_conditional_edges('strategist', router, {'legal': 'legal', 'document': 'document'})
wf.add_edge('legal',    'fusion')
wf.add_edge('document', 'fusion')
wf.add_edge('fusion',   'answer_node')
wf.add_edge('answer_node',   'critic')
wf.add_edge('critic', END)
graph = wf.compile()
print('LangGraph compiled')


# ── Redis cache layer ─────────────────────────────────────────────────────
_redis_client = None


def get_redis():
    global _redis_client
    if _redis_client is None:
        url = os.getenv('REDIS_URL')
        if not url: return None
        try:
            import redis
            _redis_client = redis.from_url(url, decode_responses=True)
            _redis_client.ping()
            print('Redis connected')
        except Exception as e:
            print(f'Redis unavailable: {e}')
    return _redis_client


# ── Public entry points ───────────────────────────────────────────────────
def run_legal_rag(question, uploaded_file=None):
    if uploaded_file: build_document_store(uploaded_file)
    return graph.invoke({
        'question':        question,
        'uploaded_file':   uploaded_file,
        'rewritten_query': None,
        'strategy':        None,
        'legal_docs':      None,
        'document_docs':   None,
        'final_docs':      None,
        'answer':          None,
        'critique':        None,
    })


def cached_rag(question, file_path=None):
    r   = get_redis()
    path_str = file_path or ''
    key = f'lexrag:{hashlib.md5(f"{question}{path_str}".encode()).hexdigest()}'
    if r:
        try:
            cached = r.get(key)
            if cached:
                print('Cache HIT')
                return json.loads(cached)
        except Exception as e:
            print(f'Redis read error: {e}')
    result = run_legal_rag(question, file_path)
    if r:
        try:
            r.setex(key, 3600, json.dumps({
                'answer':   result.get('answer', ''),
                'strategy': result.get('strategy', 'LEGAL'),
                'critique': result.get('critique', ''),
                'cached':   True,
            }))
        except Exception as e:
            print(f'Redis write error: {e}')
    return result
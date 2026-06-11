# migrate_to_pinecone.py
import os
import requests
import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

load_dotenv()

class SimpleEmbeddings:
    def embed_documents(self, texts): return [[0.0]*768 for _ in texts]

print("Connecting to your local Chroma database...")
chroma_db = Chroma(
    persist_directory=r'C:\Users\user\Desktop\ML_DL_projects\RAG Projects\Legal RAG 1\data\vector_database', 
    collection_name='legal_knowledge_v2', 
    embedding_function=SimpleEmbeddings()
)

raw = chroma_db.get()
documents = raw['documents']
metadatas = raw['metadatas']
ids = raw['ids']
total = len(documents)
print(f"Loaded {total} chunks from local Chroma. Upserting to Pinecone...")

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_INDEX_HOST")

if PINECONE_HOST and not PINECONE_HOST.startswith("https://"):
    PINECONE_HOST = f"https://{PINECONE_HOST}"

# Initialize the official client using your working authenticated token
hf_client = InferenceClient(token=HF_TOKEN)
pc_headers = {
    "Api-Key": PINECONE_API_KEY,
    "Content-Type": "application/json"
}

# Process in blocks of 64
batch_size = 64
for i in range(0, total, batch_size):
    b_docs = documents[i:i+batch_size]
    b_meta = metadatas[i:i+batch_size]
    b_ids = ids[i:i+batch_size]
    
    try:
        # Use the client interface to fetch BGE embeddings natively
        vectors = hf_client.feature_extraction(
            text=b_docs,
            model="BAAI/bge-base-en-v1.5"
        )
        
        # Handle shape squeezing if returned as nested matrices
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        if isinstance(vectors[0], list) and isinstance(vectors[0][0], list):
            vectors = [v[0] for v in vectors]
            
        # L2 Normalization explicitly for BGE expectations
        vecs = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = (vecs / norms).tolist()
        
    except Exception as hf_error:
        print(f"\nHugging Face Inference API failed at batch {i}: {hf_error}")
        break

    payload = []
    for doc, meta, idx, vec in zip(b_docs, b_meta, b_ids, vectors):
        meta_clean = {k: str(v) for k, v in meta.items()} if meta else {}
        meta_clean["text"] = doc
        payload.append({"id": idx, "values": vec, "metadata": meta_clean})
        
    try:
        pc_res = requests.post(f"{PINECONE_HOST}/vectors/upsert", headers=pc_headers, json={"vectors": payload}, timeout=30)
        pc_res.raise_for_status()
    except Exception as pc_error:
        print(f"\nPinecone Upsert failed at batch {i}: {pc_error}")
        break
        
    print(f"Progress: {i + len(b_docs)} / {total} chunks synchronized successfully.")

print("\nProcess finalized.")
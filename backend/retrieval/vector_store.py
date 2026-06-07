import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def get_production_vector_store(embeddings):
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY']) 
    INDEX_NAME = 'lexrag'

    # create index if it doesnt exist
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]: 
        pc.create_index( 
            name=INDEX_NAME, 
            dimension=768, # BGE-Base embedding dimension 
            metric='cosine', 
            spec=ServerlessSpec(cloud='aws', region='us-east-1') 
            ) 
        
        print(f'Created Pinecone index: {INDEX_NAME}') 
    
    return PineconeVectorStore( 
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        pinecone_api_key=os.environ['PINECONE_API_KEY'] )


# One-time migration from ChromaDB to Pinecone: 
# from langchain_chroma import Chroma 
# chroma_vs = Chroma(persist_directory='../data/vector_database', ...) 
# docs = chroma_vs.get(include=['documents', 'metadatas']) 
# pinecone_vs.add_texts(docs['documents'], metadatas=docs['metadatas'])
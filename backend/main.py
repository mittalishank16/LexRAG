import os, shutil, uuid 
from pathlib import Path 
from contextlib import asynccontextmanager 
from typing import Optional 

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel 

from agents.legal_graph import run_legal_rag, build_document_store, cached_rag
from agents.contract_agent import extract_contract_data 
from db.supabase_client import get_supabase 
from scheduler import scheduler 

UPLOAD_DIR = Path('/tmp/uploads') 
UPLOAD_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    #STARTUP
    scheduler.start()
    print('Scheduler started')
    yield
    # shutdown
    scheduler.shutdown()

app = FastAPI(
    title= 'LexRAG API',
    description= 'Production Indian Legal RAG System',
    version= '1.0.0',
    lifespan= lifespan
)

app.add_middleware( 
    CORSMiddleware, 
    allow_origins=['https://lexrag.vercel.app', 'http://localhost:3000'], 
    allow_credentials=True, 
    allow_methods=['*'], 
    allow_headers=['*'], 
    )

# # ── Request/Response Models ─────────────────────────────────────────────

class ChatRequest(BaseModel): 
    question: str 
    session_id: Optional[str] = None 
    file_path: Optional[str] = None 
    
class ChatResponse(BaseModel): 
    answer: str 
    critique: str 
    strategy: str 
    context_used: list[str] 
    
class ContractUploadResponse(BaseModel): 
    contract_id: str 
    party_a: str 
    party_b: str 
    renewal_date: str 
    message: str


# Health check

@app.get('/health') 
async def health(): 
    return {'status': 'ok', 'service': 'lexrag-api'}

# chat endpoint

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # Use cached_rag instead of run_legal_rag
        # Identical interface — just faster on repeated questions
        result = cached_rag(
            question=req.question,
            file_path=req.file_path
        )
        return ChatResponse(
            answer=result.get("answer", ""),
            critique=result.get("critique", ""),
            strategy=result.get("strategy", "LEGAL"),
            context_used=[
                d.page_content[:300]
                for d in result.get("final_docs", [])
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Document Upload

@app.post('/upload') 
async def upload_document(file: UploadFile = File(...)): 
    if not file.filename.lower().endswith('.pdf'): 
        raise HTTPException(400, detail='Only PDF files accepted') 
    
    file_id = str(uuid.uuid4()) 
    file_path = UPLOAD_DIR / f'{file_id}.pdf' 
    with open(file_path, 'wb') as f: 
        shutil.copyfileobj(file.file, f) 
    build_document_store(str(file_path)) 
    return {'file_path': str(file_path), 'file_id': file_id}

# contract analysis

@app.post('/contracts/analyze', response_model=ContractUploadResponse) 
async def analyze_contract(file: UploadFile = File(...), user_id: str = 'default'): 
    file_path = UPLOAD_DIR / f'{uuid.uuid4()}.pdf' 
    with open(file_path, 'wb') as f: 
        shutil.copyfileobj(file.file, f) 
    
    data = extract_contract_data(str(file_path)) 
    data['user_id'] = user_id 
    data['filename'] = file.filename 
    
    supabase = get_supabase() 
    res = supabase.table('contracts').insert(data).execute() 
    
    return ContractUploadResponse( 
        contract_id=res.data[0]['id'], 
        party_a=data.get('party_a', 'Unknown'), 
        party_b=data.get('party_b', 'Unknown'), 
        renewal_date=str(data.get('renewal_date', 'Not found')), 
        message='Contract analyzed and reminders scheduled' 
        )

@app.get('/contracts') 
async def list_contracts(user_id: str = 'default'): 
    supabase = get_supabase() 
    res = supabase.table('contracts').select('*').eq('user_id', user_id).execute() 
    return res.data

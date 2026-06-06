import json, re 
from datetime import date, timedelta 
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_groq import ChatGroq 

llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)

EXTRACT_PROMPT = '''You are a legal contract analyst. Extract the following from the contract text. 
                Return ONLY valid JSON with these exact keys: 
                
                { "party_a": "full name of first party", 
                "party_b": "full name of second party", 
                "contract_type": "e.g. Service Agreement, Lease, Employment, NDA", 
                "start_date": "YYYY-MM-DD or null", 
                "end_date": "YYYY-MM-DD or null", 
                "renewal_date": "YYYY-MM-DD or null (same as end_date if auto-renews)", 
                "notice_period_days": 30, "auto_renewal": true or false, 
                "governing_law": "jurisdiction", "key_obligations": 
                "brief 2-3 sentence summary of main obligations" 
                } 
                
Contract text: 
{text}

JSON only, no markdown:'''

def extract_contract_data(file_path: str) -> dict: 
    loader = PyMuPDFLoader(file_path) 
    pages = loader.load() 
    
    # Use first 6 pages (usually enough for key dates) 
    text = '\n'.join(p.page_content for p in pages[:6]) 
    text = ''.join(ch for ch in text if ord(ch) < 128)[:8000] 
    
    prompt = EXTRACT_PROMPT.replace('{text}', text) 
    response = llm.invoke(prompt).content.strip() 
    
    # Clean up LLM response (strip markdown fences if present) 
    response = re.sub(r'```(?:json)?|```', '', response).strip() 
    
    data = json.loads(response) 
    return data
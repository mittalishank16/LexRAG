@echo off
call conda create -n lexrag python=3.11.9 -y
call conda activate lexrag
call pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
call pip install tokenizers==0.19.1 huggingface-hub==0.24.6 transformers==4.44.2 accelerate==0.33.0 sentence-transformers==3.1.1 datasets==2.21.0
call pip install langchain==0.3.1 langchain-core==0.3.6 langchain-community==0.3.1 langchain-text-splitters==0.3.0 langchain-huggingface==0.1.0 langchain-groq==0.2.1 langchain-chroma==0.1.4 langchain-pinecone==0.2.0 langchain-experimental==0.3.1 langgraph==0.2.28
call pip install chromadb==0.5.11 pinecone-client==5.0.1 rank-bm25==0.2.2 nltk==3.9.1
call pip install pymupdf==1.24.10 pypdf==4.3.1
call pip install fastapi==0.115.0 uvicorn[standard]==0.32.0 python-multipart==0.0.12 pydantic==2.9.2 supabase==2.7.4 redis==5.1.1
call pip install apscheduler==3.10.4 sendgrid==6.11.0
call pip install ragas==0.2.3 groq==0.11.0 scikit-learn==1.5.2 numpy==1.26.4 pandas==2.2.2 python-dotenv==1.0.1 tqdm==4.66.5
call python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
call pip install jupyter ipykernel notebook
call python -m ipykernel install --user --name lexrag --display-name "LexRAG (Python 3.11)"
call python -c "import torch, transformers, sentence_transformers, langchain, langgraph, chromadb, fastapi, ragas; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('All imports OK')"
echo Setup complete!
pause
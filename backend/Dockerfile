FROM python:3.11-slim

WORKDIR /app 

# Install system dependencies 
RUN apt-get update && apt-get install -y \ 
    libmupdf-dev \ 
    && rm -rf /var/lib/apt/lists/* 
    
# Update line 10 to this:
COPY ./requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 

# Download NLTK data 
RUN python -c "import nltk; nltk.download('punkt')" 

COPY . . 

# Pre-download models at build time (faster cold starts) 
RUN python -c "from sentence_transformers import SentenceTransformer; \ 
    SentenceTransformer('BAAI/bge-base-en-v1.5')" 
    
EXPOSE 8000 


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
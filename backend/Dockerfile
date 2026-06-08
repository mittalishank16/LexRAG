FROM python:3.11-slim

WORKDIR /app 

# Install system dependencies 
RUN apt-get update && apt-get install -y \ 
    libmupdf-dev \ 
    && rm -rf /var/lib/apt/lists/* # Copy requirements file from the build context (root directory)
COPY ./requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 

# Download NLTK data 
RUN python -c "import nltk; nltk.download('punkt')" 

# ── ONNX Model Reconstruction (Bypasses GitHub & Render Memory Limits) ──

# 1. Ensure the target directory structure exists inside the container filesystem
RUN mkdir -p models/inlegalbert_onnx

# 2. Copy the tokenizer configurations and split parts from your workspace
COPY models/inlegalbert_onnx/ /app/models/inlegalbert_onnx/

# 3. Concatenate the split parts back into the single functional model.onnx binary
RUN cat /app/models/inlegalbert_onnx/model_quantized.onnx.part* > /app/models/inlegalbert_onnx/model_quantized.onnx

# 4. Clean up the leftover part files inside the container to optimize space
RUN rm /app/models/inlegalbert_onnx/model_quantized.onnx.part*

# ────────────────────────────────────────────────────────────────────────

# Copy the rest of the application code
COPY . . 

# Pre-download general models at build time (faster cold starts) 
RUN python -c "from sentence_transformers import SentenceTransformer; \ 
    SentenceTransformer('BAAI/bge-base-en-v1.5')" 
    
EXPOSE 8000 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

WORKDIR /app 

# Install system dependencies 
RUN apt-get update && apt-get install -y \ 
    libmupdf-dev \ 
    && rm -rf /var/lib/apt/lists/* 
    
COPY ./requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 

# Download NLTK data 
RUN python -c "import nltk; nltk.download('punkt')" 

# ── ONNX Model Reconstruction (Bypasses Size and Memory Bounds) ──
RUN mkdir -p models/inlegalbert_onnx models/bge_onnx

COPY models/inlegalbert_onnx/ /app/models/inlegalbert_onnx/
COPY models/bge_onnx/ /app/models/bge_onnx/

RUN cat /app/models/inlegalbert_onnx/model_quantized.onnx.part* > /app/models/inlegalbert_onnx/model_quantized.onnx
RUN cat /app/models/bge_onnx/model_quantized.onnx.part* > /app/models/bge_onnx/model_quantized.onnx

RUN rm -f /app/models/inlegalbert_onnx/model_quantized.onnx.part*
RUN rm -f /app/models/bge_onnx/model_quantized.onnx.part*
# ──────────────────────────────────────────────────────────────────

COPY . . 

EXPOSE 8000 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
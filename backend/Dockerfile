FROM python:3.11-slim

WORKDIR /app 

# tell python to treat /app as the root module directory
ENV PYTHONPATH=/app

# Install system dependencies 
RUN apt-get update && apt-get install -y \ 
    libmupdf-dev \ 
    && rm -rf /var/lib/apt/lists/* 
    
COPY ./requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 

# Download NLTK data 
RUN python -c "import nltk; nltk.download('punkt')" 

# ── ONNX Model Reconstruction (Bypasses Size and Memory Bounds) ──
RUN mkdir -p models/bge_onnx 
# models/inlegalbert_onnx

#COPY models/inlegalbert_onnx/ /app/models/inlegalbert_onnx/
COPY models/bge_onnx/ /app/models/bge_onnx/

# Use sequential appending to stream bytes instead of bundling with cat
#RUN for f in /app/models/inlegalbert_onnx/model_quantized.onnx.part*; do cat "$f" >> /app/models/inlegalbert_onnx/model_quantized.onnx; rm "$f"; done
RUN for f in /app/models/bge_onnx/model_quantized.onnx.part*; do cat "$f" >> /app/models/bge_onnx/model_quantized.onnx; rm "$f"; done

#RUN rm -f /app/models/inlegalbert_onnx/model_quantized.onnx.part*
#RUN rm -f /app/models/bge_onnx/model_quantized.onnx.part*
# ──────────────────────────────────────────────────────────────────

COPY . . 

EXPOSE 8000 

CMD ["sh", "-c", "PYTHONPATH=. uvicorn main:app --host 0.0.0.0 --port $PORT"]

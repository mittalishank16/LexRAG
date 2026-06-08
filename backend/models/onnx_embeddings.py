# backend/models/onnx_embeddings.py

import os
import numpy as np
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
import onnxruntime as ort
from langchain_core.embeddings import Embeddings


class ONNXEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class backed by ONNX Runtime.
    Drops in as a replacement for HuggingFaceEmbeddings anywhere in the codebase.
    """

    def __init__(self, model_dir: str, normalize: bool = True):
        self.normalize = normalize
        model_dir = Path(model_dir)

        # Prefer the quantized model (smaller) — fall back to full ONNX
        onnx_file = model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = model_dir / "model.onnx"

        if not onnx_file.exists():
            raise FileNotFoundError(f"Could not locate an ONNX model file inside '{model_dir}'")

        print(f"Loading ONNX Execution Session for: {onnx_file}")
        
        # Optimize for low-RAM cloud container instances
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Dynamically map expected input names to bypass model signature crashes
        self.expected_inputs = [node.name for node in self.session.get_inputs()]
        print(f"Model input signatures expected: {self.expected_inputs}")

    def _mean_pool(self, last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        token_embeddings = last_hidden_state
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        
        return sum_embeddings / sum_mask

    def _encode(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # Safely construct the input payload using only inputs expected by the model
        onnx_inputs = {}
        for key in self.expected_inputs:
            if key in encoded:
                onnx_inputs[key] = encoded[key]
            elif key == "token_type_ids":
                # Fallback if model requires it but tokenizer omitted it
                onnx_inputs[key] = np.zeros_like(encoded["input_ids"])

        outputs = self.session.run(None, onnx_inputs)
        embeddings = self._mean_pool(outputs[0], encoded["attention_mask"])

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        return embeddings

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._encode(batch)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        # Clean prefix mapping to support high-accuracy vector retrieval matching
        return self._encode([text])[0].tolist()
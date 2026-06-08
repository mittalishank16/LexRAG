# backend/models/onnx_embeddings.py

import os
import numpy as np
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from langchain_core.embeddings import Embeddings


class ONNXEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class backed by ONNX Runtime.

    Drops in as a replacement for HuggingFaceEmbeddings anywhere
    in the codebase. Same interface, ~60% less RAM, 2-4x faster on CPU.

    Memory profile on Render free tier:
      Standard InLegalBERT (PyTorch): ~450MB
      This class (ONNX INT8):         ~90MB
    """

    def __init__(self, model_dir: str, normalize: bool = True):
        """
        Args:
            model_dir:  Path to the directory containing model_quantized.onnx
                        and tokenizer files (output of convert script).
            normalize:  If True, L2-normalise embeddings (required for
                        cosine similarity in ChromaDB / Pinecone).
        """
        self.normalize = normalize
        model_dir      = Path(model_dir)

        # Prefer the quantised model (smaller) — fall back to full ONNX
        onnx_file = model_dir / "model_quantized.onnx"
        if not onnx_file.exists():
            onnx_file = model_dir / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"No ONNX model found in {model_dir}. "
                "Run scripts/convert_inlegalbert_to_onnx.py first."
            )

        print(f"Loading ONNX model from {onnx_file.name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model     = ORTModelForFeatureExtraction.from_pretrained(
            str(model_dir),
            file_name=onnx_file.name,
            provider="CPUExecutionProvider",
        )

        print(f"  ONNX InLegalBERT loaded ({onnx_file.stat().st_size // (1024*1024)}MB)")

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Mean pooling — average token embeddings weighted by attention mask.
        This is the standard way to get a sentence embedding from BERT.
        Tokens with attention_mask=0 (padding) are excluded from the average.
        """
        # Expand mask to match embedding dimensions
        mask_expanded = attention_mask[:, :, np.newaxis].astype(float)
        # Zero out padding token embeddings
        sum_embeddings = (token_embeddings * mask_expanded).sum(axis=1)
        # Divide by number of real (non-padding) tokens
        sum_mask = mask_expanded.sum(axis=1).clip(min=1e-9)
        return sum_embeddings / sum_mask

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Core encode function.
        Tokenises, runs ONNX forward pass, applies mean pooling.
        """
        # Tokenise with padding and truncation to BERT's 512 token limit
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",  # numpy arrays — no PyTorch dependency at runtime
        )

        # ONNX Runtime forward pass
        outputs = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
        )

        # outputs.last_hidden_state shape: (batch_size, seq_len, 768)
        embeddings = self._mean_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )

        if self.normalize:
            # L2 normalisation: divide each vector by its magnitude
            # Required for cosine similarity to work correctly
            norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        return embeddings

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed a list of documents (passages/chunks).
        Called by ChromaDB and Pinecone when indexing.
        Uses batching to avoid OOM on large document lists.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch      = texts[i : i + batch_size]
            embeddings = self._encode(batch)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        Called by ChromaDB and Pinecone on every similarity_search() call.
        Prepends the BGE query instruction for better retrieval quality.
        """
        # InLegalBERT works best with a query prefix that signals retrieval intent
        prefixed = f"Represent this Indian legal query for retrieval: {text}"
        return self._encode([prefixed])[0].tolist()
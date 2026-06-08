# scripts/convert_inlegalbert_to_onnx.py
# Run once locally: python scripts/convert_inlegalbert_to_onnx.py
#
# What this does:
# 1. Downloads law-ai/InLegalBERT from HuggingFace
# 2. Traces the forward pass with a dummy input
# 3. Exports the traced graph to ONNX format
# 4. Quantises to INT8 (halves size again: 180MB → ~90MB)
# 5. Saves to backend/models/inlegalbert_onnx/

import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from transformers import AutoTokenizer

MODEL_ID   = "law-ai/InLegalBERT"
OUTPUT_DIR = Path("backend/models/inlegalbert_onnx")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Converting {MODEL_ID} to ONNX...")
print("Step 1/3: Exporting to ONNX (this downloads the model if not cached)...")

# Export to ONNX — optimum handles tracing automatically
model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_ID,
    export=True,                    # True = convert from PyTorch to ONNX
    provider="CPUExecutionProvider" # always CPU — no CUDA on Render
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Save the ONNX model and tokenizer
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"  Saved ONNX model to {OUTPUT_DIR}")

# ── Step 2: Quantise to INT8 ─────────────────────────────────────────────
# INT8 quantisation reduces model size by ~50% with <1% accuracy loss
# on embedding tasks (the weights are rounded from float32 to int8)
print("Step 2/3: Quantising to INT8...")

quantizer = ORTQuantizer.from_pretrained(str(OUTPUT_DIR))
qconfig   = AutoQuantizationConfig.avx512_vnni(
    is_static=False,    # dynamic quantisation — no calibration dataset needed
    per_channel=False,
)
quantizer.quantize(
    save_dir=str(OUTPUT_DIR),
    quantization_config=qconfig,
)
print(f"  Quantised model saved")

# ── Step 3: Verify the output ────────────────────────────────────────────
print("Step 3/3: Verifying output files...")
for f in sorted(OUTPUT_DIR.iterdir()):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name:<40} {size_mb:.1f} MB")

print(f"\nConversion complete. Output: {OUTPUT_DIR.resolve()}")
print("Next: commit backend/models/inlegalbert_onnx/ to git")
print("      or upload to HuggingFace Hub as a private model")
# scripts/convert_bge_to_onnx.py
# Run once: python scripts/convert_bge_to_onnx.py

from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

MODEL_ID   = "BAAI/bge-base-en-v1.5"
OUTPUT_DIR = Path("backend/models/bge_onnx")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Converting {MODEL_ID} to ONNX...")

model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_ID,
    export=True,
    provider="CPUExecutionProvider",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("Quantising to INT8...")
quantizer = ORTQuantizer.from_pretrained(str(OUTPUT_DIR))
qconfig   = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir=str(OUTPUT_DIR), quantization_config=qconfig)

print("Files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name:<40} {f.stat().st_size // (1024*1024)}MB")

print("Done.")
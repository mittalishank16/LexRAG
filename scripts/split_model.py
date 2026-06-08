# scripts/split_model.py
# Run once locally: python scripts/split_model.py
# Splits model_quantized.onnx into 3 parts of ~35MB each
# Parts are committed to git normally (each under 100MB limit)

import os
from pathlib import Path

MODEL_PATH  = Path("backend/models/inlegalbert_onnx/model_quantized.onnx")
OUTPUT_DIR  = Path("backend/models/inlegalbert_onnx")
NUM_PARTS   = 3

def split_model(model_path: Path, output_dir: Path, num_parts: int) -> None:
    model_bytes = model_path.read_bytes()
    total_size  = len(model_bytes)
    part_size   = total_size // num_parts

    print(f"Model size  : {total_size / (1024*1024):.1f} MB")
    print(f"Parts       : {num_parts}")
    print(f"Part size   : {part_size / (1024*1024):.1f} MB each")
    print()

    for i in range(num_parts):
        start = i * part_size
        # Last part gets any remaining bytes (handles non-divisible sizes)
        end   = total_size if i == num_parts - 1 else (i + 1) * part_size

        part_path  = output_dir / f"model_quantized.onnx.part{i+1}"
        part_bytes = model_bytes[start:end]
        part_path.write_bytes(part_bytes)

        print(f"  Written: {part_path.name} ({len(part_bytes)/(1024*1024):.1f} MB)")

    print(f"\nDone. Delete the original before committing:")
    print(f"  rm {model_path}")
    print(f"  git add {output_dir}/model_quantized.onnx.part*")

split_model(MODEL_PATH, OUTPUT_DIR, NUM_PARTS)
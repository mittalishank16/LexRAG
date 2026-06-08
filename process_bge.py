# process_bge.py
import os
import sys
import subprocess
from pathlib import Path

bge_dir = Path("backend/models/bge_onnx")
target_onnx = bge_dir / "model_quantized.onnx"
output_part_prefix = bge_dir / "model_quantized.onnx.part"

print("=======================================================================")
print("END-TO-END ONNX EXPORT AND CHUNKING PIPELINE")
print("=======================================================================")

# ── STEP 1: CHECK IF SPLIT PARTS ALREADY EXIST ──────────────────────────────
if list(bge_dir.glob("model_quantized.onnx.part*")):
    print(f"Split chunks already exist inside '{bge_dir}'.")
    print("If you want to re-export, delete the .part* files and run this script again.\n")
    sys.exit(0)

# ── STEP 2: DYNAMIC ONNX EXPORT (IF NOT ALREADY EXPORTED) ───────────────────
if not target_onnx.exists():
    print(f"'{target_onnx.name}' not found. Initiating live ONNX export pipeline...")
    
    # Target directory generation
    bge_dir.mkdir(parents=True, exist_ok=True)
    
    # We invoke the exporter using python -m to bypass Windows PATH limitations safely
    export_cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", "BAAI/bge-base-en-v1.5",
        "--task", "feature-extraction",
        "--optimize", "O2",
        str(bge_dir)
    ]
    
    try:
        print(f"Running command: {' '.join(export_cmd)}")
        subprocess.run(export_cmd, check=True)
        print("ONNX export sequence successful.")
    except subprocess.CalledProcessError as e:
        print(f"\nFatal: ONNX exporter encountered an error: {e}")
        print("Ensure you have run 'pip install optimum[onnxruntime]' inside your active env.")
        sys.exit(1)

    # Automatically rename the output from optimum's version matching schemas
    source_model_optimized = bge_dir / "model_optimized.onnx"
    source_model_base = bge_dir / "model.onnx"
    
    if source_model_optimized.exists():
        os.rename(source_model_optimized, target_onnx)
    elif source_model_base.exists():
        os.rename(source_model_base, target_onnx)

# ── STEP 3: REUSABLE CHUNKING ROUTINE WITH DYNAMIC SUFFIX CALCULATION ────────
if not target_onnx.exists():
    print(f"Fatal Error: Could not locate compiled ONNX asset to slice inside {bge_dir}")
    sys.exit(1)

chunk_size = 40 * 1024 * 1024  # Strict 40MB chunks to clear GitHub and cloud systems safely
print(f"\nSlicing '{target_onnx.name}' into 40MB upload-safe chunks...")

try:
    with open(target_onnx, "rb") as f:
        chunk_num = 0
        
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
                
            # Programmatically calculate alphabetical suffixes (aa, ab, ac... az, ba, bb...)
            # This handles infinite extensions gracefully without index bounds crashes
            first_char = chr(ord('a') + (chunk_num // 26))
            second_char = chr(ord('a') + (chunk_num % 26))
            suffix = f"{first_char}{second_char}"
            
            part_name = f"{output_part_prefix}{suffix}"
            with open(part_name, "wb") as chunk_file:
                chunk_file.write(chunk_data)
                
            print(f"Created: {Path(part_name).name} ({len(chunk_data)/(1024*1024):.2f} MB)")
            chunk_num += 1

    print(f"\nSUCCESS! BGE-Base split components generated inside: '{bge_dir}'")
    print("You can now push your repository changes safely to GitHub.")

except Exception as e:
    print(f"Error during file binary chunk slicing routine: {e}")
    sys.exit(1)
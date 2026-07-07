#!/usr/bin/env python3
"""
v3/setup.py — Download base models then export ONNX INT8 models.

This is a thin wrapper; all ONNX export logic lives in optimized/setup.py.
Run from the repo root (or let the run scripts call it via the parent):

  python setup.py               # download base models (root)
  python optimized/setup.py     # export wav2vec2 + ByT5 to ONNX INT8
  python v3/app.py              # launch v3 demo on port 7862
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    root_setup = ROOT / "setup.py"
    ort_setup  = ROOT / "optimized" / "setup.py"

    if not root_setup.exists():
        print("[error] Root setup.py not found. Are you in the right repo?")
        sys.exit(1)

    print("=== Step 1: Download base models ===")
    subprocess.check_call([sys.executable, str(root_setup)])

    print("\n=== Step 2: Export + INT8-quantize ONNX models ===")
    subprocess.check_call([sys.executable, str(ort_setup)])

    print("\n=== Setup complete. Launch v3 with: ===")
    print("  v3\\run.bat  (Windows)")
    print("  ./v3/run.sh  (Linux/Mac)")


if __name__ == "__main__":
    main()

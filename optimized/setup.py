#!/usr/bin/env python3
"""
optimized/setup.py — Export + INT8-quantize ASR and correction models for ONNX Runtime.

Run AFTER the root setup.py has downloaded the base models:

  python setup.py                   # download base models (root)
  python optimized/setup.py         # export + quantize to ONNX INT8

Reads from:  ../models/{wav2vec2-asr, byt5-correction}/
Writes to:   ../models-onnx/{wav2vec2-asr, byt5-correction}/

Quantizes MatMul + Gemm ops only (attention / FFN layers).
Skipping Conv avoids the weight-normalization issue in wav2vec2's positional
conv embedding, while still covering >90% of inference FLOPs.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_SRC = ROOT / "models"
MODELS_ORT = ROOT / "models-onnx"


def _check_deps():
    missing = []
    for pkg in ("optimum", "onnxruntime"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("[error] Missing packages:", " ".join(missing))
        print("        Run: pip install optimum[onnxruntime] onnxruntime")
        sys.exit(1)


def _check_src(name: str):
    if not (MODELS_SRC / name / ".download_complete").exists():
        print(f"[error] Base model '{name}' not found in {MODELS_SRC}")
        print("        Run: python setup.py   (from repo root)")
        sys.exit(1)


def _copy_config(src_dir: Path, dst_dir: Path):
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix in (".json", ".txt", ".model", ".spm") \
                and not (dst_dir / f.name).exists():
            shutil.copy2(f, dst_dir)


def _quantize_onnx(fp32_path: Path, int8_path: Path):
    """Quantize a single ONNX file: MatMul + Gemm only (skips Conv)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"  INT8: {fp32_path.name} -> {int8_path.name}")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=False,
        reduce_range=False,
    )


def export_ctc(name: str):
    src = MODELS_SRC / name
    dst = MODELS_ORT / name
    sentinel = dst / ".ort_complete"
    if sentinel.exists():
        print(f"[skip]  {name}  (already done)")
        return

    from optimum.onnxruntime import ORTModelForCTC

    fp32_dir = dst / "_fp32"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[export] {name} -> ONNX FP32 ...")
    model = ORTModelForCTC.from_pretrained(str(src), export=True)
    model.save_pretrained(str(fp32_dir))
    _copy_config(src, fp32_dir)

    print(f"[quant]  {name} -> INT8 ...")
    fp32_onnx = fp32_dir / "model.onnx"
    if not fp32_onnx.exists():
        # some versions export with a different name
        candidates = list(fp32_dir.glob("*.onnx"))
        if not candidates:
            print(f"[error] No ONNX file found in {fp32_dir}")
            sys.exit(1)
        fp32_onnx = candidates[0]
    _quantize_onnx(fp32_onnx, dst / "model_int8.onnx")
    _copy_config(fp32_dir, dst)

    shutil.rmtree(fp32_dir)
    sentinel.touch()
    print(f"[done]   {name}")


def export_seq2seq(name: str):
    src = MODELS_SRC / name
    dst = MODELS_ORT / name
    sentinel = dst / ".ort_complete"
    if sentinel.exists():
        print(f"[skip]  {name}  (already done)")
        return

    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    fp32_dir = dst / "_fp32"
    fp32_dir.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[export] {name} -> ONNX FP32 ...")
    model = ORTModelForSeq2SeqLM.from_pretrained(str(src), export=True)
    model.save_pretrained(str(fp32_dir))
    _copy_config(src, fp32_dir)

    print(f"[quant]  {name} -> INT8 (all components) ...")
    onnx_files = sorted(fp32_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"[error] No ONNX files found in {fp32_dir}")
        sys.exit(1)
    for fp32_path in onnx_files:
        stem = fp32_path.stem  # e.g. "encoder_model"
        _quantize_onnx(fp32_path, dst / f"{stem}_int8.onnx")
    _copy_config(fp32_dir, dst)

    shutil.rmtree(fp32_dir)
    sentinel.touch()
    print(f"[done]   {name}")


def main():
    _check_deps()
    _check_src("wav2vec2-asr")
    _check_src("byt5-correction")

    MODELS_ORT.mkdir(exist_ok=True)
    print(f"Writing ONNX INT8 models to {MODELS_ORT}\n")

    export_ctc("wav2vec2-asr")
    print()
    export_seq2seq("byt5-correction")

    print("\nDone. Launch optimized demo with:")
    print("  optimized\\run.bat  (Windows)")
    print("  ./optimized/run.sh  (Linux/Mac)")


if __name__ == "__main__":
    main()

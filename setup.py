#!/usr/bin/env python3
"""
setup.py — Download and cache all models to ./models/ for offline use.
Run once before launching app.py.

  HF_TOKEN=hf_xxx python setup.py
  python setup.py          # prompts for token

Models downloaded:
  palli23/pyannote-icelandic-joint           -> models/pyannote-segmentation/
  pyannote/wespeaker-voxceleb-resnet34-LM    -> models/pyannote-embedding/
  palli23/wav2vec2-base-samromur-300h         -> models/wav2vec2-asr/
  mideind/yfirlestur-icelandic-correction-byt5 -> models/byt5-correction/

Note: pyannote models require accepting terms at huggingface.co before downloading.
"""
import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

MODELS = {
    "pyannote-segmentation": ("palli23/pyannote-icelandic-joint", []),
    "pyannote-embedding": ("pyannote/wespeaker-voxceleb-resnet34-LM", []),
    "wav2vec2-asr": ("palli23/wav2vec2-icelandic-multi-aug-podcast-v3", ["checkpoint-*/"]),
    "byt5-correction": ("mideind/yfirlestur-icelandic-correction-byt5", []),
}


def main() -> None:
    try:
        from huggingface_hub import snapshot_download, login
    except ImportError:
        print("[error] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token = input(
            "HuggingFace token (required for gated pyannote models — "
            "see huggingface.co/settings/tokens): "
        ).strip()
    if not token:
        print("[error] Token required for pyannote models.")
        sys.exit(1)

    login(token=token)

    for name, (repo_id, extra_ignore) in MODELS.items():
        local_dir = MODELS_DIR / name
        sentinel = local_dir / ".download_complete"
        if sentinel.exists():
            print(f"[skip]  {repo_id}  (already cached)")
            continue

        print(f"[fetch] {repo_id} -> {local_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"] + extra_ignore,
            )
            sentinel.touch()
            print(f"[done]  {repo_id}")
        except Exception as exc:
            print(f"[error] {repo_id}: {exc}")
            print(
                "        If this is a gated model, accept the terms at "
                f"https://huggingface.co/{repo_id}"
            )
            sys.exit(1)

    print("\nAll models ready. Launch the demo with:  run.bat  (Windows) or  ./run.sh  (Linux/Mac)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
app.py — Icelandic speaker diarization + ASR + text correction.

Pipeline:
  1. pyannote speaker diarization on full audio (sequential — needs global speaker clustering)
  2. wav2vec2 CTC ASR on each segment in parallel (ThreadPoolExecutor)
  3. ByT5 text correction on each segment in parallel (ThreadPoolExecutor)

All models loaded from ./models/ — no network access at runtime.
Run setup.py once first to download them.
"""
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

MODELS_DIR = Path(__file__).parent / "models"

SAMPLE_RATE = 16_000
MIN_SEG_DURATION = 1.5
MIN_SEG_SAMPLES = 1600       # wav2vec2 hard floor (~0.1s)
CORRECTION_MAX_LEN = 512
DIARIZATION_PARAMS = {
    "segmentation": {"min_duration_off": 0.0, "threshold": 0.4442333},
    "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
}

# Workers = half the cores; each worker gets the other half for its PyTorch ops.
# Total threads used = N_WORKERS * THREADS_PER_WORKER ≈ n_cpu.
_N_CPU = os.cpu_count() or 1
N_WORKERS = max(1, _N_CPU // 2)
THREADS_PER_WORKER = max(1, _N_CPU // N_WORKERS)

# ── Model loading ─────────────────────────────────────────────────────────────

def _check_models() -> None:
    missing = [
        name for name in ("pyannote-segmentation", "pyannote-embedding", "wav2vec2-asr", "byt5-correction")
        if not (MODELS_DIR / name / ".download_complete").exists()
    ]
    if missing:
        print("[error] Missing models:", ", ".join(missing))
        print("        Run:  python setup.py")
        sys.exit(1)


def load_all_models():
    from pyannote.audio import Model
    from pyannote.audio.pipelines import SpeakerDiarization

    print("Loading diarization segmentation model...")
    seg_model = Model.from_pretrained(str(MODELS_DIR / "pyannote-segmentation"))

    print("Loading speaker embedding model...")
    emb_model = Model.from_pretrained(str(MODELS_DIR / "pyannote-embedding"))

    print("Building diarization pipeline...")
    dia_pipeline = SpeakerDiarization(
        segmentation=seg_model,
        segmentation_step=0.3,      # default 0.1 → ~10x fewer windows at 0.3
        segmentation_batch_size=32,  # process windows in batches
        embedding=emb_model,
        embedding_batch_size=32,
        embedding_exclude_overlap=True,
    )
    dia_pipeline.instantiate(DIARIZATION_PARAMS)

    print("Loading ASR model...")
    asr_processor = Wav2Vec2Processor.from_pretrained(str(MODELS_DIR / "wav2vec2-asr"))
    asr_model = Wav2Vec2ForCTC.from_pretrained(str(MODELS_DIR / "wav2vec2-asr"))
    asr_model.eval()

    print("Loading text correction model...")
    corr_tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "byt5-correction"))
    corr_model = T5ForConditionalGeneration.from_pretrained(str(MODELS_DIR / "byt5-correction"))
    corr_model.eval()

    print(f"All models ready.  Workers: {N_WORKERS} x {THREADS_PER_WORKER} threads each.\n")
    return dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model

# ── Audio helpers ─────────────────────────────────────────────────────────────

def _to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def _fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

# ── Per-segment inference (runs inside thread pool) ───────────────────────────

def _asr(audio: np.ndarray, processor, model) -> str:
    torch.set_num_threads(THREADS_PER_WORKER)
    if len(audio) < MIN_SEG_SAMPLES:
        return ""
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.decode(pred_ids[0]).strip()


def _correct(text: str, tokenizer, model) -> str:
    torch.set_num_threads(THREADS_PER_WORKER)
    if not text.strip():
        return text
    inputs = tokenizer(text, return_tensors="pt", max_length=CORRECTION_MAX_LEN, truncation=True)
    with torch.inference_mode():
        out = model.generate(**inputs, max_length=CORRECTION_MAX_LEN, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _process_segment(args: tuple) -> tuple:
    """ASR + correction for one segment. Runs in a worker thread."""
    idx, start, end, speaker, seg_audio, asr_processor, asr_model, corr_tokenizer, corr_model = args
    text = _asr(seg_audio, asr_processor, asr_model)
    text = _correct(text, corr_tokenizer, corr_model)
    return idx, start, end, speaker, text

# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    audio_path: str,
    dia_pipeline,
    asr_processor,
    asr_model,
    corr_tokenizer,
    corr_model,
    use_diarization: bool = True,
    skip_short: bool = False,
    min_duration_off: float = 0.0,
    cluster_threshold: float = 0.6,
    progress=gr.Progress(),
) -> tuple[str, str, str | None]:
    if not audio_path:
        return "No audio uploaded.", "", None

    t_total = time.time()

    progress(0.05, desc="Loading audio...")
    audio, sr = sf.read(audio_path, dtype="float32")
    audio_16k = _to_mono_16k(audio, sr)
    audio_duration = len(audio_16k) / SAMPLE_RATE

    # ── Step 1: Diarize ───────────────────────────────────────────────────────
    t_dia = 0.0
    speaker_map: dict[str, str] = {}
    label_iter = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    segments: list[tuple] = []  # (start, end, speaker_label, seg_audio)

    if use_diarization:
        dia_pipeline.instantiate({
            "segmentation": {"min_duration_off": min_duration_off, "threshold": 0.4442333},
            "clustering": {"threshold": cluster_threshold, "Fa": 0.07, "Fb": 0.8},
        })
        progress(0.1, desc="Running diarization...")
        t0 = time.time()
        waveform = torch.from_numpy(audio_16k).unsqueeze(0)
        dia_result = dia_pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        t_dia = time.time() - t0

        annotation = (
            dia_result.speaker_diarization if hasattr(dia_result, "speaker_diarization")
            else dia_result.diarization if hasattr(dia_result, "diarization")
            else dia_result
        )

        for turn, _, raw_spk in annotation.itertracks(yield_label=True):
            if raw_spk not in speaker_map:
                speaker_map[raw_spk] = f"SPEAKER_{next(label_iter, raw_spk)}"
            start, end = turn.start, turn.end
            if skip_short and (end - start) < MIN_SEG_DURATION:
                continue
            s, e = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
            segments.append((start, end, speaker_map[raw_spk], audio_16k[s:e]))

        if not segments:
            return "No speech detected.", "", None
    else:
        segments = [(0.0, audio_duration, None, audio_16k)]

    # ── Steps 2+3: ASR + correction in parallel ───────────────────────────────
    progress(0.3, desc=f"Transcribing {len(segments)} segments in parallel...")
    t_asr_corr = time.time()

    work = [
        (i, start, end, spk, seg, asr_processor, asr_model, corr_tokenizer, corr_model)
        for i, (start, end, spk, seg) in enumerate(segments)
    ]

    results = [None] * len(work)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_process_segment, args): args[0] for args in work}
        done = 0
        for future in as_completed(futures):
            idx, start, end, speaker, text = future.result()
            results[idx] = (start, end, speaker, text)
            done += 1
            progress(0.3 + 0.65 * done / len(work), desc=f"Done {done}/{len(work)} segments...")

    t_asr_corr = time.time() - t_asr_corr

    # ── Format output ─────────────────────────────────────────────────────────
    lines = []
    for start, end, speaker, text in results:
        if not text:
            continue
        label = f"{speaker}: " if speaker else ""
        lines.append(f"[{_fmt_ts(start)}-{_fmt_ts(end)}] {label}{text}")

    t_total = time.time() - t_total
    rtf = t_total / audio_duration if audio_duration > 0 else 0.0

    metrics_parts = [
        f"Audio: {audio_duration:.1f}s",
        f"RTF: {rtf:.2f}x",
        f"Workers: {N_WORKERS}",
        f"Diarization: {t_dia:.1f}s" if use_diarization else "Diarization: off",
        f"ASR+Correction: {t_asr_corr:.1f}s",
        f"Speakers: {len(speaker_map)}" if use_diarization else "",
        f"Segments: {len(lines)}",
    ]
    metrics = " | ".join(p for p in metrics_parts if p)

    transcript = "\n".join(lines)
    progress(0.98, desc="Saving...")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(transcript)
    tmp.close()

    progress(1.0, desc="Done.")
    return transcript, metrics, tmp.name

# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model):
    def _run(audio_path, use_diarization, skip_short, min_duration_off, cluster_threshold, progress=gr.Progress()):
        try:
            return run_pipeline(
                audio_path, dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model,
                use_diarization, skip_short, min_duration_off, cluster_threshold, progress
            )
        except Exception:
            import traceback
            return traceback.format_exc(), "", None

    with gr.Blocks(title="Icelandic Speech Transcription") as demo:
        gr.Markdown("# Icelandic Speech Transcription")
        gr.Markdown("Upload an audio file to get a timestamped transcript.")

        audio_in = gr.Audio(type="filepath", label="Audio file (.wav, .mp3, .flac, .ogg)")
        run_btn = gr.Button("Transcribe", variant="primary", size="lg")

        with gr.Row():
            use_dia_cb = gr.Checkbox(label="Speaker diarization", value=True)
            skip_short_cb = gr.Checkbox(label=f"Skip short segments (< {MIN_SEG_DURATION}s)", value=False)

        with gr.Accordion("Diarization settings", open=False):
            min_dur_off_sl = gr.Slider(
                minimum=0.0, maximum=2.0, step=0.1, value=0.0,
                label="Min. silence to split segment (s) — higher = fewer, longer segments",
            )
            cluster_thresh_sl = gr.Slider(
                minimum=0.3, maximum=0.9, step=0.05, value=0.6,
                label="Speaker clustering threshold — lower = more speakers, higher = fewer",
            )

        metrics_out = gr.Textbox(lines=1, label="Metrics", interactive=False)
        transcript_out = gr.Textbox(
            lines=25,
            label="Transcript",
            placeholder="[MM:SS.ss-MM:SS.ss] SPEAKER_A: text here ...",
        )
        download_out = gr.File(label="Download .txt")

        run_btn.click(
            fn=_run,
            inputs=[audio_in, use_dia_cb, skip_short_cb, min_dur_off_sl, cluster_thresh_sl],
            outputs=[transcript_out, metrics_out, download_out],
        )

    return demo


def main() -> None:
    print(f"CPU cores: {_N_CPU}  |  Workers: {N_WORKERS}  |  Threads/worker: {THREADS_PER_WORKER}")
    torch.set_num_threads(THREADS_PER_WORKER)
    torch.set_num_interop_threads(N_WORKERS)
    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)

    _check_models()
    models = load_all_models()
    demo = build_ui(*models)
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()

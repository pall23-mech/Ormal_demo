#!/usr/bin/env python3
"""
optimized/app.py — Icelandic ASR + diarization demo, ONNX INT8 edition.

Changes vs ../app.py:
  - wav2vec2 CTC:  Wav2Vec2ForCTC      -> ORTModelForCTC        (ONNX INT8)
  - ByT5:          T5ForConditionalGen  -> ORTModelForSeq2SeqLM  (ONNX INT8)
  - pyannote:      unchanged (PyTorch)
  - outputs:       TXT + SRT + VTT
  - batch mode:    drop multiple files, processed sequentially

Run optimized/setup.py once first to export + quantize the models.
"""
import os
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from onnxruntime import SessionOptions
from transformers import AutoTokenizer, Wav2Vec2Processor

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_ORT = Path(__file__).parent.parent / "models-onnx"

SAMPLE_RATE = 16_000
MIN_SEG_DURATION = 1.5
MIN_SEG_SAMPLES = 1600
CORRECTION_MAX_LEN = 512
DIARIZATION_PARAMS = {
    "segmentation": {"min_duration_off": 0.0, "threshold": 0.4442333},
    "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
}

_N_CPU = os.cpu_count() or 1
N_WORKERS = max(1, _N_CPU // 2)
THREADS_PER_WORKER = 2


def _ort_session_options() -> SessionOptions:
    opts = SessionOptions()
    opts.intra_op_num_threads = THREADS_PER_WORKER
    opts.inter_op_num_threads = 1
    return opts


# ── Model loading ─────────────────────────────────────────────────────────────

def _check_models():
    missing_base = [
        n for n in ("pyannote-segmentation", "pyannote-embedding")
        if not (MODELS_DIR / n / ".download_complete").exists()
    ]
    missing_ort = [
        n for n in ("wav2vec2-asr", "byt5-correction")
        if not (MODELS_ORT / n / ".ort_complete").exists()
    ]
    if missing_base:
        print("[error] Missing base models:", ", ".join(missing_base))
        print("        Run: python setup.py")
        sys.exit(1)
    if missing_ort:
        print("[error] Missing ONNX models:", ", ".join(missing_ort))
        print("        Run: python optimized/setup.py")
        sys.exit(1)


def load_all_models():
    from optimum.onnxruntime import ORTModelForCTC, ORTModelForSeq2SeqLM
    from pyannote.audio import Model
    from pyannote.audio.pipelines import SpeakerDiarization

    print("Loading diarization segmentation model...")
    seg_model = Model.from_pretrained(str(MODELS_DIR / "pyannote-segmentation"))

    print("Loading speaker embedding model...")
    emb_model = Model.from_pretrained(str(MODELS_DIR / "pyannote-embedding"))

    print("Building diarization pipeline...")
    dia_pipeline = SpeakerDiarization(
        segmentation=seg_model,
        segmentation_step=0.3,
        segmentation_batch_size=32,
        embedding=emb_model,
        embedding_batch_size=32,
        embedding_exclude_overlap=True,
    )
    dia_pipeline.instantiate(DIARIZATION_PARAMS)

    print("Loading ASR processor...")
    asr_processor = Wav2Vec2Processor.from_pretrained(str(MODELS_DIR / "wav2vec2-asr"))

    print("Loading ASR model (ONNX INT8)...")
    asr_model = ORTModelForCTC.from_pretrained(
        str(MODELS_ORT / "wav2vec2-asr"),
        file_name="model_int8.onnx",
        session_options=_ort_session_options(),
    )

    print("Loading correction tokenizer...")
    corr_tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "byt5-correction"))

    print("Loading correction model (ONNX INT8)...")
    byt5_ort_dir = MODELS_ORT / "byt5-correction"
    _enc = "encoder_model_int8.onnx"
    _dec = "decoder_model_int8.onnx"
    _dec_past = "decoder_with_past_model_int8.onnx"
    corr_model = ORTModelForSeq2SeqLM.from_pretrained(
        str(byt5_ort_dir),
        encoder_file_name=_enc,
        decoder_file_name=_dec,
        decoder_with_past_file_name=_dec_past if (byt5_ort_dir / _dec_past).exists() else _dec,
        session_options=_ort_session_options(),
    )

    print(f"All models ready.  Workers: {N_WORKERS} x {THREADS_PER_WORKER} threads.\n")
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


def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _vtt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ── Subtitle formatters ───────────────────────────────────────────────────────

def _build_txt(segments: list[tuple]) -> str:
    lines = []
    for start, end, speaker, text in segments:
        label = f"{speaker}: " if speaker else ""
        lines.append(f"[{_fmt_ts(start)}-{_fmt_ts(end)}] {label}{text}")
    return "\n".join(lines)


def _build_srt(segments: list[tuple]) -> str:
    blocks = []
    for i, (start, end, speaker, text) in enumerate(segments, 1):
        label = f"{speaker}: " if speaker else ""
        blocks.append(f"{i}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{label}{text}")
    return "\n\n".join(blocks)


def _build_vtt(segments: list[tuple]) -> str:
    blocks = ["WEBVTT\n"]
    for start, end, speaker, text in segments:
        label = f"{speaker}: " if speaker else ""
        blocks.append(f"{_vtt_ts(start)} --> {_vtt_ts(end)}\n{label}{text}")
    return "\n\n".join(blocks)


# ── Per-segment inference ─────────────────────────────────────────────────────

def _asr(audio: np.ndarray, processor, model) -> str:
    if len(audio) < MIN_SEG_SAMPLES:
        return ""
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.decode(pred_ids[0]).strip()


def _correct(text: str, tokenizer, model) -> str:
    if not text.strip():
        return text
    inputs = tokenizer(text, return_tensors="pt", max_length=CORRECTION_MAX_LEN, truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=CORRECTION_MAX_LEN, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _process_segment(args: tuple) -> tuple:
    idx, start, end, speaker, seg_audio, asr_processor, asr_model, corr_tokenizer, corr_model = args
    text = _asr(seg_audio, asr_processor, asr_model)
    text = _correct(text, corr_tokenizer, corr_model)
    return idx, start, end, speaker, text


# ── Core pipeline (one file) ──────────────────────────────────────────────────

def _transcribe_file(
    audio_path: str,
    dia_pipeline,
    asr_processor,
    asr_model,
    corr_tokenizer,
    corr_model,
    use_diarization: bool,
    skip_short: bool,
    min_duration_off: float,
    cluster_threshold: float,
) -> tuple[list[tuple], dict]:
    """Returns (segments_with_text, stats_dict)."""
    t_total = time.time()
    audio, sr = sf.read(audio_path, dtype="float32")
    audio_16k = _to_mono_16k(audio, sr)
    audio_duration = len(audio_16k) / SAMPLE_RATE

    t_dia = 0.0
    speaker_map: dict[str, str] = {}
    label_iter = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    raw_segments: list[tuple] = []

    if use_diarization:
        dia_pipeline.instantiate({
            "segmentation": {"min_duration_off": min_duration_off, "threshold": 0.4442333},
            "clustering": {"threshold": cluster_threshold, "Fa": 0.07, "Fb": 0.8},
        })
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
            raw_segments.append((start, end, speaker_map[raw_spk], audio_16k[s:e]))
    else:
        raw_segments = [(0.0, audio_duration, None, audio_16k)]

    if not raw_segments:
        return [], {"duration": audio_duration, "rtf": 0, "dia": t_dia, "asr": 0, "speakers": 0, "segments": 0}

    t_asr = time.time()
    work = [
        (i, start, end, spk, seg, asr_processor, asr_model, corr_tokenizer, corr_model)
        for i, (start, end, spk, seg) in enumerate(raw_segments)
    ]
    results = [None] * len(work)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_process_segment, args): args[0] for args in work}
        for future in as_completed(futures):
            idx, start, end, speaker, text = future.result()
            results[idx] = (start, end, speaker, text)
    t_asr = time.time() - t_asr

    segments = [(s, e, spk, txt) for s, e, spk, txt in results if txt]
    t_total = time.time() - t_total
    stats = {
        "duration": audio_duration,
        "rtf": t_total / audio_duration if audio_duration else 0,
        "dia": t_dia,
        "asr": t_asr,
        "speakers": len(speaker_map),
        "segments": len(segments),
    }
    return segments, stats


def _write_outputs(segments: list[tuple], stem: str, fmt: str) -> str:
    """Write transcript in the chosen format to a temp file, return path."""
    if fmt == "SRT":
        content, suffix = _build_srt(segments), ".srt"
    elif fmt == "VTT":
        content, suffix = _build_vtt(segments), ".vtt"
    else:
        content, suffix = _build_txt(segments), ".txt"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, prefix=f"{stem}_",
        delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


# ── Gradio pipeline wrappers ──────────────────────────────────────────────────

def run_single(
    audio_path, use_diarization, skip_short, min_dur_off, cluster_thresh, fmt,
    dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model,
    progress=gr.Progress(),
):
    if not audio_path:
        return "No audio uploaded.", "", None
    try:
        progress(0.05, desc="Loading audio...")
        segments, stats = _transcribe_file(
            audio_path, dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model,
            use_diarization, skip_short, min_dur_off, cluster_thresh,
        )
        progress(0.95, desc="Formatting...")
        transcript = _build_txt(segments)
        out_path = _write_outputs(segments, Path(audio_path).stem, fmt)
        metrics = (
            f"Audio: {stats['duration']:.1f}s | RTF: {stats['rtf']:.2f}x | "
            f"Workers: {N_WORKERS} | Diarization: {stats['dia']:.1f}s | "
            f"ASR+Correction: {stats['asr']:.1f}s | Speakers: {stats['speakers']} | "
            f"Segments: {stats['segments']}"
        )
        progress(1.0, desc="Done.")
        return transcript, metrics, out_path
    except Exception:
        import traceback
        return traceback.format_exc(), "", None


def run_batch(
    file_list, use_diarization, skip_short, min_dur_off, cluster_thresh, fmt,
    dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model,
    progress=gr.Progress(),
):
    if not file_list:
        return "No files uploaded.", None

    results_log = []
    tmp_dir = Path(tempfile.mkdtemp())
    out_files = []

    for i, file_obj in enumerate(file_list):
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        stem = Path(path).stem
        progress((i) / len(file_list), desc=f"Processing {stem} ({i+1}/{len(file_list)})...")
        try:
            segments, stats = _transcribe_file(
                path, dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model,
                use_diarization, skip_short, min_dur_off, cluster_thresh,
            )
            out_path = _write_outputs(segments, stem, fmt)
            out_files.append((stem, out_path))
            results_log.append(
                f"{stem}: {stats['duration']:.0f}s | RTF {stats['rtf']:.2f}x | "
                f"{stats['segments']} segments"
            )
        except Exception as exc:
            results_log.append(f"{stem}: ERROR — {exc}")

    progress(0.95, desc="Packing zip...")
    if fmt == "SRT":
        ext = ".srt"
    elif fmt == "VTT":
        ext = ".vtt"
    else:
        ext = ".txt"

    zip_path = str(tmp_dir / "transcripts.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for stem, path in out_files:
            zf.write(path, arcname=f"{stem}{ext}")

    log = "\n".join(results_log)
    progress(1.0, desc="Done.")
    return log, zip_path


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model):
    shared_models = (dia_pipeline, asr_processor, asr_model, corr_tokenizer, corr_model)

    with gr.Blocks(title="Icelandic Speech Transcription (ONNX)") as demo:
        gr.Markdown("# Icelandic Speech Transcription")

        with gr.Row():
            use_dia_cb = gr.Checkbox(label="Speaker diarization", value=True)
            skip_short_cb = gr.Checkbox(label=f"Skip short segments (< {MIN_SEG_DURATION}s)", value=False)
            fmt_dd = gr.Dropdown(choices=["TXT", "SRT", "VTT"], value="TXT", label="Output format")

        with gr.Accordion("Diarization settings", open=False):
            min_dur_off_sl = gr.Slider(0.0, 2.0, step=0.1, value=0.0,
                label="Min. silence to split segment (s)")
            cluster_thresh_sl = gr.Slider(0.3, 0.9, step=0.05, value=0.6,
                label="Speaker clustering threshold")

        with gr.Tabs():
            with gr.Tab("Single file"):
                audio_in = gr.Audio(type="filepath", label="Audio file (.wav .mp3 .flac .ogg)")
                run_btn = gr.Button("Transcribe", variant="primary", size="lg")
                metrics_out = gr.Textbox(lines=1, label="Metrics", interactive=False)
                transcript_out = gr.Textbox(lines=22, label="Transcript",
                    placeholder="[MM:SS.ss-MM:SS.ss] SPEAKER_A: ...")
                download_out = gr.File(label="Download")

                run_btn.click(
                    fn=lambda *a, progress=gr.Progress(): run_single(*a, *shared_models, progress=progress),
                    inputs=[audio_in, use_dia_cb, skip_short_cb, min_dur_off_sl, cluster_thresh_sl, fmt_dd],
                    outputs=[transcript_out, metrics_out, download_out],
                )

            with gr.Tab("Batch"):
                batch_in = gr.File(label="Audio files", file_count="multiple",
                    file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"])
                batch_btn = gr.Button("Transcribe all", variant="primary", size="lg")
                batch_log = gr.Textbox(lines=10, label="Batch log", interactive=False)
                batch_zip = gr.File(label="Download all (.zip)")

                batch_btn.click(
                    fn=lambda *a, progress=gr.Progress(): run_batch(*a, *shared_models, progress=progress),
                    inputs=[batch_in, use_dia_cb, skip_short_cb, min_dur_off_sl, cluster_thresh_sl, fmt_dd],
                    outputs=[batch_log, batch_zip],
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
    demo.launch(server_name="0.0.0.0", server_port=7861, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
v3/app.py — Icelandic ASR + diarization, feature-complete edition.

Builds on ONNX INT8 models from optimized/.  New in v3:
  - Hotword/vocabulary injection  (fuzzy post-correction, rapidfuzz)
  - VAD pre-filter                (energy-based, offline, strips silence/music)
  - Custom speaker names          (rename SPEAKER_A -> "Kári" globally)
  - Group consecutive same-speaker turns
  - Transcript formatted with speaker-name headers
  - Interactive editor            (click segment -> plays audio, edit inline)
  - Word-level confidence highlight (CTC softmax, colour-coded HTML)
  - Number / date normalisation   (Icelandic word-to-digit pass)
  - Batch mode with zip download
  - Output: TXT, SRT, VTT
"""
import os
import re
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from onnxruntime import SessionOptions
from transformers import AutoTokenizer, Wav2Vec2Processor

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_ORT  = Path(__file__).parent.parent / "models-onnx"

SAMPLE_RATE        = 16_000
MIN_SEG_DURATION   = 1.5
MIN_SEG_SAMPLES    = 1600
CORRECTION_MAX_LEN = 512
DIARIZATION_PARAMS = {
    "segmentation": {"min_duration_off": 0.0, "threshold": 0.4442333},
    "clustering":   {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8},
}

_N_CPU           = os.cpu_count() or 1
N_WORKERS        = max(1, _N_CPU // 2)
THREADS_PER_WORKER = 2

CONF_HIGH = 0.85   # no highlight
CONF_MED  = 0.60   # amber; below this -> red


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    start:     float
    end:       float
    speaker:   str
    text:      str
    raw_words: list = field(default_factory=list)  # [(word, conf), ...]

    @property
    def conf_avg(self) -> float:
        if not self.raw_words:
            return 1.0
        return sum(c for _, c in self.raw_words) / len(self.raw_words)


# ── Hotword injection ─────────────────────────────────────────────────────────

def apply_hotwords(text: str, hotwords: list[str], threshold: int = 78) -> str:
    if not hotwords or not text.strip():
        return text
    from rapidfuzz import process, fuzz
    words = text.split()
    out = []
    for word in words:
        suffix = ""
        clean  = word
        if word and word[-1] in ".,!?:;":
            suffix = word[-1]
            clean  = word[:-1]
        match = process.extractOne(clean, hotwords, scorer=fuzz.ratio,
                                   score_cutoff=threshold)
        out.append((match[0] if match else clean) + suffix)
    return " ".join(out)


# ── Number / date normalisation ───────────────────────────────────────────────

_NUM_RULES = [
    (r"\bnúll\b",                                    "0"),
    (r"\b(?:einn|eitt|ein)\b",                       "1"),
    (r"\b(?:tveir|tvær|tvö)\b",                      "2"),
    (r"\b(?:þrír|þrjár|þrjú)\b",                    "3"),
    (r"\b(?:fjórir|fjórar|fjögur)\b",                "4"),
    (r"\bfimm\b",                                    "5"),
    (r"\bsex\b",                                     "6"),
    (r"\bsjö\b",                                     "7"),
    (r"\bátta\b",                                    "8"),
    (r"\bníu\b",                                     "9"),
    (r"\btíu\b",                                     "10"),
    (r"\bellefu\b",                                  "11"),
    (r"\btólf\b",                                    "12"),
    (r"\bþrettán\b",                                 "13"),
    (r"\bfjórtán\b",                                 "14"),
    (r"\bfimmtán\b",                                 "15"),
    (r"\bsextán\b",                                  "16"),
    (r"\bsautján\b",                                 "17"),
    (r"\bátján\b",                                   "18"),
    (r"\bnítján\b",                                  "19"),
    (r"\btuttugu\b",                                 "20"),
    (r"\bþrjátíu\b",                                 "30"),
    (r"\bfjörtíu\b",                                 "40"),
    (r"\bfimmtíu\b",                                 "50"),
    (r"\bsextíu\b",                                  "60"),
    (r"\bsjötíu\b",                                  "70"),
    (r"\báttatíu\b",                                 "80"),
    (r"\bnítíu\b",                                   "90"),
    (r"\bhundrað\b",                                 "100"),
    (r"\bþúsund\b",                                  "1.000"),
    (r"\bmilljón\b",                                 "1.000.000"),
    (r"\bmilljarður\b",                              "1.000.000.000"),
    # ordinals
    (r"\bfyrst[iu]?\b",                              "1."),
    (r"\b(?:annar|önnur|annað)\b",                   "2."),
    (r"\bþriðj[iu]?\b",                              "3."),
    (r"\bfjórð[iu]?\b",                              "4."),
    (r"\bfimmt[iu]?\b",                              "5."),
    (r"\bsjött[iu]?\b",                              "6."),
    (r"\bsjöund[iu]?\b",                             "7."),
    (r"\báttund[iu]?\b",                             "8."),
    (r"\bnífund[iu]?\b",                             "9."),
    (r"\btíund[iu]?\b",                              "10."),
]

def normalize_numbers(text: str) -> str:
    for pat, rep in _NUM_RULES:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


# ── Energy VAD ────────────────────────────────────────────────────────────────

def vad_segments(audio: np.ndarray, sr: int,
                 frame_ms: int = 25, hop_ms: int = 10,
                 min_speech_s: float = 0.3, gap_s: float = 0.4) -> list[tuple[float, float]]:
    fl, hl = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    rms = np.array([
        np.sqrt(np.mean(audio[i:i + fl] ** 2))
        for i in range(0, len(audio) - fl, hl)
    ])
    nz  = rms[rms > 1e-7]
    thr = np.percentile(nz, 25) * 4 if len(nz) else 0.01
    speech = rms > thr

    segs, in_s, t_start = [], False, 0.0
    for i, s in enumerate(speech):
        t = i * hl / sr
        if s and not in_s:
            t_start, in_s = t, True
        elif not s and in_s:
            if t - t_start >= min_speech_s:
                segs.append([t_start, t])
            in_s = False
    if in_s:
        segs.append([t_start, len(audio) / sr])

    merged = []
    for seg in segs:
        if merged and seg[0] - merged[-1][1] < gap_s:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return [(s, e) for s, e in merged] or [(0.0, len(audio) / sr)]


# ── ASR with word-level confidence ────────────────────────────────────────────

def _ctc_words(logits_2d: torch.Tensor, processor) -> list[tuple[str, float]]:
    probs    = torch.softmax(logits_2d, dim=-1)
    pred_ids = torch.argmax(logits_2d, dim=-1)
    blank    = processor.tokenizer.pad_token_id

    # CTC collapse
    tokens, prev = [], None
    for t in range(len(pred_ids)):
        tid = pred_ids[t].item()
        if tid != blank and tid != prev:
            tok  = processor.tokenizer.convert_ids_to_tokens([tid])[0]
            conf = probs[t, tid].item()
            tokens.append((tok, conf))
        prev = tid

    # group by '|' word boundary
    words, cur_chars, cur_confs = [], [], []
    for tok, conf in tokens:
        if tok == "|":
            if cur_chars:
                words.append(("".join(cur_chars), min(cur_confs)))
                cur_chars, cur_confs = [], []
        else:
            cur_chars.append(tok)
            cur_confs.append(conf)
    if cur_chars:
        words.append(("".join(cur_chars), min(cur_confs)))
    return words


def _asr(audio: np.ndarray, processor, model) -> tuple[str, list[tuple[str, float]]]:
    if len(audio) < MIN_SEG_SAMPLES:
        return "", []
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    words = _ctc_words(logits, processor)
    return " ".join(w for w, _ in words), words


def _correct(text: str, tokenizer, model) -> str:
    if not text.strip():
        return text
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=CORRECTION_MAX_LEN, truncation=True)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=CORRECTION_MAX_LEN, num_beams=2)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _process_segment(args: tuple) -> tuple:
    (idx, start, end, speaker, seg_audio,
     asr_proc, asr_model, corr_tok, corr_model,
     hotwords, do_norm) = args
    raw_text, raw_words = _asr(seg_audio, asr_proc, asr_model)
    corrected = _correct(raw_text, corr_tok, corr_model)
    if hotwords:
        corrected = apply_hotwords(corrected, hotwords)
    if do_norm:
        corrected = normalize_numbers(corrected)
    return idx, start, end, speaker, corrected, raw_words


# ── Speaker turn grouping ─────────────────────────────────────────────────────

def group_turns(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    out = [Segment(segments[0].start, segments[0].end,
                   segments[0].speaker, segments[0].text,
                   list(segments[0].raw_words))]
    for seg in segments[1:]:
        last = out[-1]
        if seg.speaker == last.speaker:
            last.end       = seg.end
            last.text      = last.text.rstrip() + " " + seg.text.lstrip()
            last.raw_words = last.raw_words + seg.raw_words
        else:
            out.append(Segment(seg.start, seg.end, seg.speaker,
                               seg.text, list(seg.raw_words)))
    return out


# ── Output formatters ─────────────────────────────────────────────────────────

def _ts(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:05.2f}"

def _srt_ts(s: float) -> str:
    h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}".replace(".", ",")

def _vtt_ts(s: float) -> str:
    h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def build_txt(segments: list[Segment], header_format: bool = True) -> str:
    lines = []
    for seg in segments:
        if not seg.text:
            continue
        if header_format:
            spk = seg.speaker or "SPEAKER"
            lines.append(f"\n{spk}  [{_ts(seg.start)} - {_ts(seg.end)}]")
            lines.append(seg.text)
        else:
            label = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"[{_ts(seg.start)}-{_ts(seg.end)}] {label}{seg.text}")
    return "\n".join(lines).strip()


def build_srt(segments: list[Segment]) -> str:
    blocks = []
    idx = 1
    for seg in segments:
        if not seg.text:
            continue
        label = f"{seg.speaker}: " if seg.speaker else ""
        blocks.append(f"{idx}\n{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}\n{label}{seg.text}")
        idx += 1
    return "\n\n".join(blocks)


def build_vtt(segments: list[Segment]) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        if not seg.text:
            continue
        label = f"{seg.speaker}: " if seg.speaker else ""
        blocks.append(f"{_vtt_ts(seg.start)} --> {_vtt_ts(seg.end)}\n{label}{seg.text}")
    return "\n\n".join(blocks)


def build_conf_html(segments: list[Segment]) -> str:
    parts = ["<div style='font-family:monospace;font-size:0.88em;line-height:1.9'>"]
    for seg in segments:
        if not seg.raw_words:
            continue
        spk  = f"<b style='color:#444'>{seg.speaker}</b> " if seg.speaker else ""
        ts_s = f"<span style='color:#999;font-size:0.8em'>[{_ts(seg.start)}]</span> "
        ws   = []
        for word, conf in seg.raw_words:
            color = ("inherit" if conf >= CONF_HIGH
                     else "#e6a817" if conf >= CONF_MED
                     else "#e63946")
            ws.append(f'<span style="color:{color}" title="{conf:.2f}">{word}</span>')
        parts.append(f"<p>{spk}{ts_s}{' '.join(ws)}</p>")
    parts.append("</div>")
    return "\n".join(parts)


def segments_to_df(segments: list[Segment]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Start":   _ts(s.start),
        "End":     _ts(s.end),
        "Speaker": s.speaker or "",
        "Text":    s.text,
        "Conf":    f"{s.conf_avg:.2f}",
    } for s in segments])


def df_to_segments(df: pd.DataFrame, orig: list[Segment]) -> list[Segment]:
    out = []
    for i, row in df.iterrows():
        base = orig[i] if i < len(orig) else Segment(0.0, 0.0, "", "")
        out.append(Segment(
            start=base.start, end=base.end,
            speaker=str(row.get("Speaker", base.speaker) or "").strip(),
            text=str(row.get("Text", base.text) or "").strip(),
            raw_words=base.raw_words,
        ))
    return out


def _write_file(segments: list[Segment], stem: str, fmt: str, hdr: bool) -> str:
    if fmt == "SRT":
        content, ext = build_srt(segments), ".srt"
    elif fmt == "VTT":
        content, ext = build_vtt(segments), ".vtt"
    else:
        content, ext = build_txt(segments, header_format=hdr), ".txt"
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=ext, prefix=stem + "_", delete=False, encoding="utf-8"
    )
    f.write(content); f.close()
    return f.name


# ── Model loading ─────────────────────────────────────────────────────────────

def _ort_opts() -> SessionOptions:
    opts = SessionOptions()
    opts.intra_op_num_threads = THREADS_PER_WORKER
    opts.inter_op_num_threads = 1
    return opts


def _check_models():
    miss_base = [n for n in ("pyannote-segmentation", "pyannote-embedding")
                 if not (MODELS_DIR / n / ".download_complete").exists()]
    miss_ort  = [n for n in ("wav2vec2-asr", "byt5-correction")
                 if not (MODELS_ORT / n / ".ort_complete").exists()]
    if miss_base:
        print("[error] Base models missing. Run: python setup.py"); sys.exit(1)
    if miss_ort:
        print("[error] ONNX models missing. Run: python optimized/setup.py"); sys.exit(1)


def load_all_models():
    from optimum.onnxruntime import ORTModelForCTC, ORTModelForSeq2SeqLM
    from pyannote.audio import Model
    from pyannote.audio.pipelines import SpeakerDiarization

    print("Loading pyannote segmentation...")
    seg = Model.from_pretrained(str(MODELS_DIR / "pyannote-segmentation"))
    print("Loading pyannote embedding...")
    emb = Model.from_pretrained(str(MODELS_DIR / "pyannote-embedding"))
    print("Building diarization pipeline...")
    dia = SpeakerDiarization(
        segmentation=seg, segmentation_step=0.3, segmentation_batch_size=32,
        embedding=emb, embedding_batch_size=32, embedding_exclude_overlap=True,
    )
    dia.instantiate(DIARIZATION_PARAMS)

    print("Loading ASR processor...")
    asr_proc = Wav2Vec2Processor.from_pretrained(str(MODELS_DIR / "wav2vec2-asr"))
    print("Loading ASR model (ONNX INT8)...")
    asr_model = ORTModelForCTC.from_pretrained(
        str(MODELS_ORT / "wav2vec2-asr"),
        file_name="model_int8.onnx",
        session_options=_ort_opts(),
    )

    print("Loading correction tokenizer...")
    corr_tok = AutoTokenizer.from_pretrained(str(MODELS_DIR / "byt5-correction"))
    print("Loading correction model (ONNX INT8)...")
    byt5_dir = MODELS_ORT / "byt5-correction"
    _dp = "decoder_with_past_model_int8.onnx"
    corr_model = ORTModelForSeq2SeqLM.from_pretrained(
        str(byt5_dir),
        encoder_file_name="encoder_model_int8.onnx",
        decoder_file_name="decoder_model_int8.onnx",
        decoder_with_past_file_name=_dp if (byt5_dir / _dp).exists() else "decoder_model_int8.onnx",
        session_options=_ort_opts(),
    )
    print(f"Ready.  Workers: {N_WORKERS} x {THREADS_PER_WORKER} threads.\n")
    return dia, asr_proc, asr_model, corr_tok, corr_model


# ── Core pipeline ─────────────────────────────────────────────────────────────

def transcribe_file(
    audio_path: str,
    dia_pipeline, asr_proc, asr_model, corr_tok, corr_model,
    use_diarization: bool = True,
    use_vad: bool = False,
    skip_short: bool = False,
    min_dur_off: float = 0.0,
    cluster_thresh: float = 0.6,
    do_group: bool = True,
    hotwords: list[str] = [],
    do_norm: bool = False,
    progress_cb=None,
) -> tuple[list[Segment], dict]:
    t0 = time.time()
    audio_np, sr = sf.read(audio_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)
    if sr != SAMPLE_RATE:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=SAMPLE_RATE)
    dur = len(audio_np) / SAMPLE_RATE

    t_dia, speaker_map, raw_segs = 0.0, {}, []
    label_iter = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    if use_diarization:
        dia_pipeline.instantiate({
            "segmentation": {"min_duration_off": min_dur_off, "threshold": 0.4442333},
            "clustering":   {"threshold": cluster_thresh, "Fa": 0.07, "Fb": 0.8},
        })
        if progress_cb:
            progress_cb(0.1, "Diarizing...")
        td = time.time()
        wf = torch.from_numpy(audio_np).unsqueeze(0)
        dia_out = dia_pipeline({"waveform": wf, "sample_rate": SAMPLE_RATE})
        t_dia = time.time() - td

        ann = (dia_out.speaker_diarization if hasattr(dia_out, "speaker_diarization")
               else dia_out.diarization if hasattr(dia_out, "diarization")
               else dia_out)
        for turn, _, raw_spk in ann.itertracks(yield_label=True):
            if raw_spk not in speaker_map:
                speaker_map[raw_spk] = f"SPEAKER_{next(label_iter, raw_spk)}"
            s, e = turn.start, turn.end
            if skip_short and (e - s) < MIN_SEG_DURATION:
                continue
            si, ei = int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)
            raw_segs.append((s, e, speaker_map[raw_spk], audio_np[si:ei]))
    else:
        regions = vad_segments(audio_np, SAMPLE_RATE) if use_vad else [(0.0, dur)]
        for s, e in regions:
            si, ei = int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)
            raw_segs.append((s, e, None, audio_np[si:ei]))

    if not raw_segs:
        return [], {"dur": dur, "rtf": 0, "dia": t_dia, "asr": 0, "speakers": 0, "n": 0}

    if progress_cb:
        progress_cb(0.3, f"Transcribing {len(raw_segs)} segments...")

    t_asr = time.time()
    work = [
        (i, s, e, spk, seg, asr_proc, asr_model, corr_tok, corr_model, hotwords, do_norm)
        for i, (s, e, spk, seg) in enumerate(raw_segs)
    ]
    buf = [None] * len(work)
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(_process_segment, a): a[0] for a in work}
        for fut in as_completed(futs):
            idx, s, e, spk, text, raw_words = fut.result()
            buf[idx] = Segment(s, e, spk or "", text, raw_words)
            done += 1
            if progress_cb:
                progress_cb(0.3 + 0.6 * done / len(work), f"{done}/{len(work)} segments")
    t_asr = time.time() - t_asr

    segments = [seg for seg in buf if seg and seg.text]
    if do_group:
        segments = group_turns(segments)

    t_total = time.time() - t0
    stats = {
        "dur": dur,
        "rtf": t_total / dur if dur else 0,
        "dia": t_dia,
        "asr": t_asr,
        "speakers": len(speaker_map),
        "n": len(segments),
    }
    return segments, stats


def _metrics(stats: dict) -> str:
    return (
        f"Audio: {stats['dur']:.1f}s | RTF: {stats['rtf']:.2f}x | Workers: {N_WORKERS} | "
        f"Diarization: {stats['dia']:.1f}s | ASR+Correction: {stats['asr']:.1f}s | "
        f"Speakers: {stats['speakers']} | Segments: {stats['n']}"
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(dia, asr_proc, asr_model, corr_tok, corr_model):  # noqa: C901
    _mdl = (dia, asr_proc, asr_model, corr_tok, corr_model)
    MAX_SPEAKERS = 10  # max speaker rename fields

    with gr.Blocks(title="Icelandic Transcription v3") as demo:
        gr.Markdown("# Icelandic Speech Transcription")

        # ── shared settings ────────────────────────────────────────────────
        with gr.Row():
            use_dia    = gr.Checkbox(label="Speaker diarization",    value=True)
            use_vad    = gr.Checkbox(label="VAD pre-filter",         value=False)
            do_group   = gr.Checkbox(label="Group consecutive turns", value=True)
            do_norm    = gr.Checkbox(label="Number normalisation",    value=False)
            skip_short = gr.Checkbox(label=f"Skip short (<{MIN_SEG_DURATION}s)", value=False)
            fmt_dd     = gr.Dropdown(["TXT","SRT","VTT"],  value="TXT", label="Format")
            header_cb  = gr.Checkbox(label="Speaker-header format (TXT)", value=True)

        with gr.Accordion("Diarization settings", open=False):
            min_dur = gr.Slider(0.0, 2.0, step=0.1, value=0.0, label="Min silence to split (s)")
            clust   = gr.Slider(0.3, 0.9, step=0.05, value=0.6, label="Clustering threshold")

        with gr.Accordion("Vocabulary / Hotwords", open=False):
            gr.Markdown("One entry per line — names, places, jargon. Fuzzy-matched against ASR output.")
            hotword_tb = gr.Textbox(lines=5, placeholder="Sigríður\nAlþingi\nReykjavík", label="Hotwords")

        # ── shared state ───────────────────────────────────────────────────
        seg_state   = gr.State(value=None)   # List[Segment]
        audio_state = gr.State(value=None)   # np.ndarray @ 16kHz

        COMMON_INPUTS = [use_dia, use_vad, do_group, do_norm, skip_short,
                         fmt_dd, header_cb, min_dur, clust, hotword_tb]

        with gr.Tabs():

            # ── Transcribe ─────────────────────────────────────────────────
            with gr.Tab("Transcribe"):
                audio_in  = gr.Audio(type="filepath",
                                     label="Audio (.wav .mp3 .flac .ogg .m4a)")
                run_btn   = gr.Button("Transcribe", variant="primary", size="lg")
                metrics_o = gr.Textbox(lines=1, label="Metrics", interactive=False)

                with gr.Accordion("Rename speakers", open=False):
                    gr.Markdown("Fill in the custom name for each detected speaker, then click **Apply**.")
                    spk_fields = [
                        gr.Textbox(label=f"Speaker {i+1}", visible=False)
                        for i in range(MAX_SPEAKERS)
                    ]
                    apply_btn = gr.Button("Apply names", size="sm")

                transcript_o = gr.Textbox(lines=22, label="Transcript",
                    placeholder="SPEAKER_A  [00:00.00 - 00:05.20]\nHello world ...")
                dl_o = gr.File(label="Download")

                # also feed the editor table
                with gr.Tab("_hidden_editor_feed", visible=False):
                    pass

                def _run(audio_path, u_dia, u_vad, grp, norm, skip,
                         fmt, hdr, mn, cl, hw_raw, progress=gr.Progress()):
                    if not audio_path:
                        empties = [gr.update(visible=False)] * MAX_SPEAKERS
                        return None, None, None, "", "", None, *empties

                    hotwords = [w.strip() for w in hw_raw.split("\n") if w.strip()]
                    try:
                        segs, stats = transcribe_file(
                            audio_path, *_mdl,
                            use_diarization=u_dia, use_vad=u_vad,
                            skip_short=skip, min_dur_off=mn, cluster_thresh=cl,
                            do_group=grp, hotwords=hotwords, do_norm=norm,
                            progress_cb=lambda v, d: progress(v, desc=d),
                        )
                        # load audio into state
                        aud, sr = sf.read(audio_path, dtype="float32")
                        if aud.ndim > 1:
                            aud = aud.mean(axis=1)
                        if sr != SAMPLE_RATE:
                            import librosa
                            aud = librosa.resample(aud, orig_sr=sr, target_sr=SAMPLE_RATE)

                        df     = segments_to_df(segs)
                        txt    = build_txt(segs, header_format=hdr)
                        path   = _write_file(segs, Path(audio_path).stem, fmt, hdr)
                        met    = _metrics(stats)

                        # build speaker rename fields
                        spk_labels = sorted({s.speaker for s in segs if s.speaker})
                        spk_updates = []
                        for i in range(MAX_SPEAKERS):
                            if i < len(spk_labels):
                                spk_updates.append(gr.update(
                                    visible=True, value=spk_labels[i], label=spk_labels[i]
                                ))
                            else:
                                spk_updates.append(gr.update(visible=False, value=""))

                        progress(1.0, desc="Done.")
                        return segs, aud, df, met, txt, path, *spk_updates
                    except Exception:
                        import traceback
                        empties = [gr.update(visible=False)] * MAX_SPEAKERS
                        return None, None, None, "", traceback.format_exc(), None, *empties

                # placeholder for editor table — defined below, referenced here
                # We use a workaround: store df in state, editor reads it
                df_state = gr.State(value=None)

                run_btn.click(
                    fn=_run,
                    inputs=[audio_in, *COMMON_INPUTS],
                    outputs=[seg_state, audio_state, df_state,
                             metrics_o, transcript_o, dl_o, *spk_fields],
                )

                def _apply_names(segs, *vals):
                    if not segs:
                        return segs, ""
                    labels  = sorted({s.speaker for s in segs if s.speaker})
                    rename  = {lbl: (vals[i].strip() or lbl)
                               for i, lbl in enumerate(labels) if i < len(vals)}
                    updated = [Segment(s.start, s.end,
                                       rename.get(s.speaker, s.speaker),
                                       s.text, s.raw_words)
                               for s in segs]
                    return updated, build_txt(updated, header_format=True)

                apply_btn.click(
                    fn=_apply_names,
                    inputs=[seg_state, *spk_fields],
                    outputs=[seg_state, transcript_o],
                )

            # ── Editor ─────────────────────────────────────────────────────
            with gr.Tab("Editor"):
                gr.Markdown(
                    "**Click a row** to play that segment. "
                    "Edit **Speaker** or **Text** cells inline. "
                    "**Conf** = avg CTC confidence — lower means less certain."
                )

                seg_table = gr.Dataframe(
                    headers=["Start","End","Speaker","Text","Conf"],
                    datatype=["str","str","str","str","str"],
                    interactive=True, wrap=True, label="Segments",
                )

                with gr.Row():
                    clip_player = gr.Audio(label="Segment playback", interactive=False)
                    with gr.Column():
                        conf_btn  = gr.Button("Show / hide confidence view", size="sm")
                        conf_html = gr.HTML(visible=False)

                with gr.Row():
                    ex_fmt  = gr.Dropdown(["TXT","SRT","VTT"], value="TXT", label="Export as")
                    ex_hdr  = gr.Checkbox(label="Speaker-header format", value=True)
                    ex_btn  = gr.Button("Export edited transcript", variant="primary")
                    ex_file = gr.File(label="Download")

                # populate table when df_state changes (after transcription)
                df_state.change(
                    fn=lambda df: df if df is not None else gr.update(),
                    inputs=[df_state],
                    outputs=[seg_table],
                )

                def _play(evt: gr.SelectData, segs, audio_np):
                    if segs is None or audio_np is None:
                        return None
                    idx = evt.index[0]
                    if idx >= len(segs):
                        return None
                    seg = segs[idx]
                    s = max(0, int(seg.start * SAMPLE_RATE))
                    e = min(len(audio_np), int(seg.end * SAMPLE_RATE))
                    return (SAMPLE_RATE, audio_np[s:e])

                seg_table.select(fn=_play, inputs=[seg_state, audio_state],
                                 outputs=[clip_player])

                _conf_vis = gr.State(value=False)

                def _toggle_conf(segs, vis):
                    new_vis = not vis
                    html = build_conf_html(segs) if (new_vis and segs) else ""
                    return gr.update(value=html, visible=new_vis), new_vis

                conf_btn.click(fn=_toggle_conf,
                               inputs=[seg_state, _conf_vis],
                               outputs=[conf_html, _conf_vis])

                def _export_edited(tbl, segs, fmt, hdr):
                    if segs is None:
                        return None
                    try:
                        if isinstance(tbl, pd.DataFrame):
                            df = tbl
                        elif isinstance(tbl, dict) and "data" in tbl:
                            df = pd.DataFrame(tbl["data"],
                                              columns=["Start","End","Speaker","Text","Conf"])
                        else:
                            return None
                        edited = df_to_segments(df, segs)
                        return _write_file(edited, "transcript", fmt, hdr)
                    except Exception:
                        return None

                ex_btn.click(fn=_export_edited,
                             inputs=[seg_table, seg_state, ex_fmt, ex_hdr],
                             outputs=[ex_file])

            # ── Batch ──────────────────────────────────────────────────────
            with gr.Tab("Batch"):
                batch_in  = gr.File(label="Audio files", file_count="multiple",
                                    file_types=[".wav",".mp3",".flac",".ogg",".m4a"])
                batch_btn = gr.Button("Transcribe all", variant="primary", size="lg")
                batch_log = gr.Textbox(lines=12, label="Log", interactive=False)
                batch_zip = gr.File(label="Download all (.zip)")

                def _batch(files, u_dia, u_vad, grp, norm, skip,
                           fmt, hdr, mn, cl, hw_raw, progress=gr.Progress()):
                    if not files:
                        return "No files.", None
                    hotwords  = [w.strip() for w in hw_raw.split("\n") if w.strip()]
                    log_lines, out_files = [], []
                    tmp_dir = Path(tempfile.mkdtemp())

                    for i, f in enumerate(files):
                        path = f if isinstance(f, str) else f.name
                        stem = Path(path).stem
                        progress(i / len(files), desc=f"{stem} ({i+1}/{len(files)})")
                        try:
                            segs, stats = transcribe_file(
                                path, *_mdl,
                                use_diarization=u_dia, use_vad=u_vad,
                                skip_short=skip, min_dur_off=mn, cluster_thresh=cl,
                                do_group=grp, hotwords=hotwords, do_norm=norm,
                            )
                            out = _write_file(segs, stem, fmt, hdr)
                            out_files.append((stem, out))
                            log_lines.append(
                                f"✓ {stem}: {stats['dur']:.0f}s, "
                                f"RTF {stats['rtf']:.2f}x, {stats['n']} segs"
                            )
                        except Exception as exc:
                            log_lines.append(f"✗ {stem}: {exc}")

                    ext = {"SRT": ".srt", "VTT": ".vtt"}.get(fmt, ".txt")
                    zip_path = str(tmp_dir / "transcripts.zip")
                    with zipfile.ZipFile(zip_path, "w") as zf:
                        for stem, p in out_files:
                            zf.write(p, arcname=f"{stem}{ext}")
                    progress(1.0, desc="Done.")
                    return "\n".join(log_lines), zip_path

                batch_btn.click(
                    fn=_batch,
                    inputs=[batch_in, *COMMON_INPUTS],
                    outputs=[batch_log, batch_zip],
                )

    return demo


def main():
    print(f"CPU: {_N_CPU}  |  Workers: {N_WORKERS}  |  Threads/worker: {THREADS_PER_WORKER}")
    torch.set_num_threads(THREADS_PER_WORKER)
    torch.set_num_interop_threads(N_WORKERS)
    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    _check_models()
    demo = build_ui(*load_all_models())
    demo.launch(server_name="0.0.0.0", server_port=7862, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()

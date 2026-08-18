"""
Benchmark tool: Distance-Threshold Speaker Count Sweep.

Sweeps distance_threshold from 0.40 to 0.60 in 0.02 increments over labeled test audio files
and reports predicted speaker count k vs ground truth speaker count per file.

Usage:
    python3 -m tools.bench_threshold_der --test-dir /home/stark/Downloads/audio_testing/
"""

import argparse
import json
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file, rms
from config import MIN_SPEECH_SECONDS, SAMPLE_RATE, SILENCE_RMS_THRESHOLD
from models.embedding import get_embedding
from models.offline_diarizer import pick_speaker_count_by_distance_threshold
from models.scd import split_on_speaker_change
from models.vad import SpeechSegmenter

MIN_SPEECH_SAMPLES = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)


def extract_embeddings_from_file(path: str) -> tuple[np.ndarray, list[tuple[float, float]]]:
    audio = load_audio_file(path)
    segmenter = SpeechSegmenter()
    vad_segments = list(segmenter.process(audio)) + list(segmenter.flush())

    embeddings, intervals = [], []
    for vad_seg in vad_segments:
        seg_audio = vad_seg["audio"]
        seg_start = vad_seg["start"]
        if len(seg_audio) == 0:
            continue

        for a, b in split_on_speaker_change(seg_audio):
            sub = seg_audio[a:b]
            if len(sub) < MIN_SPEECH_SAMPLES or rms(sub) < SILENCE_RMS_THRESHOLD:
                continue
            embeddings.append(get_embedding(sub))
            intervals.append((seg_start + a / SAMPLE_RATE, seg_start + b / SAMPLE_RATE))

    if not embeddings:
        return np.empty((0, 192), dtype=np.float32), []

    return np.asarray(embeddings, dtype=np.float32), intervals


def parse_true_speaker_count(filename: str, json_map: dict) -> int | None:
    # 1. Check json map first if present
    if filename in json_map:
        return int(json_map[filename])
    basename = os.path.basename(filename)
    if basename in json_map:
        return int(json_map[basename])

    # 2. Parse from filename pattern like spk-2, spk4, spk 3, spk1
    m = re.search(r"spk[-_\s]?(\d+)", basename, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def run_sweep_for_directory(test_dir: str):
    test_dir = test_dir.rstrip()
    if not os.path.isdir(test_dir):
        # Try stripping/handling trailing space if present
        alt_dir = test_dir + " "
        if os.path.isdir(alt_dir):
            test_dir = alt_dir
        else:
            print(f"Error: Directory not found: {test_dir}")
            return

    print("==========================================================================================")
    print("DISTANCE THRESHOLD SPEAKER COUNT SWEEP (0.40 -> 0.60, STEP 0.02)")
    print("==========================================================================================")
    print("⚠️  NOTE: This benchmark validates Bug 2 (Speaker Count Auto-Determination Across Single &")
    print("    Multi-Speaker Audio). It does NOT evaluate Bug 1 (Overlap Handling / Frame-Level DER)")
    print("    as timestamp-level ground-truth annotations are not present in this dataset.")
    print("==========================================================================================\n")

    # Load speaker_counts.json if available
    json_map = {}
    json_path = os.path.join(test_dir, "speaker_counts.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_map = json.load(f)
            print(f"Loaded ground-truth speaker counts from {os.path.basename(json_path)}")
        except Exception as exc:
            print(f"Warning: Could not parse {json_path}: {exc}")

    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    audio_files = sorted([f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in audio_exts])

    if not audio_files:
        print(f"No audio files found in {test_dir}")
        return

    thresholds = [round(t, 2) for t in np.arange(0.40, 0.61, 0.02)]
    file_results = []

    for a_file in audio_files:
        path = os.path.join(test_dir, a_file)
        true_k = parse_true_speaker_count(a_file, json_map)

        if true_k is None:
            print(f"Skipping {a_file} — ground truth speaker count not found in filename or JSON.")
            continue

        print(f"Extracting embeddings for '{a_file}' (True k = {true_k})...")
        embeddings, intervals = extract_embeddings_from_file(path)

        if len(embeddings) == 0:
            print(f"  WARNING: No valid speech embeddings extracted for '{a_file}' — skipping.")
            continue

        pred_ks = []
        for t in thresholds:
            k_pred, score, assignment = pick_speaker_count_by_distance_threshold(
                embeddings, distance_threshold=t
            )
            pred_ks.append(k_pred)

        file_results.append({
            "filename": a_file,
            "true_k": true_k,
            "segments": len(embeddings),
            "pred_ks": pred_ks,
        })

    if not file_results:
        print("No valid test files evaluated.")
        return

    # Render results table
    print("\n" + "=" * 115)
    print("PER-FILE SPEAKER COUNT PREDICTIONS ACROSS DISTANCE THRESHOLDS")
    print("=" * 115)
    header_th = " ".join(f"{t:>5.2f}" for t in thresholds)
    print(f"{'Filename':<30} | {'True k':>6} | {'Segs':>5} | {header_th}")
    print("-" * 115)

    th_errors = [[] for _ in thresholds]

    for res in file_results:
        fname = res["filename"]
        if len(fname) > 30:
            fname = fname[:27] + "..."
        true_k = res["true_k"]
        segs = res["segments"]
        preds_str = " ".join(f"{k:>5}" for k in res["pred_ks"])
        print(f"{fname:<30} | {true_k:>6} | {segs:>5} | {preds_str}")

        for idx, k_pred in enumerate(res["pred_ks"]):
            th_errors[idx].append(abs(k_pred - true_k))

    print("-" * 115)
    mae_str = " ".join(f"{np.mean(errs):>5.2f}" for errs in th_errors)
    exact_match_str = " ".join(f"{sum(e == 0 for e in errs):>5}" for errs in th_errors)
    print(f"{'Mean Abs Speaker Error (MAE)':<30} | {'--':>6} | {'--':>5} | {mae_str}")
    print(f"{'Exact Speaker Count Matches':<30} | {'--':>6} | {len(file_results):>5} | {exact_match_str}")
    print("=" * 115)

    # Pick best threshold by MAE
    maes = [np.mean(errs) for errs in th_errors]
    best_idx = int(np.argmin(maes))
    best_t = thresholds[best_idx]
    best_mae = maes[best_idx]
    best_exact = sum(e == 0 for e in th_errors[best_idx])

    print(f"\n🏆 BEST THRESHOLD: distance_threshold = {best_t:.2f}")
    print(f"   Mean Absolute Speaker Error: {best_mae:.2f}")
    print(f"   Exact Speaker Count Accuracy: {best_exact}/{len(file_results)} files ({best_exact / len(file_results) * 100:.1f}%)")
    print("==========================================================================================")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default="/home/stark/Downloads/audio_testing/", help="Directory containing test .mp3 audio files")
    args = parser.parse_args()

    run_sweep_for_directory(args.test_dir)


if __name__ == "__main__":
    main()

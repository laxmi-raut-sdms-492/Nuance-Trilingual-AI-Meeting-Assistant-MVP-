"""
Systematic Overlap Attribution Sweep Script across ALL 14 Test Audio Files.

Runs every file in ~/Downloads/audio_testing/ with OVERLAP_SEPARATION_ENABLED=True and logs:
1. Overlap segments detected
2. Separation attempts & confidence scores
3. Fallback count
4. Full attributed_spans output for high-confidence windows
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from config import SAMPLE_RATE
import config
from models.identifier import SpeakerIdentifier
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def run_overlap_sweep(test_dir: str):
    test_dir = test_dir.rstrip()
    if not os.path.isdir(test_dir):
        alt_dir = test_dir + " "
        if os.path.isdir(alt_dir):
            test_dir = alt_dir
        else:
            print(f"Error: Directory not found: {test_dir}")
            return

    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    audio_files = sorted([f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in audio_exts])

    if not audio_files:
        print(f"No audio files found in {test_dir}")
        return

    # Enable overlap separation for the sweep
    config.OVERLAP_SEPARATION_ENABLED = True

    try:
        sarvam = SarvamSTTAdapter()
    except Exception as exc:
        print(f"CRITICAL ERROR: Failed to initialize SarvamSTTAdapter: {exc}")
        return

    print("==========================================================================")
    print("SYSTEMATIC OVERLAP SEPARATION & ATTRIBUTION SWEEP ACROSS ALL 14 TEST FILES")
    print("==========================================================================\n")

    summary_table = []
    detailed_reports = []

    for idx, a_file in enumerate(audio_files, 1):
        a_path = os.path.join(test_dir, a_file)
        print(f"[{idx}/{len(audio_files)}] Processing file: '{a_file}'...")

        try:
            audio = load_audio_file(a_path)
        except Exception as exc:
            print(f"  ERROR loading {a_file}: {exc}")
            continue

        session = MeetingSession(
            session_id=f"SWEEP_OVERLAP_{idx}",
            identifier=SpeakerIdentifier(),
            stt_adapter=sarvam,
        )

        try:
            session.process_audio(audio)
        except Exception as exc:
            print(f"  ERROR processing {a_file}: {exc}")
            continue

        turns = session.transcript or []
        overlap_turns = [t for t in turns if t.get("is_overlap")]
        separated_turns = [t for t in turns if t.get("is_separated_overlap")]

        detected_overlap_count = len(overlap_turns)
        attempts = detected_overlap_count
        separated_count = len(separated_turns)

        confidences = [
            t.get("separation_confidence")
            for t in overlap_turns
            if t.get("separation_confidence") is not None
        ]

        mean_conf = f"{sum(confidences)/len(confidences):.2f}" if confidences else "N/A"
        fallback_triggered = "YES" if (detected_overlap_count > separated_count) else ("No" if detected_overlap_count > 0 else "N/A")

        summary_table.append({
            "filename": a_file,
            "overlap_detected": detected_overlap_count,
            "attempts": attempts,
            "separated_count": separated_count,
            "confidence": mean_conf,
            "fallback": fallback_triggered,
        })

        detailed_reports.append({
            "filename": a_file,
            "overlap_turns": overlap_turns,
        })

    # Summary Table Output
    print("\n" + "=" * 105)
    print("SYSTEMATIC OVERLAP SEPARATION SWEEP SUMMARY TABLE")
    print("=" * 105)
    print(f"{'Filename':<35} | {'Overlap Segs':>12} | {'Attempted':>9} | {'Separated':>9} | {'Avg Conf':>8} | {'Fallback?':>9}")
    print("-" * 105)
    for r in summary_table:
        fname = r["filename"]
        if len(fname) > 35:
            fname = fname[:32] + "..."
        print(
            f"{fname:<35} | {r['overlap_detected']:>12} | {r['attempts']:>9} | "
            f"{r['separated_count']:>9} | {r['confidence']:>8} | {r['fallback']:>9}"
        )
    print("=" * 105)

    # Detailed Overlap Segment Spans Report
    print("\n" + "=" * 105)
    print("DETAILED ATTRIBUTED_SPANS REPORT FOR ALL DETECTED OVERLAP SEGMENTS")
    print("=" * 105)

    for report in detailed_reports:
        fname = report["filename"]
        ov_turns = report["overlap_turns"]
        print(f"\n📄 File: {fname}")
        if not ov_turns:
            print("   ✓ No overlapping speech segments detected (VAD/SCD isolated sequential turns).")
            continue

        for idx, t in enumerate(ov_turns, 1):
            s = t.get("start_sec", 0.0)
            e = t.get("end_sec", 0.0)
            spk = t.get("speaker", "Unknown")
            lbl = t.get("speaker_label", "Unknown")
            is_sep = t.get("is_separated_overlap", False)
            conf = t.get("separation_confidence", "N/A")
            spans = t.get("attributed_spans", [])

            status_str = f"SEPARATED (conf={conf})" if is_sep else "FALLBACK (low conf or single stream)"
            print(f"\n   Segment #{idx} [{s:.1f}-{e:.1f}s] {spk} ({lbl}) -> Status: {status_str}")
            print(f"     Full Text: \"{t.get('text', '')}\"")
            if spans:
                print("     attributer_spans payload:")
                print(f"     {json.dumps(spans, ensure_ascii=False, indent=7)}")

    print("\n" + "=" * 105)
    print("OVERLAP SWEEP COMPLETE")
    print("=" * 105)


if __name__ == "__main__":
    test_dir = "/home/stark/Downloads/audio_testing /"
    if not os.path.exists(test_dir):
        test_dir = "/home/stark/Downloads/audio_testing/"
    run_overlap_sweep(test_dir)

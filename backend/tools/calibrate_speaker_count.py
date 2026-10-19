"""
Sweep decide_speakers over the per-line voice prints cached by
tools/diagnose_speaker_count.py and print the speaker count each setting
gives per meeting, next to the count the meeting shows today.

    python3 -m tools.calibrate_speaker_count
    python3 -m tools.calibrate_speaker_count --split 0.5 0.55 0.6 --same 0.35 0.4 0.45
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from db import repository as repo  # noqa: E402
from models.offline_diarizer import decide_speakers  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage", "embeddings")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, nargs="+", default=[0.50, 0.55, 0.60])
    ap.add_argument("--same", type=float, nargs="+", default=[0.40])
    args = ap.parse_args()

    titles = {m["id"]: m for m in repo.list_meetings()}
    for cache in sorted(glob.glob(os.path.join(CACHE_DIR, "*_lines.npz"))):
        mid = os.path.basename(cache).replace("_lines.npz", "")
        data = np.load(cache)
        m = titles.get(mid, {})
        shown = {s.get("name") for s in (m.get("speakerStats") or [])}
        line_dur = data["ends"] - data["starts"]
        print(f"{mid} '{(m.get('title') or '')[:24]:24s}' shows {len(shown):2d}  ", end="")
        for split in args.split:
            for same in args.same:
                a = decide_speakers(data["embeddings"], line_dur, split_distance=split, same_voice_distance=same)
                k_turn = len(set(a.tolist())) if a is not None else "-"
                if len(data["chunk_emb"]):
                    b = decide_speakers(data["chunk_emb"], data["chunk_dur"], split_distance=split, same_voice_distance=same)
                    secs = {}
                    for lab, d in zip(b, data["chunk_dur"]):
                        secs[int(lab)] = secs.get(int(lab), 0.0) + float(d)
                    k_chunk = len(secs)
                    top = " ".join(f"{s:.0f}" for s in sorted(secs.values(), reverse=True)[:6])
                else:
                    k_chunk, top = "-", ""
                print(f"| s{split:.2f}/m{same:.2f}: turns={k_turn} chunks={k_chunk} [{top}] ", end="")
        print()


if __name__ == "__main__":
    main()

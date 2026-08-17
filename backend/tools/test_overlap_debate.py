"""
Manual verification tool for Source Separation & Frontend attributed_spans output on real overlap files:
1. audio.mp3 (Yashraj & Siddesh referral overlap)
2. रहल गध... (Debate audio)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.identifier import SpeakerIdentifier
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter


def verify_overlap_spans():
    files = [
        ("/home/stark/Downloads/audio_testing /audio.mp3", "audio.mp3 (Yashraj & Siddesh)"),
        (
            "/home/stark/Downloads/audio_testing /रहल गध त छछरपन पर उतर#shortsfeed #rahulgandhispeech #ranchi #debate #shorts.mp3",
            "debate.mp3 (Rahul Gandhi Debate)",
        ),
    ]

    sarvam = SarvamSTTAdapter()
    import config

    print("==========================================================================")
    print("MANUAL VERIFICATION OF ATTRIBUTED SPANS ON REAL OVERLAP AUDIO FILES")
    print("==========================================================================\n")

    for a_path, label in files:
        if not os.path.exists(a_path):
            print(f"File not found: {a_path}")
            continue

        print(f"--- Testing File: {label} ---")
        audio = load_audio_file(a_path)

        # 1. Test Default (OVERLAP_SEPARATION_ENABLED = False)
        config.OVERLAP_SEPARATION_ENABLED = False
        session_off = MeetingSession(session_id=f"TEST_OFF_{label}", identifier=SpeakerIdentifier(), stt_adapter=sarvam)
        session_off.process_audio(audio)

        print("\n[Default: OVERLAP_SEPARATION_ENABLED = False]")
        print(f"Turns count: {len(session_off.transcript)}")
        for t in session_off.transcript:
            spk = t.get("speaker")
            lbl = t.get("speaker_label")
            is_ov = " [OVERLAP]" if t.get("is_overlap") else ""
            print(f"  [{t['start_sec']:.1f}-{t['end_sec']:.1f}s] {spk} ({lbl}){is_ov}: \"{t['text'][:70]}...\"")

        # 2. Test Opt-in (OVERLAP_SEPARATION_ENABLED = True)
        config.OVERLAP_SEPARATION_ENABLED = True
        session_on = MeetingSession(session_id=f"TEST_ON_{label}", identifier=SpeakerIdentifier(), stt_adapter=sarvam)
        session_on.process_audio(audio)

        print("\n[Opt-in: OVERLAP_SEPARATION_ENABLED = True]")
        print(f"Turns count: {len(session_on.transcript)}")
        for t in session_on.transcript:
            spk = t.get("speaker")
            lbl = t.get("speaker_label")
            is_ov = " [SEPARATED OVERLAP]" if t.get("is_separated_overlap") else (" [OVERLAP FALLBACK]" if t.get("is_overlap") else "")
            conf = f" (conf={t.get('separation_confidence')})" if t.get("separation_confidence") else ""
            spans = t.get("attributed_spans", [])
            print(f"  [{t['start_sec']:.1f}-{t['end_sec']:.1f}s] {spk} ({lbl}){is_ov}{conf}: \"{t['text'][:70]}...\"")
            if spans:
                print("     Attributed Spans Payload for Frontend:")
                print(f"     {json.dumps(spans, ensure_ascii=False, indent=6)}")
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    verify_overlap_spans()

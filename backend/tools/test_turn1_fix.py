"""
Test rapid turn fix on Turn 1 of MTG-26ac2b9ef3dc
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import load_audio_file
from models.identifier import SpeakerIdentifier
from pipeline import MeetingSession
from stt.sarvam_adapter import SarvamSTTAdapter
import models.scd as scd_module
import config as config_module


def test_rapid_fix():
    audio_path = "storage/audio/MTG-26ac2b9ef3dc/Recording Aug 19, 10:51 AM.webm"
    audio = load_audio_file(audio_path)

    # Set parameters for rapid turn detection
    scd_module.SCD_WINDOW_SECONDS = 0.4
    scd_module.WINDOW_SAMPLES = int(0.4 * 16000)
    scd_module.SCD_HOP_SECONDS = 0.15
    scd_module.HOP_SAMPLES = int(0.15 * 16000)
    scd_module.SCD_MIN_SUBSEGMENT_SECONDS = 0.5
    scd_module.MIN_SUBSEGMENT_SAMPLES = int(0.5 * 16000)
    scd_module.SCD_MIN_SEGMENT_SECONDS = 1.0

    sarvam = SarvamSTTAdapter()
    session = MeetingSession(session_id="RAPID_FIX", identifier=SpeakerIdentifier(), stt_adapter=sarvam)
    
    # Process audio
    session.process_audio(audio)

    print("==========================================================================")
    print("RESULTS AFTER RAPID TURN TUNING ON MTG-26ac2b9ef3dc")
    print("==========================================================================\n")

    turns = session.transcript
    print(f"Total turns generated: {len(turns)}\n")
    for idx, t in enumerate(turns, 1):
        st = t.get('start_sec', 0.0)
        en = t.get('end_sec', 0.0)
        spk = t.get('speaker')
        lbl = t.get('speaker_label')
        txt = t.get('text', '')
        print(f"Turn #{idx:02d} [{st:6.2f}s - {en:6.2f}s] (dur={en-st:4.2f}s) {spk:<15} ({lbl}):")
        print(f"  \"{txt}\"\n")


if __name__ == "__main__":
    test_rapid_fix()

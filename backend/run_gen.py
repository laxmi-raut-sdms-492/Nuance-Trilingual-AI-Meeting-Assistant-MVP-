import sys
import time
import json

sys.path.insert(0, '.')
import db.repository as repo
import api

meeting_id = sys.argv[1] if len(sys.argv) > 1 else "MTG-a7de196be666"
meeting = repo.get_meeting(meeting_id)

t0 = time.time()
fields = api._generate_summary(meeting_id, meeting["transcript"])
elapsed = time.time() - t0

repo.update_meeting(meeting_id, **fields)

print(f"OLLAMA_PROCESSING_TIME: {elapsed:.2f}s")
print(f"FALLBACK_TRIGGERED: {fields.get('summaryEngine') == 'extractive'}")
print(f"SUMMARY_ENGINE_USED: {fields.get('summaryEngine')}")
print("\n--- SUMMARY ---")
print(fields.get("summary"))
print("\n--- DECISIONS ---")
print(json.dumps(fields.get("decisions"), indent=2))
print("\n--- ACTION ITEMS ---")
print(json.dumps(fields.get("actionItems"), indent=2))
print("\n--- KEY INSIGHTS ---")
print(json.dumps(fields.get("insights"), indent=2))

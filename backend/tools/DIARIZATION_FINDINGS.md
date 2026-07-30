# Diarization over-split — root cause was the recording, not the clustering

Status: **evidence from three recordings. No code or config changed.**

| Meeting | Audio | How captured | Segments | Truth | Pipeline |
|---|---|---|---|---|---|
| `MTG-85407e28d5eb` "Test 1" | 381.6s | YouTube played on speakers, recorded on laptop mic | 123 | 5-6 | **23** |
| `MTG-de4df0ad891a` "Test 2" | 107.5s | same loopback path, short | 19 | 5 | **5** |
| `MTG-1c443bab406f` "Test 3" | 370.6s | **same video, MP3 uploaded directly** | 79 | 5-6 | **8** |

**Test 1 and Test 3 are the same meeting.** The only difference is that Test 1
went out through speakers and back in through the microphone, and Test 3 was
the file. That alone is 23 speakers versus 8, and it is the single largest
effect measured in this investigation. The initial diagnosis — "the merge
threshold is too tight" — was real but secondary; it was being blamed for
damage the capture path had already done.

Audio measurements behind that (`tools/audio_quality.py`):

```
                    Test 1 (loopback)   Test 3 (direct file)
crest factor           14.7 dB              20.7 dB
level drift             8.0 dB              11.6 dB
clipping               0.001%               0.000%
same-speaker dist       0.729                0.377
cross-speaker dist      0.840                0.953
separation             +0.111               +0.576     <- 5x
```

Clipping was never the problem in either. The mechanism is microphone AGC:
Test 1 has *less* dynamic range across a *longer* recording, which is gain
riding. AGC continuously renormalises level and timbre per speaker, and that is
exactly the information ECAPA encodes. At +0.111 separation, two segments of
one person sit almost as far apart as two different people, and no clustering
algorithm can recover that. At +0.576 the problem is easy.

**Rule that follows: loopback recordings are not valid input for diarization
evaluation.** Capture digitally — upload the file, or record the PipeWire
monitor source — before drawing any conclusion about clustering quality.

The length-dependence noted earlier still holds as a secondary effect (Test 2,
two minutes on the same bad path, came out correct because centroids had no
time to drift), but it is not the main story.

## The observation

`MTG-85407e28d5eb` ("Test 1", 381.6s, English, recorded in-browser).
Ground truth from whoever recorded it: **5-6 speakers**. Pipeline produced
**23**, of which 19 held under 5 seconds of talk time each.

```
Speaker_02  225.9s  60.5%     <- the four real, substantial voices
Speaker_05   52.0s  13.9%
Speaker_14   31.5s   8.4%
Speaker_13   30.7s   8.2%
--- 19 more clusters, 1.0-5.0s each, 0.3-1.3% ---
```

ASR itself was fine: 122 lines, 0 dropped, 0 failed segments, text reads
correctly. This is a clustering failure, nothing else.

## Root cause

From the run's own diarizer log: **99 matches, 23 new clusters, 0 merges.**

`CLUSTER_MERGE_DISTANCE = 0.25`, but segments match their cluster at distances
up to 0.61 (the dynamic threshold grows to `0.55 + 0.15`). Two clusters
belonging to the same person therefore sit 0.3-0.6 apart — outside 0.25. The
merge pass ran roughly 9 times and rejected every pair. The net that exists to
catch premature splits is set tighter than the splits it is meant to catch.

Secondary: `_maybe_merge_clusters()` is only called on the *match* branch of
`add_segment`, so a run that is busy minting new clusters checks for merges
least often — exactly backwards.

## Why raising the match threshold is not the fix

Measured with `tools/replay_diarizer.py` (ARI is agreement with offline
agglomerative clustering at k=6; cluster count alone is not the score):

```
thresh  merge  clusters  dust    ARI   top talk-time
 0.55    0.25     24      19   +0.845  226s 52s 32s 31s 5s 3s 2s   <- shipped
 0.55    0.65     17      13   +0.922  232s 58s 34s 32s 2s 2s 2s
 0.65    0.25     12       9   +0.794  247s 82s 30s 3s 2s 2s 2s
 0.75    0.25      6       4   +0.278  326s 42s 4s 2s 1s 1s
 0.85    0.25      2       0   +0.107  209s 166s
```

The 0.75 row lands on the right *count* and is far worse: one cluster absorbs
326s of a 374s meeting. Anyone tuning by cluster count alone will pick it.

A "short segments may not found a new cluster" rule was also tried
(`MinDurationDiarizer` in the replay tool). It cuts the dust — 2.5s minimum
gives 6 clusters, 1 dust — but ARI *falls* to +0.606, because those short
segments then attach to the wrong speaker instead of to a label of their own.
Fewer clusters, more wrong frames. Not an improvement.

## The regression check — why no streaming parameter is safe

Test 2 is already correct under the shipped config, so it is the control. Any
change has to leave it alone. Same sweep, both meetings:

```
                    Test 1 (374s, truth 5-6)   Test 2 (94s, truth 5)
merge 0.25 (shipped)  24 clusters  ARI +0.845    5 clusters  ARI +1.000
merge 0.45            22 clusters  ARI +0.886    5 clusters  ARI +1.000
merge 0.55            21 clusters  ARI +0.890    5 clusters  ARI +1.000
merge 0.65            17 clusters  ARI +0.922    4 clusters  ARI +0.752  <- fuses two real speakers
threshold 0.65        12 clusters  ARI +0.794    4 clusters  ARI +0.752  <- same damage
threshold 0.85         2 clusters  ARI +0.107    1 cluster   ARI +0.000
min_new 2.0-3.0      5-10 clusters ARI +0.606..  5 clusters  ARI +1.000  <- harmless here, harmful on Test 1
```

`CLUSTER_MERGE_DISTANCE = 0.65` is the best setting found on Test 1 and it
merges two genuine speakers on Test 2 (40s + 16s become one 56s cluster).
**0.55 is the largest value safe on both**, and it only moves Test 1 from 24
clusters to 21.

Test 3 — the clean capture — changes what tuning is worth:

```
config                              Test 1 (dirty)   Test 2 (clean-ish)   Test 3 (clean)
merge 0.25 (shipped)                24  ARI +0.845    5  ARI +1.000        8  ARI +0.988
merge 0.55                          21  ARI +0.890    5  ARI +1.000        6  ARI +0.999
merge 0.55 + on_new + min_new 2.0    9  ARI +0.744    5  ARI +1.000        5  ARI +1.000
threshold 0.75                       6  ARI +0.278    4  ARI +0.752        2  ARI +0.434
```

On clean audio the shipped config is already close (8 clusters against a truth
of 5-6, ARI +0.988). Two candidate changes, both evidenced across all three:

1. **`CLUSTER_MERGE_DISTANCE` 0.25 → 0.55.** Safe everywhere: improves Test 1
   and Test 3, leaves Test 2 untouched. One constant, no new behaviour.
2. **That, plus merge-checking on the new-cluster path and a 2.0s minimum to
   found a cluster.** Hits ground truth exactly on both clean recordings
   (5 and 5) and cuts Test 1 from 24 to 9. Costs a new knob and a behaviour
   change; the ARI drop on Test 1 is against a reference partition computed
   from degraded audio, so it is weak evidence either way.

Neither is applied. Decide with clean recordings only.

## What the embeddings actually support

Offline agglomerative clustering (average linkage, cosine) over the same cached
embeddings, with k chosen by silhouette and **no ground truth supplied**:

```
k=2 +0.154   k=3 +0.216   k=4 +0.209   k=5 +0.235
k=6 +0.265  <- picked     k=7 +0.261   k=8 +0.256   k=9 +0.251   k=10 +0.229
talk-time at k=6: 227s 60s 39s 32s 10s 7s
```

k=6 matches ground truth. So the embeddings, the VAD segmentation and SCD are
adequate; the **streaming assignment** is what fails.

Silhouette picked the right k on all three recordings with no ground truth
supplied: **k=6** on Test 1 (truth 5-6), **k=5** on Test 2 (truth 5), **k=5**
on Test 3 (truth 5-6, talk-time 230s 43s 28s 12s 4s). Three for three.

**Test 1's peak is shallow** (0.265 vs 0.261 at k=7). Test 2's is not:

```
k=2 +0.311   k=3 +0.329   k=4 +0.447
k=5 +0.464  <- picked, truth is 5     k=6 +0.447   k=7 +0.406   k=8 +0.335
talk-time at k=5: 40s 21s 16s 10s 7s
```

Two recordings, two correct picks, and on Test 2 the offline partition
reproduces the streaming labels **exactly** (ARI +1.000, identical talk-times).
That is the property that makes the second pass safe: on a meeting the streaming
diarizer already got right, it changes nothing.

Still only two recordings, both English, same mic and room, one of them short.

## Candidate fix, not implemented

A second pass over uploads: keep the streaming labels for the live WebSocket
path, and once a file finishes, re-cluster its cached segment embeddings
offline (agglomerative, silhouette-picked k), relabel the transcript, recompute
speaker stats and colours. That reproduces the k=6 result by construction.
Roughly 100 lines across `pipeline.py` and `api.py`; the embeddings already
exist in memory during processing and are currently discarded.

**Required guard: silhouette cannot evaluate k=1.** Neither recording tests a
solo speaker, and unguarded the pass would force a single-person dictation into
at least two speakers — turning a correct result into a wrong one. The pass
needs a floor: if the best silhouette score is weak, keep the streaming labels
instead of inventing a split. Pick that floor against a real one-speaker
recording; do not guess it.

## Reproducing

```bash
cd backend
python3 -m tools.extract_embeddings <meeting-id>      # ~40s, caches embeddings
python3 -m tools.replay_diarizer <meeting-id> --expect <n>
python3 -m tools.replay_diarizer <meeting-id> --auto-k-only
```

Extraction runs VAD + SCD + ECAPA once and caches the vectors to
`storage/embeddings/<meeting-id>.npz`; the sweep then costs seconds per
configuration instead of a 4-minute Whisper pass. The segmentation in
`extract_embeddings.py` mirrors `MeetingSession._consume` /
`_process_subsegment` — if those change, it must change with them.

## Note on the lost sample

`trilingual-meeting.wav`, the recording CLAUDE.md cites for the documented
baseline (20 lines, 4 speakers, en 39.8 / hi 40.2 / mr 20.0), was deleted along
with its meeting through the UI. Deletion is immediate and removes the audio;
there is no soft delete and the file was never in git. That baseline cannot be
re-measured until an equivalent trilingual recording exists. Worth keeping any
future reference recording outside `backend/storage/`, where the delete button
cannot reach it.

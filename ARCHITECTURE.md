# Architecture and performance

How the live pipeline is built, what constrains it, what has been measured, and
what is still unverified.

---

## 1. Where the time went — measured, not guessed

`scripts/profile_pipeline.py` times each stage of a detection frame separately
(1280×720, median ms):

| stage | ms | share |
|---|---|---|
| **YOLOv5s (CPU, imgsz=640)** | **75.8** | **77%** |
| MediaPipe Hands | 13.4 | 14% |
| MediaPipe Pose | 9.6 | 10% |
| MediaPipe FaceMesh | 1.7 | 2% |
| `cv2.cvtColor` | 0.1 | 0% |
| **total** | **100.6** | **~10 FPS** |

**The bottleneck is unambiguous: YOLOv5s, on CPU, at its default inference size,
at 77% of the frame budget.** Everything else combined is a quarter of it.

---

## 2. Two mistakes that were in there

**Mistake 1 — YOLO never used the GPU.** It ran on CPU. Measured:

| device | imgsz | ms | vs baseline |
|---|---|---|---|
| cpu | 640 | 75.8 | 1.0× ← what we were doing |
| cpu | 320 | 23.6 | 3.2× |
| mps | 640 | 17.5 | 4.3× |
| mps | 320 | **9.2** | **8.2×** |

**Mistake 2 — resizing the input frame did nothing.** This is a trap worth
recording: feeding YOLO a 320-px-wide frame instead of 1280 changed the time by
**0%**, because ultralytics letterboxes to `imgsz` internally regardless. Only
`imgsz` moves the needle. My first attempt at profiling resized the frame and
concluded "no headroom" — which was wrong, and the second measurement caught it.

---

## 3. The cadence principle

Every stage ran on every frame. But the signals they measure move at completely
different speeds:

| feature | responds to | changes in | cadence | cost |
|---|---|---|---|---|
| `eye_ratio` | head/gaze left–right | ~300 ms | **dense** | 1.7 ms |
| `head_tilt` | head roll | ~300 ms | **dense** | (same call) |
| `looking_up` | nose high in frame | follows head | **dense** | (same call) |
| `body_tilt` | leaning | slow, **and already 10-frame averaged** | sparse | 9.6 ms |
| `hand_on_table` | wrist below desk line | seconds | sparse | 13.4 ms |
| `phone_detected` | phone visible | seconds — a phone is *held* | **very sparse** | 75.8 ms |

**The insight: spend compute on the fast signals, not the expensive ones.**
FaceMesh keeps every frame because it drives the two fastest-changing features
and costs almost nothing. YOLO is 77% of the cost and measures the slowest
signal in the system, so it is decimated hardest. That is not a hack — it is
matching sampling rate to signal bandwidth.

Implemented in `scripts/detector.py`. Both cadence modes are supported and the
distinction matters:

- **`stride`** — every Nth call. **Deterministic.** Correct for the offline
  evaluation that produces the paper's numbers, because it reproduces exactly.
- **`interval_ms`** — at most once per N ms. **Adaptive.** Correct for a live
  deployment, where "every 3 frames" means different things at 15 fps and 30 fps.

### Measured effect

| configuration | ms/frame | FPS | phone checks/s | speedup |
|---|---|---|---|---|
| baseline (cpu, 640, every frame) | 93.5 | 10.7 | 10.7 | 1.0× |
| + GPU + imgsz 320 | 37.8 | 26.5 | 26.5 | 2.5× |
| **+ cadence on slow signals** | **14.9** | **67.1** | **13.4** | **6.3×** |
| + harder decimation | 11.2 | 89.3 | 11.2 | 8.4× |

The recommended configuration is **6.3× faster than the baseline while making
13.4 phone checks per second — more than the baseline's 10.7.** It is better on
both axes, not a trade.

Live loop, end to end: **122 ms/frame (8.2 FPS) → 28 ms/frame (35.7 FPS)**, and
capture stopped being throttled by the detector (11.1 → 23.4 fps).

---

## 4. ⚠️ The open risk: accuracy at imgsz=320 is NOT verified

**Speed is proven. Detection quality is not.** Two changes could plausibly cost
recall, and neither has been measured against ground truth:

1. **`imgsz=320`** halves the linear resolution YOLO sees. A phone held at arm's
   length occupies fewer pixels. The default 640 exists for a reason, and small
   objects are exactly what suffers.
2. **Phone decimation** means a phone is checked 13.4×/second instead of on every
   frame. A phone raised and pocketed inside ~75 ms would be missed.

**This must be measured before the number goes in the paper.** The test is
straightforward once clips exist: run the phone detector at imgsz 640 and 320
over the S7 clips and compare recall. Until then, **treat imgsz=320 as a
performance hypothesis, not a validated setting.** If recall drops, raise imgsz
and buy the time back from a harder pose stride — Pose is 9.6 ms and its signal
is already smoothed over 10 frames, so it is the cheapest thing to sacrifice.

Relaxing the confidence threshold (currently 0.1) is the other lever, but it
trades the opposite way: more false phone positives, which in a proctoring
system means false accusations. That trade needs explicit thought, not a nudge.

---

## 5. Remaining architectural work — decouple capture from inference

The live loop is still **serial**: capture → detect → encode → write, all in one
thread. That has three consequences:

1. **The camera is throttled to the detector.** Requested 15 fps, captured 23.4
   — the loop rate *is* the passage rate. On a slower machine the camera would
   drip.
2. **Frames queue in the driver** while detection runs, so decisions are made on
   stale frames. Frame age is invisible right now.
3. **Video encoding blocks inference.** `VideoWriter.write` is I/O and it sits in
   the same thread as YOLO.

The standard fix is a producer/consumer split:

```
  capture thread ──► bounded queue (maxlen, drop OLDEST) ──► inference thread
       │                                                          │
   never blocks                                       freshest frame wins
       │                                                          │
       └──────────────────────────────────────────────► writer thread
                                                       (video + telemetry)
```

**The bound is the point.** An unbounded queue is precisely the failure mode that
caused the RAM incident: a fast producer feeding a slow consumer with nothing to
stop it. The queue must have a hard `maxlen` and drop the *oldest* frame — for a
real-time system the newest frame is the valuable one, and a dropped frame is a
correct outcome, not an error. Drops get counted and reported.

That work is not done yet. Until it is, the loop is honest about being serial,
and the guardrails below are what keep it safe.

---

## 6. Guardrails currently enforced

Every one of these is a hard stop, not a warning.

| Guard | Default | Where |
|---|---|---|
| No streaming endpoints — polling only | — | dashboard serves one cached JPEG per request; there is no long-lived connection to flood |
| Frame payload cap | 300 KB | a frame above this is dropped, not served |
| Web view downscaled + quality-capped | 960 px, q70 | keeps the payload ~46 KB measured |
| Exactly one cached frame in memory | — | replaced, never appended |
| Bounded history | 400 entries | score trace trimmed |
| Wall-clock cap | 30 min | independent of `--duration` |
| **Peak-RSS guard** | **1500 MB** | checked every 2 s; aborts the session |
| **Free-disk guard** | **2 GB** | refuses to start; stops the session if it falls |
| Interrupt-safe shutdown | — | repeat Ctrl-C ignored during cleanup, so `writer.release()` always runs and the MP4 is never left corrupt |
| Disk-space pre-flight | 2 GB | refuses to start at all |

**Measured after these were added:** RSS 926 → 967 MB across a full session with
the dashboard being polled — flat, that being model warmup. Previously the same
test pulled **6.4 GB in six seconds**.

---

## 7. Production checklist

| | item | state |
|---|---|---|
| 1 | Profile before optimising | ✅ done, bottleneck identified |
| 2 | Single source of truth for thresholds | ✅ `rule_score.py` with a drift guard against `exam.py` |
| 3 | Bounded every producer | ✅ polling, bounded queue-of-one, bounded history |
| 4 | Hard resource stops | ✅ RSS, disk, wall-clock |
| 5 | Clean shutdown | ✅ signal-safe, repeat-interrupt-safe |
| 6 | Interruptible long operations | ✅ stop flag, never an unkillable loop |
| 7 | Config-driven, no magic numbers | ✅ every tunable is a flag, recorded in `summary.json` |
| 8 | Observability | ✅ ms/stage, FPS, RSS, disk, phone-check rate in the live readout and the bundle |
| 9 | Reusable, not copy-pasted | ✅ `detector.py` shared by live and offline paths |
| 10 | **Validate the optimisations against ground truth** | ❌ **open — see §4** |
| 11 | Decouple capture from inference | ❌ open — see §5 |
| 12 | Unit tests for the guards | ❌ open — bounded queue, payload cap, shutdown |
| 13 | Adaptive degradation under load | ❌ open — raise stride automatically if the loop falls behind |

Items 10–13 are the honest gap between "works, fast, and safe" and
"production-grade". Item 10 is the one that affects the paper.

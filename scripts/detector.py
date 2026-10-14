#!/usr/bin/env python
"""
The feature detector: tuned, cadence-aware, and reusable.

WHY THIS MODULE EXISTS
======================
Profiling the original per-frame path (median ms):

    stage                     ms      share
    ----------------------------------------
    YOLOv5s (CPU, imgsz=640)  75.8    77%   <-- the bottleneck
    MediaPipe Hands           13.4    14%
    MediaPipe Pose             9.6    10%
    MediaPipe FaceMesh         1.7     2%
    cv2.cvtColor               0.1     0%
    ----------------------------------------
    total                    100.6    ~10 FPS

Two separate mistakes were in there:

1.  YOLO ran on **CPU at the default imgsz=640** - it never touched the GPU.
    Resizing the *input frame* changed nothing, because ultralytics letterboxes
    to `imgsz` internally anyway. Only `imgsz` and `device` matter:

        device  imgsz        ms     vs before
        cpu     640          75.8   1.0x   <- what we were doing
        cpu     320          23.6   3.2x
        mps     640          17.5   4.3x
        mps     320           9.2   8.2x

2.  **Every stage ran on every frame**, even though the signals they measure move
    at wildly different speeds.

THE CADENCE IDEA
================
This is the optimisation that matters, and it comes from the physics of the
signals rather than from the code:

    feature          what it responds to     how fast it changes      cadence
    --------------------------------------------------------------------------
    eye_ratio        head/gaze left-right    ~300 ms for a head turn  DENSE
    head_tilt        head roll               ~300 ms                  DENSE
    body_tilt        leaning                 slow, AND already         SPARSE
                                             averaged over 10 frames
    looking_up       nose high in frame      follows the head         DENSE
    hand_on_table    wrist below desk line   seconds                  SPARSE
    phone_detected   phone visible to camera seconds - a phone is     VERY SPARSE
                                             held, not flickered

FaceMesh stays dense because it produces the two fastest-changing features and
costs only 1.7 ms. YOLO is 77% of the cost and measures the slowest-moving
signal in the system, so it is the one worth decimating hardest. That is the
whole insight: **spend the compute on the fast signals, not the expensive ones.**

Both cadence modes are supported, and the difference matters:

  * `stride` - every Nth detect() call. **Deterministic**, so the offline
    evaluation that produces the paper's numbers is reproducible run to run.
    This is the default and the right choice for anything published.
  * `interval_ms` - at most once per N milliseconds. Adaptive: it self-adjusts
    when the machine is slow. Right for a live deployment across unknown client
    hardware, where "every 3 frames" means different things at 15 and 30 fps.

`stride` wins when both are given.

HONEST LIMIT
------------
Decimating YOLO trades detection latency for speed, and the trade is real: at
8 detections/s with a 3-frame phone stride, a phone is checked ~2.7 times per
second. A phone raised and pocketed inside ~370 ms could be missed. That is a
tuning decision with a number attached, not a free win - so it is a parameter,
it is logged in the session summary, and the limit is stated here.

Usage
-----
    from detector import Detector
    det = Detector(imgsz=320, stride={"phone": 5, "pose": 4, "hands": 2})
    feats = det.detect(frame)
    print(det.report())

    uv run python scripts/detector.py            # benchmark the configurations
"""
from __future__ import annotations

import os
import time
from collections import deque

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from rule_score import (HEAD_TILT_THRESHOLD, LOOKING_UP_THRESHOLD,  # noqa: E402
                        PHONE_CLASS_ID, PHONE_CONF, SHOULDER_LANDMARK_ERROR,
                        SHOULDER_LEVEL_MAX_DIFF, SHOULDER_TILT_THRESHOLD,
                        SHOULDER_WINDOW, TABLE_REGION_THRESHOLD)

FEATURE_KEYS = ("eye_ratio", "head_tilt", "body_tilt", "looking_up",
                "hand_on_table", "phone_detected")

# every stage except FaceMesh runs at a fraction of the detection rate
DEFAULT_STRIDE = {"face": 1, "hands": 2, "pose": 4, "phone": 5}


def pick_device(prefer: str = "auto") -> str:
    """Apple GPU when it exists, else CPU."""
    if prefer != "auto":
        return prefer
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Detector:
    """
    Multimodal feature extractor with per-stage cadence.

    device        "auto" | "cpu" | "mps" | "cuda"
    imgsz         YOLO inference size. 320 is the profiling sweet spot.
    half          fp16 on the GPU.
    stride        {stage: run every Nth detect() call}. Deterministic.
    interval_ms   {stage: minimum ms between runs}. Adaptive. Stride wins.
    """

    def __init__(self, device: str = "auto", imgsz: int = 320, half: bool = False,
                 stride: dict | None = None, interval_ms: dict | None = None):
        from ultralytics import YOLO
        import mediapipe as mp

        self.device = pick_device(device)
        self.imgsz = imgsz
        self.half = bool(half) and self.device != "cpu"
        self.stride = dict(DEFAULT_STRIDE)
        if stride:
            self.stride.update(stride)
        self.interval_ms = interval_ms or {}
        self.use_time = bool(self.interval_ms)

        self._fm = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._hd = mp.solutions.hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._ps = mp.solutions.pose.Pose(
            min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self._yolo = YOLO("yolov5s.pt")

        self._shoulder = deque(maxlen=SHOULDER_WINDOW)
        self.calls = 0
        self._last_run = {k: 0.0 for k in self.stride}
        self._n = {"face": 0, "hands": 0, "pose": 0, "phone": 0}
        self._ms = {"face": 0.0, "hands": 0.0, "pose": 0.0, "phone": 0.0}

        # instance-level carry-forward cache, so two Detectors never share state
        self._cache = {"face": False, "eye_ratio": None, "head_tilt": None,
                       "hand_on_table": 0, "body_tilt": 0, "phone_detected": 0}

    # ------------------------------------------------------------------
    def _due(self, stage: str) -> bool:
        if self.use_time and stage in self.interval_ms:
            return (time.perf_counter() - self._last_run[stage]) * 1000 >= \
                self.interval_ms[stage]
        return self.calls % max(1, self.stride.get(stage, 1)) == 0

    def _timed(self, stage, fn):
        t0 = time.perf_counter()
        out = fn()
        self._ms[stage] += (time.perf_counter() - t0) * 1000
        self._n[stage] += 1
        self._last_run[stage] = time.perf_counter()
        return out

    # ------------------------------------------------------------------
    def detect(self, frame) -> dict:
        """One frame -> the seven features. Definitions unchanged; cadence staged."""
        import cv2
        h, w = frame.shape[:2]
        self.calls += 1
        c = self._cache

        f = dict(face=False, eye_ratio=None, head_tilt=None, body_tilt=0,
                 looking_up=0, hand_on_table=0, phone_detected=0,
                 suspicious_count=0, stages_run=0)

        need_rgb = self._due("face") or self._due("hands") or self._due("pose")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if need_rgb else None

        # ---- FaceMesh: eye ratio + head roll. Dense - fast-changing, cheap.
        if self._due("face"):
            res = self._timed("face", lambda: self._fm.process(rgb))
            f["stages_run"] += 1
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark

                def px(i):
                    return int(lm[i].x * w), int(lm[i].y * h)

                lx, ly = px(33)
                rx, ry = px(263)
                _, ny = px(1)
                c["face"] = True
                c["eye_ratio"] = ((lx + rx) // 2) / w
                c["head_tilt"] = abs(ly - ry)
                c["looking_up"] = int(ny < h * LOOKING_UP_THRESHOLD)
            else:
                c["face"] = False
                c["eye_ratio"] = None
                c["head_tilt"] = None
                c["looking_up"] = 0
        f["face"] = c["face"]
        f["eye_ratio"] = c["eye_ratio"]
        f["head_tilt"] = c["head_tilt"]
        f["looking_up"] = c["looking_up"]

        # ---- Hands: wrist below the desk line. Sparse - sustained posture.
        if self._due("hands"):
            res = self._timed("hands", lambda: self._hd.process(rgb))
            f["stages_run"] += 1
            hit = 0
            if res.multi_hand_landmarks:
                for hand in res.multi_hand_landmarks:
                    if hand.landmark[0].y > TABLE_REGION_THRESHOLD:
                        hit = 1
            c["hand_on_table"] = hit
        f["hand_on_table"] = c["hand_on_table"]

        # ---- Pose: body lean. Very sparse - already 10-frame averaged.
        if self._due("pose"):
            res = self._timed("pose", lambda: self._ps.process(rgb))
            f["stages_run"] += 1
            tilt = 0
            if res.pose_landmarks:
                ls = res.pose_landmarks.landmark[11]
                rs = res.pose_landmarks.landmark[12]
                self._shoulder.append((ls.x, rs.x))
                al = sum(s[0] for s in self._shoulder) / len(self._shoulder)
                ar = sum(s[1] for s in self._shoulder) / len(self._shoulder)
                d = abs(al - ar)
                if d <= SHOULDER_LANDMARK_ERROR and d < SHOULDER_LEVEL_MAX_DIFF and (
                        al > ar + SHOULDER_TILT_THRESHOLD
                        or ar > al + SHOULDER_TILT_THRESHOLD):
                    tilt = 1
            c["body_tilt"] = tilt
        f["body_tilt"] = c["body_tilt"]

        # ---- YOLO: phone. Most expensive, slowest signal - decimate hardest.
        if self._due("phone"):
            kw = dict(imgsz=self.imgsz, device=self.device, verbose=False)
            if self.half:
                kw["half"] = True      # only pass it when actually wanted; the
                                       # kwarg is deprecated and warns otherwise
            res = self._timed("phone", lambda: self._yolo(frame, **kw))
            f["stages_run"] += 1
            hit = 0
            for r in res:
                for b in r.boxes:
                    if int(b.cls[0]) == PHONE_CLASS_ID and float(b.conf[0]) > PHONE_CONF:
                        hit = 1
            c["phone_detected"] = hit
        f["phone_detected"] = c["phone_detected"]

        f["suspicious_count"] = (
            int(f["eye_ratio"] is not None and abs(f["eye_ratio"] - 0.5) > 0.25)
            + int(f["head_tilt"] is not None and f["head_tilt"] > HEAD_TILT_THRESHOLD)
            + int(bool(f["body_tilt"]))
            + int(bool(f["looking_up"]))
        )
        return f

    # ------------------------------------------------------------------
    def warmup(self, w: int = 1280, h: int = 720):
        """
        Run one dummy detection and reset the counters.

        The first YOLO inference on MPS compiles the graph and takes ~900 ms,
        roughly 60x a steady-state frame. Without this, that single call skews
        every mean in the session summary and makes the system look far slower
        than it is.
        """
        try:
            import numpy as np
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            self.calls = 0
            self.detect(dummy)
        except Exception:
            pass
        finally:
            self.calls = 0
            for k in self._ms:
                self._ms[k] = 0.0
            for k in self._n:
                self._n[k] = 0
            self._shoulder.clear()
            self._last_run = {k: 0.0 for k in self.stride}

    def mean_ms(self) -> dict:
        return {k: (self._ms[k] / self._n[k] if self._n[k] else 0.0) for k in self._ms}

    def amortised_ms(self) -> float:
        """Mean cost per detect() call, accounting for cadence."""
        if not self.calls:
            return 0.0
        return sum(self._ms.values()) / self.calls

    def phone_rate(self, detect_fps: float) -> float:
        """How many phone checks per second this cadence yields at a given rate."""
        s = max(1, self.stride.get("phone", 1))
        return detect_fps / s

    def report(self) -> str:
        m = self.mean_ms()
        total = sum(m.values())
        out = [f"  device={self.device}  imgsz={self.imgsz}  half={self.half}",
               f"  stride={self.stride}"
               + (f"  interval_ms={self.interval_ms}" if self.use_time else "")]
        for k in ("phone", "hands", "pose", "face"):
            share = (100 * m[k] / total) if total else 0
            out.append(f"    {k:6} {m[k]:7.2f} ms/call  {self._n[k]:6} calls  "
                       f"{share:5.1f}% of stage time")
        if total and self.calls:
            am = self.amortised_ms()
            out.append(f"    amortised {am:6.2f} ms per frame over {self.calls} frames"
                       f"  ->  {1000/max(am,1e-9):.1f} FPS")
        return "\n".join(out)

    def close(self):
        for obj in (self._fm, self._hd, self._ps):
            try:
                obj.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# benchmark
# ----------------------------------------------------------------------
def _bench():
    import statistics

    import numpy as np

    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    configs = [
        ("baseline (cpu, 640, every frame)", dict(device="cpu", imgsz=640,
                                                  stride=dict(face=1, hands=1,
                                                              pose=1, phone=1))),
        ("+ GPU + imgsz 320", dict(device="auto", imgsz=320,
                                   stride=dict(face=1, hands=1, pose=1, phone=1))),
        ("+ cadence on slow signals", dict(device="auto", imgsz=320)),
        ("+ harder decimation", dict(device="auto", imgsz=320,
                                     stride=dict(face=1, hands=3, pose=5, phone=8))),
    ]

    print("=" * 78)
    print("DETECTOR BENCHMARK  (1280x720, median over 40 frames each)")
    print("=" * 78)
    print(f"  {'configuration':36} {'ms/frame':>10} {'FPS':>8} {'phone checks/s':>16}")
    print("  " + "-" * 74)

    results = []
    for label, kw in configs:
        d = Detector(**kw)
        d.detect(frame)  # warm
        ts = []
        for _ in range(40):
            t0 = time.perf_counter()
            d.detect(frame)
            ts.append((time.perf_counter() - t0) * 1000)
        med = statistics.median(ts)
        fps = 1000 / med
        pr = d.phone_rate(fps)
        print(f"  {label:36} {med:10.1f} {fps:8.1f} {pr:16.1f}")
        results.append((label, med, fps, pr, d.report()))
        d.close()

    base = results[0][1]
    print()
    print("=" * 78)
    print("SPEEDUP vs BASELINE")
    print("=" * 78)
    for label, med, fps, pr, _ in results:
        print(f"  {label:36} {base/med:5.1f}x   ({med:6.1f} ms, {fps:5.1f} FPS, "
              f"{pr:4.1f} phone checks/s)")

    print()
    print("=" * 78)
    print("DETAIL OF THE RECOMMENDED CONFIGURATION")
    print("=" * 78)
    print(results[2][4])
    print()
    print("  NOTE: 'phone checks/s' is the real trade being made. A phone raised")
    print("  and pocketed faster than 1/phone_checks_per_s can be missed. That is")
    print("  a tuning decision with a number attached, not a free win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_bench())

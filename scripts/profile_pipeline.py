#!/usr/bin/env python
"""
Profile the per-frame pipeline to find the real bottleneck.

Measures each stage of the detection path separately, and quantifies the
obvious optimisations, so tuning decisions are driven by numbers rather than
intuition.

Everything here is bounded: a fixed number of iterations, no streams, no
threads, no unbounded queues.

Usage
-----
    uv run python scripts/profile_pipeline.py
    uv run python scripts/profile_pipeline.py --iters 30
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent


from sysutil import peak_rss_mb  # noqa: E402


def timeit(fn, iters):
    """Return (mean_ms, p50_ms, p95_ms, max_ms) and swallow warmup."""
    fn()  # warmup, not counted
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    p95 = ts[min(len(ts) - 1, int(len(ts) * 0.95))]
    return statistics.mean(ts), statistics.median(ts), p95, max(ts)


def row(name, m, p50, p95, mx, total=None):
    share = f"  {100*m/total:5.1f}%" if total else "       "
    print(f"  {name:34} {m:8.1f} {p50:8.1f} {p95:8.1f} {mx:8.1f}{share}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    import cv2
    import numpy as np
    import mediapipe as mp
    import torch
    from ultralytics import YOLO

    print("=" * 78)
    print("PIPELINE PROFILE")
    print("=" * 78)
    print(f"  python {sys.version.split()[0]} | opencv {cv2.__version__} | "
          f"mediapipe {mp.__version__} | torch {torch.__version__}")
    print(f"  mps available : {torch.backends.mps.is_available()}")
    print(f"  input frame   : {args.width}x{args.height}   iterations: {args.iters}")
    print()

    frame = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)

    fm = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
    hd = mp.solutions.hands.Hands(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    ps = mp.solutions.pose.Pose(min_detection_confidence=0.7,
                                min_tracking_confidence=0.7)
    yolo = YOLO("yolov5s.pt")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(f"  {'stage':34} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8}   share")
    print("  " + "-" * 76)

    results = {}
    results["cvtColor"] = timeit(lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), args.iters)
    row("cv2.cvtColor", *results["cvtColor"])

    results["face_mesh"] = timeit(lambda: fm.process(rgb), args.iters)
    row("MediaPipe FaceMesh", *results["face_mesh"])

    results["hands"] = timeit(lambda: hd.process(rgb), args.iters)
    row("MediaPipe Hands", *results["hands"])

    results["pose"] = timeit(lambda: ps.process(rgb), args.iters)
    row("MediaPipe Pose", *results["pose"])

    results["yolo_1280"] = timeit(lambda: yolo(frame, verbose=False), args.iters)
    row(f"YOLOv5s @ {args.width}x{args.height}", *results["yolo_1280"])

    # ---- what would a smaller YOLO input buy us? -------------------------
    for size in (640, 416, 320):
        small = cv2.resize(frame, (size, int(size * args.height / args.width)))
        results[f"yolo_{size}"] = timeit(lambda s=small: yolo(s, verbose=False), args.iters)
        row(f"YOLOv5s @ {size}px wide", *results[f"yolo_{size}"])

    # ---- imgsz parameter (ultralytics resizes internally) ----------------
    results["yolo_imgsz320"] = timeit(
        lambda: yolo(frame, imgsz=320, verbose=False), args.iters)
    row("YOLOv5s @ imgsz=320 (letterboxed)", *results["yolo_imgsz320"])

    # ---- does MPS help? ---------------------------------------------------
    if torch.backends.mps.is_available():
        try:
            import copy
            y2 = YOLO("yolov5s.pt")
            y2.model = copy.deepcopy(y2.model).to("mps")
            results["yolo_mps"] = timeit(lambda: y2(frame, verbose=False), min(args.iters, 15))
            row(f"YOLOv5s on MPS @ {args.width}px", *results["yolo_mps"])
        except Exception as e:  # noqa: BLE001
            print(f"  MPS test failed: {type(e).__name__}: {str(e)[:60]}")
    else:
        print("  MPS not available on this torch build")

    # ---- pose complexity --------------------------------------------------
    ps0 = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.7,
                                 min_tracking_confidence=0.7)
    results["pose_c0"] = timeit(lambda: ps0.process(rgb), args.iters)
    row("MediaPipe Pose (complexity 0)", *results["pose_c0"])
    ps0.close()

    # ---- totals -----------------------------------------------------------
    full = sum(results[k][0] for k in ("cvtColor", "face_mesh", "hands", "pose", "yolo_1280"))
    print()
    print("  " + "-" * 76)
    print(f"  {'FULL STAGE SUM (median)':34} "
          f"{sum(results[k][1] for k in ('cvtColor','face_mesh','hands','pose','yolo_1280')):8.1f} ms"
          f"   -> {1000/sum(results[k][1] for k in ('cvtColor','face_mesh','hands','pose','yolo_1280')):.2f} FPS")

    for k in ("cvtColor", "face_mesh", "hands", "pose", "yolo_1280"):
        print(f"      {k:30} {100*results[k][1]/sum(results[x][1] for x in ('cvtColor','face_mesh','hands','pose','yolo_1280')):5.1f}% of the total")

    print()
    print("=" * 78)
    print("OPTIMISATION HEADROOM (median ms)")
    print("=" * 78)
    cur = results["yolo_1280"][1]
    for k, label in (("yolo_640", "YOLO downscaled to 640px"),
                     ("yolo_416", "YOLO downscaled to 416px"),
                     ("yolo_320", "YOLO downscaled to 320px"),
                     ("yolo_imgsz320", "YOLO imgsz=320")):
        v = results[k][1]
        print(f"  {label:34} {v:7.1f} ms   ({cur/v:4.1f}x faster, "
              f"saves {cur-v:6.1f} ms/frame)")
    if "yolo_mps" in results:
        v = results["yolo_mps"][1]
        print(f"  {'YOLO on MPS':34} {v:7.1f} ms   ({cur/v:4.1f}x faster, saves {cur-v:6.1f} ms/frame)")
    v = results["pose_c0"][1]
    pv = results["pose"][1]
    print(f"  {'Pose at complexity 0':34} {v:7.1f} ms   ({pv/v:4.1f}x faster, saves {pv-v:6.1f} ms/frame)")
    print()
    print(f"  peak RSS during profiling: {peak_rss_mb():.0f} MB")
    for d in (fm, hd, ps):
        try:
            d.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

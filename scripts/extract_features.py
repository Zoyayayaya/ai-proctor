#!/usr/bin/env python
"""
Offline feature extraction: scenario clips -> the labelled dataset.

Turns the recorded clips into `data/dataset_labelled.csv`, one row per processed
frame, with columns the evaluation harness consumes:

    session_id, clip_id, scenario, repeat, frame_idx, t_seconds,
    eye_ratio, head_tilt, body_tilt, looking_up, hand_on_table,
    phone_detected, duration, label

The LABEL comes from the filename, i.e. from the scenario the person was asked
to perform - never from the system's own score. That is the whole point: the
old dataset.csv had `label = 1 if score >= 60`, which made the reported accuracy
circular.

The feature maths is a line-by-line port of CheatingDetector.detect_cheating in
exam.py, so a number measured offline means the same thing as one measured live.

Requires the heavy detector stack:
    uv sync --extra detect

Usage
-----
    uv run python scripts/extract_features.py
    uv run python scripts/extract_features.py --clips data/clips --out data/dataset_labelled.csv
    uv run python scripts/extract_features.py --stride 2      # match live FRAME_SKIP
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_score as rs  # noqa: E402

# Scenario -> label. The scenario is prescribed behaviour, so this is ground truth.
# The scenario table. It must match live_test.SCENARIOS and RECORDING_SPEC.md.
#
# It did not. This table covered S1-S7, with names that were off by one from S5
# onward - S5 was labelled "look down" where the spec says "head roll", S6
# "phone" where the spec says "look down", S7 "out of frame" where the spec says
# "phone" - and S8 was absent entirely.
#
# With the S[1-7] filename regex below, that meant every S8 clip was SILENTLY
# SKIPPED: six clips of a 48-clip set, dropped without a word, and the run
# reported success. The operator caught it by noticing the extractor processing
# 42 clips and skipping the S8 list.
#
# The binary labels happened to be right, because the safe/cheating split is the
# same in both tables. So nothing downstream could tell, and no result would have
# looked wrong. Only the names were wrong, and only the missing clips were lost -
# which is the worst kind of bug, because the numbers still add up.
#
# test_scenario_tables_agree_with_the_recorder pins this now, so the three
# copies cannot drift apart again without a test going red.
SCENARIO_LABELS = {
    "S1": 0,  # focused on screen
    "S2": 0,  # fidgeting
    "S3": 0,  # reading / nodding
    "S4": 1,  # look away
    "S5": 1,  # head roll
    "S6": 1,  # look down at notes
    "S7": 1,  # phone visible
    "S8": 1,  # lean / leave frame
}

import telemetry  # noqa: E402
from rule_score import (PHONE_CLASS_ID, PHONE_CONF,  # noqa: E402
                        SHOULDER_LANDMARK_ERROR, SHOULDER_LEVEL_MAX_DIFF,
                        SHOULDER_WINDOW)

FNAME = re.compile(r"^(S[1-8])_p(\d+)_r(\d+)$")


def parse_name(stem: str):
    """`S6_p1_r2` -> ('S6', 1, 2). Returns None if the name does not conform."""
    m = FNAME.match(stem)
    if not m:
        return None
    scen, person, rep = m.group(1), int(m.group(2)), int(m.group(3))
    if scen not in SCENARIO_LABELS:
        return None
    return scen, person, rep


def build_detector(device: str, imgsz: int, half: bool, stride: dict):
    """
    The shared, tuned detector (scripts/detector.py).

    NOTE ON CADENCE. Offline extraction runs every stage on every frame by
    default - stride 1 across the board. Decimation is a *live* optimisation for
    real-time throughput; here there is no time pressure, so the faithful choice
    is to evaluate the paper's feature definitions on every frame. The GPU and
    imgsz settings are pure speed and change nothing about the features, so they
    are used freely.

    Pass --imgsz 320 --stride-phone 5 etc. if you deliberately want offline
    numbers that mirror the live deployment's settings instead.
    """
    from detector import Detector
    return Detector(device=device, imgsz=imgsz, half=half, stride=stride)


def process_clip(path: Path, det, stride: int = 1) -> list[dict]:
    """
    One clip -> rows. Frame stride here is a sampling choice (every Nth frame of
    the video), independent of the detector's per-stage cadence.
    """
    import cv2

    scen, person, rep = parse_name(path.stem)
    label = SCENARIO_LABELS[scen]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    rows: list[dict] = []
    cheating_start: float | None = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue

        t = idx / fps
        idx += 1
        f = det.detect(frame)
        if not f["face"]:
            continue

        # duration accrues only while a suspicious state holds (exam.py logic)
        active = f["suspicious_count"] >= 2 or f["phone_detected"] == 1
        if active:
            if cheating_start is None:
                cheating_start = t
            duration = t - cheating_start
        else:
            cheating_start = None
            duration = 0.0

        rows.append(dict(
            session_id=f"p{person}", clip_id=f"{scen}_p{person}_r{rep}",
            scenario=scen, repeat=rep, frame_idx=idx - 1, t_seconds=round(t, 3),
            eye_ratio=f["eye_ratio"], head_tilt=f["head_tilt"],
            body_tilt=f["body_tilt"], looking_up=f["looking_up"],
            hand_on_table=f["hand_on_table"], phone_detected=f["phone_detected"],
            duration=round(duration, 3), label=label,
        ))

    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--out", default="data/dataset_labelled.csv")
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth VIDEO frame (1 = all frames)")

    # detector tuning. Defaults are chosen for offline FIDELITY: every stage on
    # every frame, at the resolution the paper's numbers were produced at.
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO inference size. 640 = paper fidelity (default offline); "
                         "320 is the live deployment setting, ~8x faster")
    ap.add_argument("--half", action="store_true", help="fp16 inference on the GPU")
    ap.add_argument("--stride-face", type=int, default=1)
    ap.add_argument("--stride-hands", type=int, default=1)
    ap.add_argument("--stride-pose", type=int, default=1)
    ap.add_argument("--stride-phone", type=int, default=1,
                    help="1 = check for a phone on every frame (offline default)")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="do not report progress or failures")
    ap.add_argument("--run-label", default="")
    args = ap.parse_args()

    clips_dir = Path(args.clips)
    if not clips_dir.is_dir():
        msg = f"clips folder not found: {clips_dir}"
        print(f"error: {msg}", file=sys.stderr)
        print("  Record the clips first - see RECORDING_SPEC.md.", file=sys.stderr)
        telemetry.report_error("extract_features.no_clips",
                               FileNotFoundError(msg),
                               {"clips_dir": str(clips_dir)},
                               run_label=args.run_label,
                               enabled=not args.no_telemetry)
        return 1

    vids = [p for p in sorted(clips_dir.iterdir())
            if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}]
    good, bad = [], []
    for p in vids:
        (good if parse_name(p.stem) else bad).append(p)
    if bad:
        print("skipping files whose names do not match <scenario>_p<person>_r<repeat>:")
        for p in bad:
            print("   ", p.name)
    if not good:
        print("error: no usable clips found.", file=sys.stderr)
        telemetry.report_error(
            "extract_features.no_usable_clips", ValueError("no usable clips"),
            {"clips_dir": str(clips_dir), "video_files_seen": len(vids),
             "rejected_names": [p.name for p in bad][:20]},
            run_label=args.run_label, enabled=not args.no_telemetry)
        return 1

    print(f"processing {len(good)} clips "
          f"(video stride={args.stride}, detector imgsz={args.imgsz}) ...")
    telemetry.report("extract_started",
                     {"clips": len(good), "video_stride": args.stride,
                      "imgsz": args.imgsz},
                     run_label=args.run_label,
                     enabled=not args.no_telemetry)
    det = build_detector(args.device, args.imgsz, args.half,
                         {"face": args.stride_face, "hands": args.stride_hands,
                          "pose": args.stride_pose, "phone": args.stride_phone})
    print(f"  device {det.device}   stride {det.stride}")
    det.warmup(1280, 720)

    all_rows = []
    for i, p in enumerate(good, 1):
        try:
            rows = process_clip(p, det, stride=args.stride)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(good)}] {p.name:22} -> FAILED ({type(e).__name__})")
            telemetry.report_error("extract_features.process_clip", e,
                                   {"clip": p.name, "index": i,
                                    "of": len(good)},
                                   run_label=args.run_label,
                                   enabled=not args.no_telemetry)
            continue
        all_rows.extend(rows)
        print(f"  [{i}/{len(good)}] {p.name:22} -> {len(rows):4d} frames")
        if i % 5 == 0 or i == len(good):
            telemetry.report("extract_progress",
                             {"done": i, "of": len(good),
                              "frames_so_far": len(all_rows)},
                             run_label=args.run_label,
                             enabled=not args.no_telemetry)

    df = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nwrote {out}")
    print(f"  {len(df)} frames, {df['clip_id'].nunique()} clips, "
          f"{df['session_id'].nunique()} sessions")
    print(f"  class balance: {df['label'].value_counts().sort_index().to_dict()}")
    print(f"  phone-positive frames: {int(df['phone_detected'].sum())}")
    print()
    print(det.report())
    det.close()

    telemetry.report("extract_finished",
                     {"clips": int(df["clip_id"].nunique()),
                      "frames": len(df),
                      "sessions": int(df["session_id"].nunique()),
                      "class_balance": {str(k): int(v) for k, v in
                                        df["label"].value_counts().sort_index().items()},
                      "phone_positive_frames": int(df["phone_detected"].sum()),
                      "device": det.device, "imgsz": det.imgsz},
                     run_label=args.run_label,
                     enabled=not args.no_telemetry)
    print("\nnext: uv run python scripts/eval.py")
    return 0


if __name__ == "__main__":
    telemetry.install_error_reporter(
        "extract_features", enabled="--no-telemetry" not in sys.argv)
    raise SystemExit(main())

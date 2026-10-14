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
SCENARIO_LABELS = {
    "S1": 0,  # focused on screen
    "S2": 0,  # normal fidgeting
    "S3": 0,  # reading / nodding
    "S4": 1,  # sustained look away
    "S5": 1,  # sustained look down
    "S6": 1,  # phone visible
    "S7": 1,  # out of frame
}

from rule_score import (PHONE_CLASS_ID, PHONE_CONF,  # noqa: E402
                        SHOULDER_LANDMARK_ERROR, SHOULDER_LEVEL_MAX_DIFF,
                        SHOULDER_WINDOW)

FNAME = re.compile(r"^(S[1-7])_p(\d+)_r(\d+)$")


def parse_name(stem: str):
    """`S6_p1_r2` -> ('S6', 1, 2). Returns None if the name does not conform."""
    m = FNAME.match(stem)
    if not m:
        return None
    scen, person, rep = m.group(1), int(m.group(2)), int(m.group(3))
    if scen not in SCENARIO_LABELS:
        return None
    return scen, person, rep


def build_detectors():
    """FaceMesh / Hands / Pose / YOLO, with exam.py's confidence settings."""
    import mediapipe as mp
    from ultralytics import YOLO

    fm = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.7, min_tracking_confidence=0.7)
    yolo = YOLO("yolov5s.pt")
    return fm, hands, pose, yolo


def process_clip(path: Path, fm, hands, pose, yolo, stride: int = 1) -> list[dict]:
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
    shoulder_hist: list[tuple[float, float]] = []
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
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_res = fm.process(rgb)
        hand_res = hands.process(rgb)
        pose_res = pose.process(rgb)
        yolo_res = yolo(frame, verbose=False)

        idx += 1
        if not face_res.multi_face_landmarks:
            continue

        fl = face_res.multi_face_landmarks[0]

        def px(lm):
            return int(lm.x * w), int(lm.y * h)

        left_eye = px(fl.landmark[33])
        right_eye = px(fl.landmark[263])
        nose = px(fl.landmark[1])
        # exam.py uses integer floor division here, so the offline port does too
        eye_center_x = (left_eye[0] + right_eye[0]) // 2
        head_tilt = abs(left_eye[1] - right_eye[1])
        eye_ratio = eye_center_x / w

        hand_on_table = 0
        if hand_res.multi_hand_landmarks:
            for hand in hand_res.multi_hand_landmarks:
                if hand.landmark[0].y > rs.TABLE_REGION_THRESHOLD:
                    hand_on_table = 1

        body_tilt = 0
        looking_up = 0
        if pose_res.pose_landmarks:
            pl = pose_res.pose_landmarks
            ls = pl.landmark[11]  # LEFT_SHOULDER
            rsh = pl.landmark[12]  # RIGHT_SHOULDER
            shoulder_hist.append((ls.x, rsh.x))
            if len(shoulder_hist) > SHOULDER_WINDOW:
                shoulder_hist.pop(0)
            avg_l = float(np.mean([s[0] for s in shoulder_hist]))
            avg_r = float(np.mean([s[1] for s in shoulder_hist]))
            diff = abs(avg_l - avg_r)
            if diff <= SHOULDER_LANDMARK_ERROR and diff < SHOULDER_LEVEL_MAX_DIFF:
                if avg_l > avg_r + rs.SHOULDER_TILT_THRESHOLD:
                    body_tilt = 1
                elif avg_r > avg_l + rs.SHOULDER_TILT_THRESHOLD:
                    body_tilt = 1
            if nose[1] < h * rs.LOOKING_UP_THRESHOLD:
                looking_up = 1

        phone_detected = 0
        for r in yolo_res:
            for b in r.boxes:
                if (int(b.cls[0]) == PHONE_CLASS_ID
                        and float(b.conf[0]) > PHONE_CONF):
                    phone_detected = 1

        # duration accrues only while a suspicious state holds (exam.py logic)
        suspicious = rs.suspicious_indicators(eye_ratio, head_tilt, body_tilt, looking_up)
        active = suspicious >= 2 or phone_detected == 1
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
            eye_ratio=eye_ratio, head_tilt=head_tilt, body_tilt=body_tilt,
            looking_up=looking_up, hand_on_table=hand_on_table,
            phone_detected=phone_detected, duration=round(duration, 3),
            label=label,
        ))

    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--out", default="data/dataset_labelled.csv")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (live exam.py uses FRAME_SKIP=2)")
    args = ap.parse_args()

    clips_dir = Path(args.clips)
    if not clips_dir.is_dir():
        print(f"error: {clips_dir} not found. Record the clips first - see "
              f"RECORDING_PROTOCOL.md in the paper folder.", file=sys.stderr)
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
        return 1

    print(f"processing {len(good)} clips (stride={args.stride}) ...")
    fm, hands, pose, yolo = build_detectors()

    all_rows = []
    for i, p in enumerate(good, 1):
        rows = process_clip(p, fm, hands, pose, yolo, stride=args.stride)
        all_rows.extend(rows)
        print(f"  [{i}/{len(good)}] {p.name:22} -> {len(rows):4d} frames")

    df = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nwrote {out}")
    print(f"  {len(df)} frames, {df['clip_id'].nunique()} clips, "
          f"{df['session_id'].nunique()} sessions")
    print(f"  class balance: {df['label'].value_counts().sort_index().to_dict()}")
    print(f"  phone-positive frames: {int(df['phone_detected'].sum())}")
    print("\nnext: uv run python scripts/eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

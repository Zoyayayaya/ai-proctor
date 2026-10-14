#!/usr/bin/env python
"""
Analyse a saved video: the seven features plus real head pose (yaw/pitch/roll).

Why head pose matters here
--------------------------
The paper's feature set has **no head-pitch measure**. `head_tilt` is
`abs(left_eye_y - right_eye_y)`, which is head *roll*, not how far the head is
nodded down. That is exactly why "looking down at notes" (scenario S6) is hard
for the system, and it is the single biggest methodological weakness in the
paper.

This script recovers the missing angles by solving the standard 3D-face-model
PnP problem on the same FaceMesh landmarks the live pipeline already produces.
It costs almost nothing (same landmarks, a 6-point solve) and it lets us:

  * measure what the head was actually doing in a recorded session
  * quantify how much of the error comes from the missing pitch signal
  * decide, on evidence, whether a pitch feature is worth adding to the paper

It is a diagnostic first. Whether it becomes a paper feature is a separate call.

Works on any video: a session recording, a clip from the recording protocol, or
an OEP subject. Note that analysing an *annotated* session video means the
burned-in overlay is in the pixels; the top band and corners are covered, so
prefer the raw video when you have it (`--keep-raw`).

Usage
-----
    uv run python scripts/analyse_video.py --video data/sessions/<id>/video_annotated.mp4
    uv run python scripts/analyse_video.py --video clip.mp4 --annotate
    uv run python scripts/analyse_video.py --video clip.mp4 --no-yolo   # fast
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_score as rs  # noqa: E402
from detector import PHONE_CLASS_ID, PHONE_CONF  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# A realistically PROPORTIONED head model, mm, +Y down, +Z forward.
# Landmark 1 is the nose TIP so it sits proud of the eye plane, which is why the
# eyes and mouth have negative Z. Scale does not change the recovered rotation,
# but a sane aspect ratio keeps the solver out of the flipped basin - with a
# badly proportioned model every solver tested converged to pitch = -180.
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],        # 1   nose tip
    [0.0, 120.0, -18.0],    # 152 chin
    [-45.0, -35.0, -40.0],  # 33  left eye, outer corner
    [45.0, -35.0, -40.0],   # 263 right eye, outer corner
    [-30.0, 70.0, -32.0],   # 61  left mouth corner
    [30.0, 70.0, -32.0],    # 291 right mouth corner
], dtype=np.float64)
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# A 6-point PnP on a frontal model degrades past roughly 45 degrees of yaw, and
# a generic model never matches an individual face exactly. Both bounds are
# deliberately loose: the goal is to drop obviously broken solves, not to
# pretend the remainder are precise measurements.
MAX_PNP_ERR_PX = 18.0
MAX_ABS_YAW = 70.0
MAX_ABS_PITCH = 80.0

# Absolute angles from a generic model and an assumed focal length carry a
# systematic bias of up to ~15 degrees. What is reliable is the CHANGE relative
# to the subject's own neutral pose, so the analysis reports both and the
# paper-facing quantity is the deviation.
BASELINE_FRAMES = 20

FEATURES = ("eye_ratio", "head_tilt", "body_tilt", "looking_up",
            "hand_on_table", "phone_detected")


def head_pose(landmarks_px, w, h, prev=None):
    """
    6-point PnP -> (pitch, yaw, roll) in degrees plus reprojection error.

    `prev` is the previous frame's (rvec, tvec). Seeding the solver with it is
    what makes this usable on video: unseeded, the solver jumps into the mirrored
    basin and reports pitch = -180 for a frontal face.

    The seed must be a SENSIBLE starting pose, not None. An unseeded first solve
    can land in the flipped basin, and temporal continuity then propagates that
    flip for the rest of the video - which is exactly what happened here: pitch
    was fine but roll sat near 180 for the whole session. The caller seeds with
    identity (head facing camera), which is the right prior for someone sitting
    at a desk.

    A large frame-to-frame jump is also treated as a flip and re-solved from
    identity, so one bad frame cannot poison the rest.

    Returns None when the solve is not trustworthy.
    """
    import cv2
    image_points = np.array([landmarks_px[i] for i in LANDMARK_IDS], dtype=np.float64)
    focal = float(w)
    cam = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))

    IDENTITY = (np.zeros((3, 1)), np.array([[0.0], [0.0], [600.0]]))

    def solve(seed):
        return cv2.solvePnP(MODEL_POINTS, image_points, cam, dist,
                            rvec=seed[0].copy(), tvec=seed[1].copy(),
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_ITERATIVE)

    def angles(rvec):
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        return (math.degrees(math.atan2(rmat[2, 1], rmat[2, 2])),
                math.degrees(math.atan2(-rmat[2, 0], sy)),
                math.degrees(math.atan2(rmat[1, 0], rmat[0, 0])))

    def error(rvec, tvec):
        proj, _ = cv2.projectPoints(MODEL_POINTS, rvec, tvec, cam, dist)
        return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - image_points, axis=1)))

    seed = prev if prev is not None else IDENTITY
    ok, rvec, tvec = solve(seed)
    if not ok:
        return None
    p, y, r = angles(rvec)
    err = error(rvec, tvec)

    # implausible, or a jump big enough to be a flip -> retry from identity
    jumped = prev is not None and (abs(p - prev[2]) > 45 or abs(r - prev[3]) > 45)
    if abs(p) > MAX_ABS_PITCH or abs(y) > MAX_ABS_YAW or jumped:
        ok2, rvec2, tvec2 = solve(IDENTITY)
        if ok2:
            p2, y2, r2 = angles(rvec2)
            e2 = error(rvec2, tvec2)
            if e2 <= err * 1.6 and abs(p2) < MAX_ABS_PITCH and abs(y2) < MAX_ABS_YAW:
                rvec, tvec, p, y, r, err = rvec2, tvec2, p2, y2, r2, e2

    if err > MAX_PNP_ERR_PX or abs(yaw := y) > MAX_ABS_YAW or abs(p) > MAX_ABS_PITCH:
        return None
    return p, y, r, err, (rvec, tvec, p, r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="", help="output dir (default: alongside the video)")
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame")
    ap.add_argument("--no-yolo", action="store_true", help="skip phone detection (fast)")
    ap.add_argument("--no-pose", action="store_true", help="skip body pose")
    ap.add_argument("--annotate", action="store_true",
                    help="also write an annotated video with the angles drawn")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    import cv2
    import mediapipe as mp

    vid = Path(args.video)
    if not vid.exists():
        print(f"error: {vid} not found", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else vid.parent / (vid.stem + "_analysis")
    out.mkdir(parents=True, exist_ok=True)

    fm = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
    ps = None if args.no_pose else mp.solutions.pose.Pose(
        min_detection_confidence=0.7, min_tracking_confidence=0.7)
    yolo = None
    if not args.no_yolo:
        from detector import pick_device
        from ultralytics import YOLO
        yolo = YOLO("yolov5s.pt")
        dev = pick_device(args.device)

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        print(f"error: cannot open {vid}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  video   : {vid.name}")
    print(f"  {w}x{h} @ {fps:.1f} fps, {total} frames "
          f"({total/fps:.1f}s)  stride={args.stride}")
    print(f"  output  : {out}")
    print(f"  stages  : face{' + pose' if ps else ''}"
          f"{' + yolo' if yolo else ''} + head pose (PnP)")
    print()

    writer = None
    if args.annotate:
        writer = cv2.VideoWriter(str(out / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    recs = []
    idx = 0
    shoulder = []
    pose_prev = None          # carries rvec/tvec forward for solver continuity
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride:
            idx += 1
            continue
        t = idx / fps
        idx += 1
        fh, fw, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = dict(t=round(t, 3), frame=idx - 1, face=0, eye_ratio=None,
                 head_tilt=None, body_tilt=0, looking_up=0, hand_on_table=0,
                 phone_detected=0, pitch=None, yaw=None, roll=None, pnp_err=None)

        res = fm.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            def px(i):
                return (int(lm[i].x * fw), int(lm[i].y * fh))

            lx, ly = px(33)
            rx, ry = px(263)
            _, ny = px(1)
            r["face"] = 1
            r["eye_ratio"] = round(((lx + rx) // 2) / fw, 5)
            r["head_tilt"] = abs(ly - ry)
            r["looking_up"] = int(ny < fh * rs.LOOKING_UP_THRESHOLD)
            hp = head_pose([px(i) for i in range(468)], fw, fh, pose_prev)
            if hp:
                p_, y_, r_, err_, pose_prev = hp
                r["pitch"] = round(p_, 2)
                r["yaw"] = round(y_, 2)
                r["roll"] = round(r_, 2)
                r["pnp_err"] = round(err_, 2)

        if ps is not None:
            pres = ps.process(rgb)
            if pres.pose_landmarks:
                ls = pres.pose_landmarks.landmark[11]
                rsh = pres.pose_landmarks.landmark[12]
                shoulder.append((ls.x, rsh.x))
                if len(shoulder) > 10:
                    shoulder.pop(0)
                al = sum(s[0] for s in shoulder) / len(shoulder)
                ar = sum(s[1] for s in shoulder) / len(shoulder)
                d = abs(al - ar)
                if d <= 0.6 and d < 0.35 and (
                        al > ar + rs.SHOULDER_TILT_THRESHOLD
                        or ar > al + rs.SHOULDER_TILT_THRESHOLD):
                    r["body_tilt"] = 1

        if yolo is not None:
            for y in yolo(frame, imgsz=args.imgsz, device=dev, verbose=False):
                for b in y.boxes:
                    if int(b.cls[0]) == PHONE_CLASS_ID and float(b.conf[0]) > PHONE_CONF:
                        r["phone_detected"] = 1

        r["suspicious_count"] = (
            int(r["eye_ratio"] is not None and abs(r["eye_ratio"] - 0.5) > 0.25)
            + int(r["head_tilt"] is not None and r["head_tilt"] > rs.HEAD_TILT_THRESHOLD)
            + int(bool(r["body_tilt"])) + int(bool(r["looking_up"])))
        recs.append(r)

        if writer is not None:
            a = frame.copy()
            if r["pitch"] is not None:
                cv2.putText(a, f"yaw {r['yaw']:+.1f}  pitch {r['pitch']:+.1f}  "
                               f"roll {r['roll']:+.1f}   err {r['pnp_err']:.1f}px",
                            (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2)
                cv2.putText(a, f"susp {r['suspicious_count']}  phone {r['phone_detected']}",
                            (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
            writer.write(a)

    cap.release()
    fm.close()
    if ps:
        ps.close()
    if writer:
        writer.release()

    # ---- outputs ---------------------------------------------------------
    with open(out / "frames.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    def stats(key, only_face=True):
        vals = [r[key] for r in recs
                if r[key] is not None and (not only_face or r["face"])]
        if not vals:
            return None
        a = np.array(vals, dtype=float)
        return dict(n=len(a), mean=round(float(a.mean()), 2),
                    sd=round(float(a.std()), 2), min=round(float(a.min()), 2),
                    p05=round(float(np.percentile(a, 5)), 2),
                    p50=round(float(np.percentile(a, 50)), 2),
                    p95=round(float(np.percentile(a, 95)), 2),
                    max=round(float(a.max()), 2))

    faces = sum(r["face"] for r in recs)

    # Baseline-relative angles. Absolute pose from a generic model carries a
    # systematic bias, so the meaningful quantity is the deviation from the
    # subject's own neutral pose, taken as the median of the opening frames.
    base = {}
    for k in ("pitch", "yaw", "roll"):
        vals = [r[k] for r in recs[:BASELINE_FRAMES] if r[k] is not None]
        if vals:
            base[k] = round(float(np.median(vals)), 2)
    for r in recs:
        for k in ("pitch", "yaw", "roll"):
            r[f"{k}_dev"] = (round(r[k] - base[k], 2)
                             if r[k] is not None and k in base else None)

    summary = dict(video=str(vid), frames_analysed=len(recs),
                   face_frames=faces,
                   face_pct=round(100 * faces / max(len(recs), 1), 1),
                   duration_s=round(len(recs) / fps * args.stride, 1),
                   head_pose_note=(
                       "Absolute angles come from a generic 3D model and an assumed "
                       "focal length, so they carry a systematic bias of up to ~15 deg. "
                       "The *_dev fields are deviations from the subject's own neutral "
                       "pose (median of the first %d valid frames) and are the "
                       "trustworthy quantity." % BASELINE_FRAMES),
                   head_pose_baseline=base,
                   head_pose_rejected=sum(1 for r in recs
                                          if r["face"] and r["pitch"] is None),
                   statistics={k: stats(k) for k in
                               ("eye_ratio", "head_tilt", "pitch_dev", "yaw_dev",
                                "roll_dev", "pnp_err")},
                   indicator_fires={k: sum(1 for r in recs if r[k]) for k in FEATURES},
                   suspicious_ge2=sum(1 for r in recs if r["suspicious_count"] >= 2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 76)
    print("ANALYSIS SUMMARY")
    print("=" * 76)
    print(f"  frames analysed : {len(recs)}   face found: {summary['face_pct']}%")
    print()
    print(f"  {'measure':12} {'mean':>8} {'sd':>7} {'p05':>8} {'p50':>8} "
          f"{'p95':>8} {'min':>8} {'max':>8}")
    print("  " + "-" * 74)
    for k, s in summary["statistics"].items():
        if s:
            print(f"  {k:12} {s['mean']:8.2f} {s['sd']:7.2f} {s['p05']:8.2f} "
                  f"{s['p50']:8.2f} {s['p95']:8.2f} {s['min']:8.2f} {s['max']:8.2f}")
    print()
    print("  indicator fires : " +
          ", ".join(f"{k}={v}" for k, v in summary["indicator_fires"].items() if v))
    print(f"  frames with >=2 indicators : {summary['suspicious_ge2']}")
    print()
    print(f"  written: {out}/frames.jsonl, {out}/summary.json"
          + (", annotated.mp4" if writer else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

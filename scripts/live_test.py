#!/usr/bin/env python
"""
Live camera validation, and guided clip recording.

Two modes:

  (default)  LIVE TEST - runs the real pipeline against the webcam, prints a
             live readout of the extracted features and the resulting score,
             and saves an annotated snapshot so the detection can be eyeballed.
             This is the fastest way to confirm the whole chain works before
             recording anything.

  --record   GUIDED RECORDING - walks through the scenarios in
             RECORDING_SPEC.md, prompts for each one, counts down, records a
             12-second clip and saves it under the correct filename. The
             filename is the label, so getting it right automatically removes
             the main source of labelling error.

macOS note: the camera must be authorised for whichever app launches this
(Settings > Privacy & Security > Camera). OpenCV reports
"not authorized to capture video" if it is not.

Usage
-----
    uv run python scripts/live_test.py                     # 20 s live test
    uv run python scripts/live_test.py --seconds 40 --show
    uv run python scripts/live_test.py --record --person 1
    uv run python scripts/live_test.py --record --person 1 --skip-done
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_score as rs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = ROOT / "data" / "clips"

# mirrors exam.py
CAM_INDEX = 0
WIDTH, HEIGHT = 1280, 720
PHONE_CLASS_ID = 67
PHONE_CONF = 0.1

SCENARIOS = {
    "S1": ("Focused on screen", 0,
           "Face the screen, eyes on it, sit naturally still."),
    "S2": ("Fidgeting", 0,
           "Shift weight, adjust posture, scratch your face, stretch. Do NOT turn or tilt your head."),
    "S3": ("Reading / nodding", 0,
           "Eyes on the screen, small nods, lean in slightly to read."),
    "S4": ("Look away", 1,
           "Turn your head clearly LEFT or RIGHT past your shoulder line and HOLD."),
    "S5": ("Head roll", 1,
           "Roll your head sideways, ear toward shoulder, and HOLD. Alternate sides across repeats."),
    "S6": ("Look down at notes", 1,
           "Tilt your head down as if reading notes in your lap and HOLD. (Hard for this system - expected.)"),
    "S7": ("Phone visible", 1,
           "Bring a phone UP into the camera's view, above the desk, and HOLD. Must be visible."),
    "S8": ("Lean / leave frame", 1,
           "Lean out sideways so your face partly leaves frame, or stand and step out, then return."),
}
CLIP_SECONDS = 12.0
HOLD_ADVICE = 6.0


def open_camera():
    import cv2
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("\n  Could not open the camera.\n")
        print("  On macOS this is almost always a permission issue, not a code one:")
        print("    System Settings > Privacy & Security > Camera")
        print("    -> enable the app you launched this from (iTerm / Terminal / opencode)")
        print("    -> fully quit and reopen that app, then try again")
        print()
        print("  OpenCV's own message if so is: \"not authorized to capture video\".")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


def build_detectors():
    import mediapipe as mp
    from ultralytics import YOLO
    fm = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hd = mp.solutions.hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    ps = mp.solutions.pose.Pose(
        min_detection_confidence=0.7, min_tracking_confidence=0.7)
    yolo = YOLO("yolov5s.pt")
    return fm, hd, ps, yolo


def frame_features(frame, fm, hd, ps, yolo, shoulder_hist):
    """One frame -> the seven features, mirroring exam.py exactly."""
    import cv2
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = {}
    f["face"] = False
    f["eye_ratio"] = f["head_tilt"] = float("nan")
    f["body_tilt"] = f["looking_up"] = f["hand_on_table"] = f["phone_detected"] = 0

    res = fm.process(rgb)
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        def px(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        lx, ly = px(33)
        rx, ry = px(263)
        _, ny = px(1)
        f["face"] = True
        f["eye_ratio"] = ((lx + rx) // 2) / w
        f["head_tilt"] = abs(ly - ry)
        f["looking_up"] = int(ny < h * rs.LOOKING_UP_THRESHOLD)

    hres = hd.process(rgb)
    if hres.multi_hand_landmarks:
        for hand in hres.multi_hand_landmarks:
            if hand.landmark[0].y > rs.TABLE_REGION_THRESHOLD:
                f["hand_on_table"] = 1

    pres = ps.process(rgb)
    if pres.pose_landmarks:
        ls = pres.pose_landmarks.landmark[11]
        rsh = pres.pose_landmarks.landmark[12]
        shoulder_hist.append((ls.x, rsh.x))
        if len(shoulder_hist) > 10:
            shoulder_hist.pop(0)
        al = float(np.mean([s[0] for s in shoulder_hist]))
        ar = float(np.mean([s[1] for s in shoulder_hist]))
        d = abs(al - ar)
        if d <= 0.6 and d < 0.35:
            if al > ar + rs.SHOULDER_TILT_THRESHOLD or ar > al + rs.SHOULDER_TILT_THRESHOLD:
                f["body_tilt"] = 1

    for r in yolo(frame, verbose=False):
        for b in r.boxes:
            if int(b.cls[0]) == PHONE_CLASS_ID and float(b.conf[0]) > PHONE_CONF:
                f["phone_detected"] = 1
    return f


def annotate(frame, f, score, decision):
    import cv2
    out = frame.copy()
    col = (0, 0, 255) if decision else (0, 200, 0)
    lines = [
        f"score {score:3d}   {'CHEATING' if decision else 'safe'}",
        f"eye_ratio {f['eye_ratio']:.2f}" if np.isfinite(f["eye_ratio"]) else "eye_ratio  --",
        f"head_tilt {f['head_tilt']:.0f}px" if np.isfinite(f["head_tilt"]) else "head_tilt  --",
        f"body {f['body_tilt']}  up {f['looking_up']}  hand {f['hand_on_table']}  phone {f['phone_detected']}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(out, t, (12, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    if not f["face"]:
        cv2.putText(out, "NO FACE", (12, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    return out


# --------------------------------------------------------------------------
def live_test(args):
    cap = open_camera()
    if cap is None:
        return 1
    fm, hd, ps, yolo = build_detectors()
    hist, scores, indicators = [], [], defaultdict(int)
    frames = faces = 0
    t_end = time.time() + args.seconds
    last_print = 0.0
    snapshot = None
    print(f"\nlive test for {args.seconds:.0f}s  -  Ctrl-C to stop early\n")
    print(f"  {'t':>5} {'score':>5} {'decision':>9}  eye   tilt  body  up    hand  phone")
    print("  " + "-" * 68)
    try:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            f = frame_features(frame, fm, hd, ps, yolo, hist)
            if f["face"]:
                faces += 1
            susp = rs.suspicious_indicators(
                f["eye_ratio"] if np.isfinite(f["eye_ratio"]) else 0.5,
                f["head_tilt"] if np.isfinite(f["head_tilt"]) else 0, 
                f["body_tilt"], f["looking_up"])
            raw = rs.raw_score(susp, f["phone_detected"], 0.0)
            scores.append(raw)
            sm = rs.smooth(scores)[-1]
            dec = rs.hybrid_decision(sm, 0)
            if abs(f["eye_ratio"] - 0.5) > 0.25 if np.isfinite(f["eye_ratio"]) else False:
                indicators["eye_ratio"] += 1
            if np.isfinite(f["head_tilt"]) and f["head_tilt"] > rs.HEAD_TILT_THRESHOLD:
                indicators["head_tilt"] += 1
            if f["looking_up"]:
                indicators["looking_up"] += 1
            if f["body_tilt"]:
                indicators["body_tilt"] += 1
            if f["hand_on_table"]:
                indicators["hand_on_table"] += 1
            if f["phone_detected"]:
                indicators["phone_detected"] += 1
            snapshot = annotate(frame, f, sm, dec)
            if args.show:
                import cv2
                cv2.imshow("live test (q to quit)", snapshot)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                er = f"{f['eye_ratio']:.2f}" if np.isfinite(f["eye_ratio"]) else " -- "
                ht = f"{f['head_tilt']:.0f}" if np.isfinite(f["head_tilt"]) else " --"
                print(f"  {args.seconds - (t_end-now):5.0f} {sm:5d} "
                      f"{'CHEATING' if dec else 'safe':>9}  {er:>5} {ht:>5} "
                      f"{f['body_tilt']:^5} {f['looking_up']:^5} {f['hand_on_table']:^5} "
                      f"{f['phone_detected']:^6}")
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        cap.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("\n" + "=" * 72)
    print("LIVE TEST SUMMARY")
    print("=" * 72)
    if frames == 0:
        print("  no frames captured - the camera opened but produced nothing")
        return 1
    print(f"  frames captured        : {frames}")
    print(f"  face detected          : {faces} ({100*faces/frames:.0f}%)")
    if scores:
        print(f"  raw score  mean / max  : {np.mean(scores):.1f} / {max(scores)}")
        print(f"  smoothed score (final) : {rs.smooth(scores)[-1]}")
    print(f"  frames needing >=2     : {sum(1 for s in scores if s >= rs.SIGNAL_WEIGHT)}")
    print("  indicator fire counts  : " +
          (", ".join(f"{k}={v}" for k, v in sorted(indicators.items())) or "none fired"))
    print()
    if faces / frames < 0.5:
        print("  WARNING: a face was found in under half the frames.")
        print("  Sit closer, raise the lighting, and centre yourself in the frame.")
    else:
        print("  Face detection looks healthy.")
    print()
    print("  Pipeline end-to-end: camera -> MediaPipe/YOLO -> features -> score. OK")
    if snapshot is not None:
        out = ROOT / "figures" / "live_test_snapshot.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(out), snapshot)
        print(f"  annotated snapshot saved: {out}")
    return 0


# --------------------------------------------------------------------------
def record(args):
    import cv2
    cap = open_camera()
    if cap is None:
        return 1
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    todo = []
    for code, (name, _lab, _instr) in SCENARIOS.items():
        for rep in (1, 2, 3):
            p = CLIPS_DIR / f"{code}_p{args.person}_r{rep}.mp4"
            if p.exists() and args.skip_done:
                continue
            todo.append((code, rep, p))
    if not todo:
        print("nothing to record - all clips already exist (drop --skip-done to redo)")
        return 0

    print(f"\nGuided recording for person p{args.person}: {len(todo)} clips, "
          f"{CLIP_SECONDS:.0f}s each.\n")
    print("Follow the prompts. Hold the behaviour for at least "
          f"{HOLD_ADVICE:.0f}s in the middle of each clip.\n")
    try:
        for i, (code, rep, path) in enumerate(todo, 1):
            name, label, instr = SCENARIOS[code]
            print("=" * 72)
            print(f"  [{i}/{len(todo)}]  {code}  ({'CHEATING' if label else 'SAFE'})  "
                  f"repeat {rep}  ->  {path.name}")
            print(f"  {name}")
            print(f"  {instr}")
            print("=" * 72)
            input("  press Enter when ready (Ctrl-C to abort) ... ")
            for c in (3, 2, 1):
                print(f"   {c} ...", end="", flush=True)
                time.sleep(1)
            print(" RECORDING")

            vw = cv2.VideoWriter(str(path), fourcc, 15.0, (WIDTH, HEIGHT))
            if not vw.isOpened():
                print(f"  !! could not open a writer for {path}")
                return 1
            t0 = time.time()
            written = 0
            while time.time() - t0 < CLIP_SECONDS:
                ok, frame = cap.read()
                if not ok:
                    break
                vw.write(frame)
                written += 1
            vw.release()
            print(f"  done - {written} frames, {time.time()-t0:.1f}s -> {path.name}\n")
    except KeyboardInterrupt:
        print("\n  aborted")
        return 1
    finally:
        cap.release()

    print("=" * 72)
    print(f"recorded {len(todo)} clips into {CLIPS_DIR}")
    print()
    print("now verify them before anything else:")
    print("  uv run python scripts/check_clips.py --people " + str(args.person))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="guided clip recording")
    ap.add_argument("--person", type=int, default=1, help="person id for --record")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip clips that already exist")
    ap.add_argument("--seconds", type=float, default=20.0, help="live test duration")
    ap.add_argument("--show", action="store_true", help="also open a preview window")
    args = ap.parse_args()
    return record(args) if args.record else live_test(args)


if __name__ == "__main__":
    raise SystemExit(main())

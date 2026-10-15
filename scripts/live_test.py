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
from rule_score import PHONE_CLASS_ID, PHONE_CONF  # noqa: E402
from sysutil import camera_failure_report, open_camera, open_writer  # noqa: E402
import telemetry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = ROOT / "data" / "clips"

# mirrors exam.py
CAM_INDEX = 0
WIDTH, HEIGHT = 1280, 720

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
def camera_failed(where: str = "live_test", index: int = 0) -> None:
    """
    Explain a camera failure to the operator AND report it.

    The explanation is for the person in the room; the report is for whoever is
    watching the endpoint, so the cause can be diagnosed and a fix shipped
    without a second session.
    """
    print("\n  Could not open the camera.\n")
    print("  This is almost always a permission issue, not a code problem.\n")
    if sys.platform.startswith("win"):
        print("    Settings > Privacy & security > Camera")
        print("      -> turn on 'let desktop apps access your camera'")
        print("      -> then fully close and reopen this terminal")
        print("      -> and check no other app (Teams, Zoom, Camera) is using it")
    else:
        print("    System Settings > Privacy & Security > Camera")
        print("      -> enable the app you launched this from")
        print("      -> fully quit and reopen that app, then try again")
    print()
    r = camera_failure_report(where, index)
    print(f"  Reported: {'sent - the failure and its context are on their way' if r.get('sent') else r.get('detail')}")
    print("  Nobody needs to debug this here; a fix can come back to you.\n")


def live_test(args):
    cap = open_camera(CAM_INDEX, WIDTH, HEIGHT, 15.0)
    if cap is None:
        camera_failed("live_test", CAM_INDEX)
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
    fps_seen = frames / max(args.seconds, 1e-6)
    print(f"  frames captured        : {frames}  ({fps_seen:.1f} fps)")
    print(f"  face detected          : {faces} ({100*faces/frames:.0f}%)")
    if scores:
        print(f"  raw score  mean / max  : {np.mean(scores):.1f} / {max(scores)}")
        print(f"  smoothed score (final) : {rs.smooth(scores)[-1]}")
    print(f"  frames needing >=2     : {sum(1 for s in scores if s >= rs.SIGNAL_WEIGHT)}")
    print("  indicator fire counts  : " +
          (", ".join(f"{k}={v}" for k, v in sorted(indicators.items())) or "none fired"))
    print()
    # The rate matters twice over: it sets the file's duration when recording
    # (frames / fps), and extract_features reads that timeline back for the
    # `duration` feature. A camera quietly running at 1 fps corrupts both, and
    # nothing downstream can tell. Say it here, before anything is recorded.
    if fps_seen < 5:
        print(f"  WARNING: the camera is delivering {fps_seen:.1f} frames per second.")
        print("  A webcam normally manages 15 to 30. This is almost always low light -")
        print("  cameras lengthen their exposure in the dark and the rate collapses.")
        print("  Add a lamp in front of you, or move somewhere brighter, and run this")
        print("  again. Do NOT start recording until this number is well above 1.")
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
def measure_fps(cap, seconds: float = 1.0) -> float:
    """
    How fast is this camera actually delivering frames?

    The writer's fps sets the file's duration (frames / fps), and both
    check_clips.py and extract_features.py read that timeline back out of the
    file. A webcam asked for 15 fps may deliver 1, so the rate has to be measured
    rather than assumed.

    Reads and discards frames for a moment, which doubles as camera warm-up.
    Never returns less than 1.0: a stalled camera must not yield a zero-fps
    writer or a division by zero downstream.
    """
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    dt = time.time() - t0
    return max(1.0, n / dt) if dt > 0 else 1.0


def record(args):
    import cv2
    cap = open_camera(CAM_INDEX, WIDTH, HEIGHT, 15.0)
    if cap is None:
        camera_failed("live_test.record", CAM_INDEX)
        return 1
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    # codec chosen by sysutil.open_writer with a fallback chain
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
    print("Progress is reported after every clip, so whoever is watching the")
    print("other end can see the session advancing rather than waiting blind.\n")

    done_before = 24 - len(todo)
    telemetry.report("recording_started",
                     {"person": args.person, "clips_to_record": len(todo),
                      "clips_already_present": done_before,
                      "scenarios": sorted(SCENARIOS)},
                     session=f"p{args.person}", run_label=args.run_label,
                     enabled=not args.no_telemetry)
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

            # ------------------------------------------------------------------
            # The writer's fps decides the file's DURATION: duration = frames/fps.
            # Both check_clips.py and extract_features.py read that timeline back
            # out of the file, and extract_features uses it for the `duration`
            # feature - how long a suspicious state held.
            #
            # A webcam asked for 15 fps may deliver 1. Writing 12 frames at a
            # declared 15 fps turns a 12-second clip into a 0.8-second one: it
            # fails the >=10s check and the duration feature tops out at 0.8s
            # instead of 12s. Measure the real rate and write at that.
            # ------------------------------------------------------------------
            fps_real = measure_fps(cap)
            vw, _codec = open_writer(path, fps_real, (WIDTH, HEIGHT))
            if vw is None:
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
            elapsed_s = time.time() - t0

            # If the rate moved after the probe, the file's timeline is wrong and
            # the clip should not be trusted. Say so rather than let it through.
            fps_actual = written / elapsed_s if elapsed_s else 0.0
            drift = abs(fps_actual - fps_real) / fps_real if fps_real else 0.0
            if drift > 0.20:
                print(f"  !! camera rate moved during the clip: probed "
                      f"{fps_real:.1f} fps, achieved {fps_actual:.1f} fps.")
                print(f"     The clip's timeline is off by {drift*100:.0f}% - "
                      f"re-record {path.name}.")

            print(f"  done - {written} frames, {elapsed_s:.1f}s, "
                  f"{fps_real:.1f} fps -> {path.name}")
            print(f"  progress reported: {i}/{len(todo)} clips this session\n")

            telemetry.report(
                "clip_recorded",
                {"person": args.person, "scenario": code, "repeat": rep,
                 "scenario_name": name, "label": "cheating" if label else "safe",
                 "filename": path.name, "frames": written,
                 "seconds": round(elapsed_s, 1),
                 "fps": round(fps_real, 2),
                 "fps_actual": round(fps_actual, 2),
                 "fps_drift": round(drift, 3),
                 "progress": f"{i}/{len(todo)}",
                 "total_for_person": 24},
                session=f"p{args.person}", run_label=args.run_label,
                enabled=not args.no_telemetry)
    except KeyboardInterrupt:
        print("\n  aborted")
        telemetry.report("recording_aborted",
                         {"person": args.person, "recorded_so_far": len(todo)},
                         session=f"p{args.person}", run_label=args.run_label,
                         enabled=not args.no_telemetry)
        return 1
    except Exception as e:  # noqa: BLE001
        telemetry.report_error("live_test.record", e,
                               {"person": args.person},
                               session=f"p{args.person}",
                               enabled=not args.no_telemetry)
        raise
    finally:
        cap.release()

    telemetry.report("recording_finished",
                     {"person": args.person, "clips_recorded": len(todo)},
                     session=f"p{args.person}", run_label=args.run_label,
                     enabled=not args.no_telemetry)

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
    ap.add_argument("--no-telemetry", action="store_true",
                    help="do not report progress or failures")
    ap.add_argument("--run-label", default="", help="label for this run")
    args = ap.parse_args()
    telemetry.install_error_reporter(
        "live_test", enabled=not args.no_telemetry,
        run_label=args.run_label)
    return record(args) if args.record else live_test(args)


if __name__ == "__main__":
    raise SystemExit(main())

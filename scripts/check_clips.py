#!/usr/bin/env python
"""
Validate the recorded clips BEFORE they are processed.

The whole evaluation rests on these clips, and a misnamed file or an unusable
recording is not obvious by eye. This checker reads every clip, runs the same
MediaPipe face detector the pipeline uses, and reports whether the clips will
actually produce usable data - so problems are caught on the recording machine
rather than after the fact.

Checks:
  * filename matches <scenario>_p<person>_r<repeat>.<ext>
  * every scenario x person x repeat is present, with no strays
  * the clip opens, and its resolution and duration are sufficient
  * a face is detected in a high fraction of sampled frames
  * the rule engine's indicators behave as expected for safe vs cheating scenes
    (S6 is expected to be weak - the feature set has no head-pitch measure)

Usage
-----
    uv run python scripts/check_clips.py
    uv run python scripts/check_clips.py --stride 10      # faster, samples less
    uv run python scripts/check_clips.py --clips path/to/clips
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_score as rs  # noqa: E402
import telemetry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIPS = ROOT / "data" / "clips"

EXPECTED_SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
SAFE = {"S1", "S2", "S3"}
# S6 ("look down at notes") cannot be seen by the current feature set: head_tilt
# measures roll, not pitch. It is expected to be the weakest scenario.
EXPECTED_HARD = {"S6"}

# S7's signal IS the phone.
#
# The score is 20*[susp>=2] + 40*phone + min(4*duration, 40) - the phone is its
# own term, not one of the behavioural indicators. S7 exists to exercise that
# term: the subject holds a phone up without moving their head, so the
# behavioural indicators are correctly quiet.
#
# Judging S7 by the behavioural fraction therefore reports a failure on a clip
# that is working exactly as designed. It did: the operator's S7 passed its phone
# line and still showed as weak cheating, and they had to ask whether it was
# acceptable. Judging it by the phone fraction instead is no better - the phone
# bar is 30% and this one is 50%, so a good take can pass one and fail the other.
#
# S7 gets one verdict, and it is the phone line. Same reasoning as S6: the check
# should not double-judge a scenario on a signal it was never meant to produce.
PHONE_SCENARIOS = {"S7"}

FNAME = re.compile(r"^(S[1-8])_p(\d+)_r(\d+)$")
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

MIN_DURATION_S = 10.0
MIN_WIDTH, MIN_HEIGHT = 1280, 720
MIN_FACE_FRACTION = 0.80          # sampled frames that must contain a face
MIN_CHEAT_SUSPICIOUS = 0.50       # cheating clips: frames with >=1 indicator
MAX_SAFE_SUSPICIOUS = 0.20        # safe clips: frames with >=1 indicator

# ---------------------------------------------------------------------------
# The phone
#
# phone_detected is the heaviest signal in the score (+40, and binary), and the
# one S7 exists to exercise. This checker did not run it: it verified face,
# framing, naming and three of the five indicators, so a clip whose phone never
# fired could still pass. That is exactly what happened to S7 on the first pass -
# it was caught, but as "weak cheating", which points at the wrong cause.
#
# YOLO is the most expensive stage by a distance, so it can be sampled more
# sparsely than the face work. 1 means every sampled frame, which is the default
# because a 12-second clip only yields ~36 sampled frames at the default stride -
# sampling those again would leave too few looks at the phone to mean anything.
# Raise it only if the check is too slow on a given machine, and accept that a
# phone appearing briefly in a safe clip could then slip past.
#
# NOTE the counter is `sampled`, not `idx`: `idx` is incremented again after the
# stride `continue`, so it runs 1, 6, 11, 16... at stride 5 and never lands on a
# multiple of 10. Using it here silently checked nothing at all.
# ---------------------------------------------------------------------------
PHONE_EVERY = 1
PHONE_IMGSZ = 640                 # matches the offline extraction default
MIN_PHONE_FRAMES = 0.30           # S7: the phone must be seen in >=30% of samples


def load_phone_model():
    """
    YOLO, or None.

    Worth having, not worth failing over: a machine without the detect extra
    should still get every other check rather than nothing at all.
    """
    try:
        from ultralytics import YOLO
        return YOLO("yolov5s.pt")
    except Exception as e:  # noqa: BLE001
        print(f"  note: phone detection unavailable ({type(e).__name__}) - "
              f"S7 will not be verified\n")
        return None


def phone_visible(model, frame) -> bool:
    """Same class id and same threshold as the evaluation, so this measures it."""
    if model is None:
        return False
    try:
        res = model(frame, imgsz=PHONE_IMGSZ, verbose=False)[0]
        for b in res.boxes:
            if int(b.cls[0]) == rs.PHONE_CLASS_ID and float(b.conf[0]) > rs.PHONE_CONF:
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def analyse(path: Path, stride: int, phone_model=None):
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (n / fps) if fps else 0.0

    fm = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    sampled = with_face = flagged = 0
    phone_frames = phone_hits = 0
    brightness = 0.0
    # indicator coverage: how often each behavioural signal fires
    fires = defaultdict(int)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue
        idx += 1
        sampled += 1
        # Mean brightness is cheap here and it is the one number that settles
        # arguments about lighting. Without it, "the room is dim" is an opinion.
        brightness += float(frame.mean())

        # The phone is independent of the face, and it is the heaviest signal in
        # the score. Checked before the face work so a frame with no face still
        # gets its phone looked at - S8 leaves frame, and a phone there matters.
        if phone_model is not None and sampled % PHONE_EVERY == 0:
            phone_frames += 1
            if phone_visible(phone_model, frame):
                phone_hits += 1
                fires["phone_detected"] += 1

        fh, fw, _ = frame.shape
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        with_face += 1
        lm = res.multi_face_landmarks[0].landmark

        def px(i):
            return int(lm[i].x * fw), int(lm[i].y * fh)

        lx, ly = px(33)
        rx, ry = px(263)
        nx, ny = px(1)
        eye_ratio = ((lx + rx) // 2) / fw
        head_tilt = abs(ly - ry)
        looking_up = int(ny < fh * rs.LOOKING_UP_THRESHOLD)
        # pose is not run here: the checker is about framing and naming, and
        # running BlazePose as well would triple the runtime for little benefit
        susp = rs.suspicious_indicators(eye_ratio, head_tilt, 0, looking_up)
        if abs(eye_ratio - 0.5) > rs.EYE_OFF_CENTRE:
            fires["eye_ratio"] += 1
        if head_tilt > rs.HEAD_TILT_THRESHOLD:
            fires["head_tilt"] += 1
        if looking_up:
            fires["looking_up"] += 1
        if susp >= 1:
            flagged += 1

    cap.release()
    fm.close()

    return {
        "fps": fps, "frames": n, "w": w, "h": h, "duration": duration,
        "sampled": sampled, "with_face": with_face, "flagged": flagged,
        "phone_frames": phone_frames, "phone_hits": phone_hits,
        "mean_brightness": round(brightness / sampled, 1) if sampled else 0.0,
        "fires": dict(fires),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=str(DEFAULT_CLIPS))
    ap.add_argument("--stride", type=int, default=5,
                    help="sample every Nth frame (default 5; lower is slower)")
    ap.add_argument("--people", default="",
                    help="comma-separated person ids to expect, e.g. 1,2")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="do not report the result or failures")
    args = ap.parse_args()

    clips_dir = Path(args.clips)
    if not clips_dir.is_dir():
        print(f"error: {clips_dir} not found.")
        print("  Record the clips per RECORDING_SPEC.md, then put them here.")
        telemetry.report_error("check_clips.no_clips",
                               FileNotFoundError(str(clips_dir)),
                               {"clips_dir": str(clips_dir)},
                               enabled=not args.no_telemetry)
        return 1

    vids = sorted(p for p in clips_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    if not vids:
        print(f"error: no video files in {clips_dir}")
        telemetry.report_error("check_clips.no_videos",
                               ValueError("no video files"),
                               {"clips_dir": str(clips_dir)},
                               enabled=not args.no_telemetry)
        return 1

    phone_model = load_phone_model()
    phone_available = phone_model is not None

    print(f"checking {len(vids)} clips in {clips_dir}  (stride={args.stride})\n")

    parsed, bad_names, rows = [], [], []
    for v in vids:
        m = FNAME.match(v.stem)
        if not m:
            bad_names.append(v.name)
            continue
        scen, person, rep = m.group(1), int(m.group(2)), int(m.group(3))
        parsed.append((scen, person, rep, v))

    print("=" * 104)
    print(f"{'clip':16} {'res':>10} {'fps':>5} {'dur':>6} "
          f"{'face%':>6} {'flagged%':>9} {'bright':>7}  indicators")
    print("=" * 104)

    for scen, person, rep, v in sorted(parsed):
        a = analyse(v, args.stride, phone_model)
        if a is None:
            print(f"{v.name:16} UNREADABLE")
            rows.append((scen, person, rep, v, None))
            continue
        facepct = 100 * a["with_face"] / a["sampled"] if a["sampled"] else 0
        flagpct = 100 * a["flagged"] / a["with_face"] if a["with_face"] else 0
        fires = " ".join(f"{k}:{v_}" for k, v_ in sorted(a["fires"].items())) or "-"
        print(f"{v.name:16} {a['w']}x{a['h']:<6} {a['fps']:5.0f} {a['duration']:5.1f}s "
              f"{facepct:5.1f}% {flagpct:8.1f}% {a['mean_brightness']:6.1f}  {fires}")
        rows.append((scen, person, rep, v, a))

    # ---- presence grid ---------------------------------------------------
    people = sorted({p for _, p, _, _ in parsed})
    if args.people:
        people = [int(x) for x in args.people.split(",") if x.strip()]
    print()
    print("=" * 104)
    print("PRESENCE GRID   (rows = scenario, cols = person; value = repeats found)")
    print("=" * 104)
    header = "scenario  " + "".join(f"  p{p:<6}" for p in people)
    print(header)
    missing = []
    for s in EXPECTED_SCENARIOS:
        cells = []
        for p in people:
            found = sorted(r for sc, pp, r, _ in parsed if sc == s and pp == p)
            cells.append(f"  {len(found):<7}")
            if len(found) < 3:
                missing.append(f"{s}_p{p} (has {len(found)}/3)")
        print(f"{s:<10}" + "".join(cells))

    # ---- acceptance ------------------------------------------------------
    print()
    print("=" * 104)
    print("ACCEPTANCE CRITERIA")
    print("=" * 104)
    problems = []

    if bad_names:
        problems.append(f"{len(bad_names)} file(s) do not match the naming pattern: "
                        + ", ".join(bad_names[:5]))
    if missing:
        problems.append(f"{len(missing)} scenario/person group(s) incomplete: "
                        + ", ".join(missing[:6]))

    low_face, low_res, short = [], [], []
    weak_cheat, noisy_safe = [], []
    weak_phone, spurious_phone = [], []
    for scen, person, rep, v, a in rows:
        if a is None:
            problems.append(f"{v.name} could not be read")
            continue
        if a["w"] < MIN_WIDTH or a["h"] < MIN_HEIGHT:
            low_res.append(f"{v.name} ({a['w']}x{a['h']})")
        if a["duration"] < MIN_DURATION_S:
            short.append(f"{v.name} ({a['duration']:.1f}s)")
        facefrac = a["with_face"] / a["sampled"] if a["sampled"] else 0
        if facefrac < MIN_FACE_FRACTION:
            low_face.append(f"{v.name} ({100*facefrac:.0f}%)")
        flagfrac = a["flagged"] / a["with_face"] if a["with_face"] else 0
        if scen in SAFE and flagfrac > MAX_SAFE_SUSPICIOUS:
            noisy_safe.append(f"{v.name} ({100*flagfrac:.0f}%)")
        elif scen not in SAFE and scen not in EXPECTED_HARD \
                and scen not in PHONE_SCENARIOS:
            if flagfrac < MIN_CHEAT_SUSPICIOUS:
                weak_cheat.append(f"{v.name} ({100*flagfrac:.0f}%)")

        # The phone, kept separate because the cause is different from a posing
        # problem: the phone is either visible to the camera or it is not.
        if a["phone_frames"]:
            phonefrac = a["phone_hits"] / a["phone_frames"]
            if scen == "S7" and phonefrac < MIN_PHONE_FRAMES:
                weak_phone.append(f"{v.name} ({100*phonefrac:.0f}%)")
            elif scen in SAFE and phonefrac > MAX_SAFE_SUSPICIOUS:
                spurious_phone.append(f"{v.name} ({100*phonefrac:.0f}%)")

    per_clip = [
        {"clip": v.name, "scenario": scen, "brightness": a["mean_brightness"],
         "phone_pct": (round(100 * a["phone_hits"] / a["phone_frames"], 1)
                       if a["phone_frames"] else None)}
        for scen, person, rep, v, a in rows if a is not None
    ]

    def verdict(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if detail:
            print(f"         {detail}")

    verdict(not bad_names, "filenames match <scenario>_p<person>_r<repeat>",
            "; ".join(bad_names[:4]))
    verdict(not missing, "all 8 scenarios x each person x 3 repeats present",
            "; ".join(missing[:4]))
    verdict(not low_res, f"resolution >= {MIN_WIDTH}x{MIN_HEIGHT}", "; ".join(low_res[:4]))
    verdict(not short, f"duration >= {MIN_DURATION_S:.0f}s", "; ".join(short[:4]))
    verdict(not low_face, f"face detected in >= {MIN_FACE_FRACTION*100:.0f}% of frames "
                          f"(re-record these if they fail)", "; ".join(low_face[:5]))
    verdict(not noisy_safe, f"safe clips flagged in <= {MAX_SAFE_SUSPICIOUS*100:.0f}% of frames",
            "; ".join(noisy_safe[:4]))
    verdict(not weak_cheat,
            f"cheating clips flagged in >= {MIN_CHEAT_SUSPICIOUS*100:.0f}% of frames "
            f"(S6 excluded - expected hard; S7 judged on its phone line)",
            "; ".join(weak_cheat[:5]))

    # Only meaningful when YOLO loaded. Said plainly rather than silently skipped.
    if phone_available:
        verdict(not weak_phone,
                f"S7: phone visible in >= {MIN_PHONE_FRAMES*100:.0f}% of sampled frames",
                "; ".join(weak_phone[:4]))
        verdict(not spurious_phone,
                "safe clips: no phone detected", "; ".join(spurious_phone[:4]))
        if weak_phone:
            print("         If the phone was clearly held in frame and this still")
            print("         failed, that is a finding about the detector, not a bad")
            print("         take. Report it - do not re-record to make it pass.")
    else:
        print("  [SKIP] phone detection - ultralytics is not installed here, so S7")
        print("         was NOT verified. Install with: uv sync --extra detect")

    print()
    print("  NOTE on S6 (look down at notes): the feature set has no head-pitch")
    print("  measure, so S6 is expected to score low. That is a real finding to")
    print("  report, not a recording fault - do not re-record it for that reason.")

    # Blocking vs advisory.
    #
    # The hard checks are objective: a misnamed file, a missing clip, the wrong
    # resolution, too short, no face. Those make a clip unusable and there is
    # nothing to interpret.
    #
    # The behavioural checks are different in kind. A cheating clip that does not
    # look suspicious could mean the subject did not perform the behaviour - or it
    # could mean the system cannot see a behaviour that WAS performed, which is a
    # finding rather than a fault. A fraction alone cannot tell those apart, so
    # they are reported as observations instead of being made into a gate. That is
    # also why S6 and S7 are excluded from them.
    #
    # What this must not do is print FAIL on four lines and PASS at the bottom,
    # which is what it used to do. It cost real time: the operator read the PASS,
    # correctly, because that is what the tool said - and was then told by us that
    # the gate would fail on those same clips. Both readings were defensible and
    # neither was clear, and the tool caused that.
    blocking = [bad_names, missing, low_res, short, low_face]
    advisory = [noisy_safe, weak_cheat, weak_phone, spurious_phone]
    ok = not any(blocking)

    n_advisory = sum(1 for a in advisory if a)
    print()
    print("=" * 104)
    print("RESULT:", "READY - the clips can be processed" if ok
          else "NOT READY - fix the blocking items above")
    if ok and n_advisory:
        print(f"        with {n_advisory} behavioural observation(s) listed above.")
        print("        Those are not faults on their own. A clip that does not look")
        print("        the way its label says is either a bad take or a limit of the")
        print("        system, and the fraction cannot tell you which - so look at")
        print("        them, decide, and say which you think each one is. They are")
        print("        worth reporting either way.")
    print("=" * 104)

    # Report the verdict. A FAIL matters more than a PASS here: it means
    # someone has to re-record, and knowing that early is the whole point.
    telemetry.report(
        "clip_check",
        {"passed": ok, "clips_checked": len(vids),
         "problems": {"bad_names": bad_names[:8], "missing": missing[:8],
                      "low_resolution": low_res[:8], "too_short": short[:8],
                      "low_face_fraction": low_face[:8],
                      "noisy_safe_clips": noisy_safe[:8],
                      "weak_cheating_clips": weak_cheat[:8],
                      "weak_phone_clips": weak_phone[:8],
                      "spurious_phone_clips": spurious_phone[:8]},
         "counts": {"bad_names": len(bad_names), "missing": len(missing),
                    "low_resolution": len(low_res), "too_short": len(short),
                    "low_face_fraction": len(low_face),
                    "noisy_safe_clips": len(noisy_safe),
                    "weak_cheating_clips": len(weak_cheat),
                    "weak_phone_clips": len(weak_phone),
                    "spurious_phone_clips": len(spurious_phone)},
         # Lighting and phone coverage, per clip.
         #
         # These are the two things that have been argued about without data:
         # "the room is dim" and "the phone was visible". Both are measured here
         # and sent, so the next such question is settled from the record rather
         # than from inference. Brightness also gives a number to compare against
         # footage shot elsewhere.
         "lighting": {
             "mean": round(sum(c["brightness"] for c in per_clip) / len(per_clip), 1)
             if per_clip else None,
             "min": min((c["brightness"] for c in per_clip), default=None),
             "max": max((c["brightness"] for c in per_clip), default=None),
             "scale": "0-255 mean of sampled frames",
         },
         "phone": {
             "model_ran": phone_available,
             "clips_with_any_phone": sum(1 for c in per_clip if c["phone_pct"]),
             "per_clip_pct": {c["clip"]: c["phone_pct"] for c in per_clip
                              if c["phone_pct"] is not None},
         },
         "per_clip": per_clip},
        enabled=not args.no_telemetry)
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse as _ap
    _no = "--no-telemetry" in sys.argv
    telemetry.install_error_reporter("check_clips", enabled=not _no)
    raise SystemExit(main())

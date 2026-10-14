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

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIPS = ROOT / "data" / "clips"

EXPECTED_SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
SAFE = {"S1", "S2", "S3"}
# S6 ("look down at notes") cannot be seen by the current feature set: head_tilt
# measures roll, not pitch. It is expected to be the weakest scenario.
EXPECTED_HARD = {"S6"}

FNAME = re.compile(r"^(S[1-8])_p(\d+)_r(\d+)$")
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

MIN_DURATION_S = 10.0
MIN_WIDTH, MIN_HEIGHT = 1280, 720
MIN_FACE_FRACTION = 0.80          # sampled frames that must contain a face
MIN_CHEAT_SUSPICIOUS = 0.50       # cheating clips: frames with >=1 indicator
MAX_SAFE_SUSPICIOUS = 0.20        # safe clips: frames with >=1 indicator


def analyse(path: Path, stride: int):
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
    # indicator coverage: how often each of the four behavioural signals fires
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
        if abs(eye_ratio - 0.5) > 0.25:
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
    args = ap.parse_args()

    clips_dir = Path(args.clips)
    if not clips_dir.is_dir():
        print(f"error: {clips_dir} not found.")
        print("  Record the clips per RECORDING_SPEC.md, then put them here.")
        return 1

    vids = sorted(p for p in clips_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    if not vids:
        print(f"error: no video files in {clips_dir}")
        return 1

    print(f"checking {len(vids)} clips in {clips_dir}  (stride={args.stride})\n")

    parsed, bad_names, rows = [], [], []
    for v in vids:
        m = FNAME.match(v.stem)
        if not m:
            bad_names.append(v.name)
            continue
        scen, person, rep = m.group(1), int(m.group(2)), int(m.group(3))
        parsed.append((scen, person, rep, v))

    print("=" * 96)
    print(f"{'clip':16} {'res':>10} {'fps':>5} {'dur':>6} "
          f"{'face%':>6} {'flagged%':>9}  indicators")
    print("=" * 96)

    for scen, person, rep, v in sorted(parsed):
        a = analyse(v, args.stride)
        if a is None:
            print(f"{v.name:16} UNREADABLE")
            rows.append((scen, person, rep, v, None))
            continue
        facepct = 100 * a["with_face"] / a["sampled"] if a["sampled"] else 0
        flagpct = 100 * a["flagged"] / a["with_face"] if a["with_face"] else 0
        fires = " ".join(f"{k}:{v_}" for k, v_ in sorted(a["fires"].items())) or "-"
        print(f"{v.name:16} {a['w']}x{a['h']:<6} {a['fps']:5.0f} {a['duration']:5.1f}s "
              f"{facepct:5.1f}% {flagpct:8.1f}%  {fires}")
        rows.append((scen, person, rep, v, a))

    # ---- presence grid ---------------------------------------------------
    people = sorted({p for _, p, _, _ in parsed})
    if args.people:
        people = [int(x) for x in args.people.split(",") if x.strip()]
    print()
    print("=" * 96)
    print("PRESENCE GRID   (rows = scenario, cols = person; value = repeats found)")
    print("=" * 96)
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
    print("=" * 96)
    print("ACCEPTANCE CRITERIA")
    print("=" * 96)
    problems = []

    if bad_names:
        problems.append(f"{len(bad_names)} file(s) do not match the naming pattern: "
                        + ", ".join(bad_names[:5]))
    if missing:
        problems.append(f"{len(missing)} scenario/person group(s) incomplete: "
                        + ", ".join(missing[:6]))

    low_face, low_res, short = [], [], []
    weak_cheat, noisy_safe = [], []
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
        elif scen not in SAFE and scen not in EXPECTED_HARD:
            if flagfrac < MIN_CHEAT_SUSPICIOUS:
                weak_cheat.append(f"{v.name} ({100*flagfrac:.0f}%)")

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
            f"(S6 excluded - expected hard)", "; ".join(weak_cheat[:5]))

    print()
    print("  NOTE on S6 (look down at notes): the feature set has no head-pitch")
    print("  measure, so S6 is expected to score low. That is a real finding to")
    print("  report, not a recording fault - do not re-record it for that reason.")

    ok = not any([bad_names, missing, low_res, short, low_face])
    print()
    print("=" * 96)
    print("RESULT:", "PASS - the clips are ready to process" if ok
          else "FAIL - fix the items marked above, then re-run this check")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

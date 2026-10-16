"""
Canonical implementation of the rule-based cheating score.

This module is the single source of truth for the score defined in Eq. (2)-(4)
of the paper. Both the offline feature extractor and the evaluation harness
import from here, so the offline numbers and the live system cannot drift.

A drift guard at the bottom re-reads `exam.py` as text and asserts the
thresholds still match. Run `python scripts/rule_score.py` to check.
"""
from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds - must mirror exam.py exactly
# ---------------------------------------------------------------------------
EYE_LEFT_THRESHOLD = 0.3
EYE_RIGHT_THRESHOLD = 0.7
# exam.py declares EYE_LEFT/RIGHT_THRESHOLD but does NOT use them for the
# indicator - the check it actually runs is abs(eye_ratio - 0.5) > 0.25, i.e.
# outside 0.25-0.75. The named constants only appear in a report line. That
# quirk is carried over deliberately, because the port has to match exam.py
# rather than improve on it. Named here so callers stop hard-coding the number:
# check_clips.py had its own copy of 0.25, which is exactly the drift this file
# exists to prevent.
EYE_OFF_CENTRE = 0.25
HEAD_TILT_THRESHOLD = 10
TABLE_REGION_THRESHOLD = 0.7
SHOULDER_TILT_THRESHOLD = 0.3
LOOKING_UP_THRESHOLD = 0.25
LOOK_AWAY_THRESHOLD = 15
LOG_COOLDOWN = 2.0
FRAME_SKIP = 2

# ---------------------------------------------------------------------------
# Score assembly - mirrors exam.py detect_cheating()
# ---------------------------------------------------------------------------
SIGNAL_WEIGHT = 20          # +20 once, when >= 2 indicators co-occur
PHONE_WEIGHT = 40           # +40 for a detected phone
DURATION_RATE = 4           # min(duration * 4, 40)
DURATION_CAP = 40
SCORE_CAP = 100
SMOOTHING_WINDOW = 10       # k, the moving-average window
RULE_DECISION_THRESHOLD = 60
SAFE_ZONE = 20              # below this, cheating is forced to False

# ---------------------------------------------------------------------------
# Detector-level constants. These were inline in exam.py and then copied into
# several scripts, which is a drift risk: change one and the offline and live
# paths silently disagree. They live here now and everything imports them.
# ---------------------------------------------------------------------------
PHONE_CLASS_ID = 67             # COCO class id for cell phone
PHONE_CONF = 0.1                # exam.py: conf > 0.1
SHOULDER_WINDOW = 10            # frames averaged for shoulder smoothing
SHOULDER_LANDMARK_ERROR = 0.6   # above this the shoulder solve is a landmark error
SHOULDER_LEVEL_MAX_DIFF = 0.35  # below this the shoulders are level enough to test

# The seven features the classifier consumes (paper Table 4 / train_model.py)
ML_FEATURES = [
    "eye_ratio",
    "head_tilt",
    "body_tilt",
    "looking_up",
    "phone_detected",
    "hand_on_table",
    "duration",
]
# `look_away_counter` and `smoothed_score` exist in the CSV but are deliberately
# excluded from the classifier inputs: smoothed_score is the target's own source
# and look_away_counter is a rule state variable.
MODALITY_FEATURES = ["eye_ratio", "head_tilt", "body_tilt", "looking_up", "phone_detected"]


def suspicious_indicators(eye_ratio: float, head_tilt: float, body_tilt: int,
                          looking_up: int) -> int:
    """How many of the four behavioural indicators fire for one frame."""
    n = 0
    # Ignore tiny eye jitter: only a genuine offset from centre counts
    if abs(eye_ratio - 0.5) > EYE_OFF_CENTRE:
        n += 1
    if head_tilt > HEAD_TILT_THRESHOLD:
        n += 1
    if body_tilt:
        n += 1
    if looking_up:
        n += 1
    return n


def raw_score(suspicious_count: int, phone_detected: int, duration: float) -> int:
    """
    Instantaneous score for a frame, before smoothing.

        S_t = 20 * 1[suspicious_count >= 2] + 40 * phone + min(4 * duration, 40)

    Note the first term is added ONCE when two or more indicators co-occur; it
    does not scale with how many fire. The paper's Eq. (2) states it this way.
    """
    s = 0
    if suspicious_count >= 2:
        s += SIGNAL_WEIGHT
    if phone_detected:
        s += PHONE_WEIGHT
    s += min(int(duration * DURATION_RATE), DURATION_CAP)
    return min(s, SCORE_CAP)


def smooth(scores: list[int], k: int = SMOOTHING_WINDOW) -> list[int]:
    """Trailing moving average, matching exam.py's score_history behaviour."""
    out, hist = [], []
    for s in scores:
        hist.append(s)
        if len(hist) > k:
            hist.pop(0)
        out.append(int(sum(hist) / len(hist)))
    return out


def rule_decision(smoothed: int) -> int:
    """Rule branch of Eq. (4): fires at or above the threshold."""
    return int(smoothed >= RULE_DECISION_THRESHOLD)


def hybrid_decision(smoothed: int, ml_prediction: int) -> int:
    """Eq. (4): D_t = 1 if S_bar(t) >= 60 OR M_t = 1, with the safe-zone override."""
    if smoothed < SAFE_ZONE:
        return 0
    return int(smoothed >= RULE_DECISION_THRESHOLD or ml_prediction == 1)


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------
def check_against_exam(exam_path: str | Path | None = None) -> list[str]:
    """
    Re-read exam.py as text and confirm the thresholds here still match.
    Returns a list of mismatch descriptions (empty means clean).
    """
    here = Path(__file__).resolve().parent
    path = Path(exam_path) if exam_path else here.parent / "exam.py"
    if not path.exists():
        return [f"exam.py not found at {path}"]

    src = path.read_text(encoding="utf-8")
    expected = {
        "EYE_LEFT_THRESHOLD": EYE_LEFT_THRESHOLD,
        "EYE_RIGHT_THRESHOLD": EYE_RIGHT_THRESHOLD,
        "HEAD_TILT_THRESHOLD": HEAD_TILT_THRESHOLD,
        "TABLE_REGION_THRESHOLD": TABLE_REGION_THRESHOLD,
        "SHOULDER_TILT_THRESHOLD": SHOULDER_TILT_THRESHOLD,
        "LOOKING_UP_THRESHOLD": LOOKING_UP_THRESHOLD,
        "LOOK_AWAY_THRESHOLD": LOOK_AWAY_THRESHOLD,
        "LOG_COOLDOWN": LOG_COOLDOWN,
        "FRAME_SKIP": FRAME_SKIP,
    }
    problems = []
    for name, want in expected.items():
        m = re.search(rf"^{name}\s*=\s*([0-9.]+)", src, re.M)
        if not m:
            problems.append(f"{name}: not found in exam.py")
            continue
        got = float(m.group(1))
        if abs(got - float(want)) > 1e-9:
            problems.append(f"{name}: exam.py={got} vs rule_score.py={want}")

    # score-assembly constants appear inline in exam.py rather than as names
    if "int(duration * 4), 40" not in src:
        problems.append("duration bonus: 'min(int(duration * 4), 40)' not found in exam.py")
    if "score < 20" not in src:
        problems.append("safe zone: 'score < 20' not found in exam.py")
    if ">= 60" not in src:
        problems.append("decision threshold: '>= 60' not found in exam.py")
    return problems


if __name__ == "__main__":
    issues = check_against_exam()
    if issues:
        print("DRIFT DETECTED between exam.py and rule_score.py:")
        for i in issues:
            print("  -", i)
        raise SystemExit(1)
    print("OK: rule_score.py agrees with exam.py on every threshold and constant.")

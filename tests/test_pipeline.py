"""
Pipeline tests for the ICAIC-2026 revision.

These run without the recorded clips, so the whole chain can be verified before
the data exists. The extractor is exercised against a synthetic video with the
detectors stubbed out, which pins down the feature maths: if someone changes a
landmark index or a threshold, these tests fail.

    uv run pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import rule_score as rs  # noqa: E402

W, H = 320, 240


# --------------------------------------------------------------------------
# rule score
# --------------------------------------------------------------------------
def test_rule_score_matches_exam_py():
    """The canonical thresholds must still agree with exam.py."""
    problems = rs.check_against_exam(ROOT / "exam.py")
    assert problems == [], f"drift between exam.py and rule_score.py: {problems}"


def test_indicators_and_weights():
    # centred, upright, looking straight ahead -> nothing fires
    assert rs.suspicious_indicators(0.5, 0.0, 0, 0) == 0
    # a single indicator must NOT be enough
    assert rs.suspicious_indicators(0.8, 0.0, 0, 0) == 1
    assert rs.raw_score(1, 0, 0.0) == 0
    # two indicators fire the +20 once
    assert rs.suspicious_indicators(0.8, 20, 0, 0) == 2
    assert rs.raw_score(2, 0, 0.0) == 20
    # the +20 does not scale with how many indicators fire
    assert rs.raw_score(4, 0, 0.0) == 20
    # phone is worth +40 by itself
    assert rs.raw_score(0, 1, 0.0) == 40
    # duration bonus is min(4t, 40)
    assert rs.raw_score(0, 0, 3.0) == 12
    assert rs.raw_score(0, 0, 100.0) == 40
    # clamped at 100
    assert rs.raw_score(4, 1, 100.0) == 100


def test_smoothing_is_a_trailing_average():
    assert rs.smooth([10, 20, 30]) == [10, 15, 20]
    long_run = rs.smooth([100] * 20)
    assert long_run[-1] == 100
    assert rs.smooth([0] * 10 + [100])[-1] == 10  # 100/10


def test_decisions_and_safe_zone():
    assert rs.rule_decision(59) == 0
    assert rs.rule_decision(60) == 1
    assert rs.hybrid_decision(10, 1) == 0        # safe zone overrides the classifier
    assert rs.hybrid_decision(60, 0) == 1        # rule fires
    assert rs.hybrid_decision(30, 1) == 1        # classifier fires
    assert rs.hybrid_decision(30, 0) == 0


# --------------------------------------------------------------------------
# filename parsing / labelling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stem,expected", [
    ("S1_p1_r1", ("S1", 1, 1)),
    ("S4_p2_r3", ("S4", 2, 3)),
    ("S7_p1_r2", ("S7", 1, 2)),
])
def test_parse_name_accepts(stem, expected):
    import extract_features as ef
    assert ef.parse_name(stem) == expected


@pytest.mark.parametrize("stem", ["s6_p1_r1", "S9_p1_r1", "random", "S6_p1", "S6_p1_r"])
def test_parse_name_rejects(stem):
    import extract_features as ef
    assert ef.parse_name(stem) is None


def test_labels_come_from_scenario_not_from_score():
    import re
    import extract_features as ef
    safe = {"S1", "S2", "S3"}
    cheat = {"S4", "S5", "S6", "S7"}
    assert set(ef.SCENARIO_LABELS) == safe | cheat
    for s in safe:
        assert ef.SCENARIO_LABELS[s] == 0
    for s in cheat:
        assert ef.SCENARIO_LABELS[s] == 1

    # The whole point of this rewrite: the label is never derived from the rule
    # score. Strip docstrings/comments first, otherwise we match the prose that
    # explains what the OLD dataset did wrong.
    src = (ROOT / "scripts" / "extract_features.py").read_text()
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    code = re.sub(r"^\s*#.*$", "", code, flags=re.M)
    assert "score >= 60" not in code, "label must not be derived from the score"
    assert "smoothed_score" not in code, "label must not be derived from the score"
    # and the label really is taken from the scenario map
    assert re.search(r"label\s*=\s*SCENARIO_LABELS\[scen\]", code)


# --------------------------------------------------------------------------
# extractor against a synthetic video, detectors stubbed
# --------------------------------------------------------------------------
class _LM:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FaceMeshResult:
    def __init__(self, landmarks):
        if landmarks is None:
            self.multi_face_landmarks = None
        else:
            class _FL:
                def __init__(self, lm):
                    self.landmark = lm
            self.multi_face_landmarks = [_FL(landmarks)]


class _HandsResult:
    def __init__(self, wrist_y=None):
        if wrist_y is None:
            self.multi_hand_landmarks = None
        else:
            class _H:
                def __init__(self, lm):
                    self.landmark = lm
            self.multi_hand_landmarks = [_H([_LM(0.5, wrist_y)])]


class _PoseResult:
    def __init__(self, lx=0.3, rx=0.7):
        if lx is None:
            self.pose_landmarks = None
        else:
            lm = [_LM(0.5, 0.5) for _ in range(33)]
            lm[11] = _LM(lx, 0.5)
            lm[12] = _LM(rx, 0.5)

            class _P:
                def __init__(self, l):
                    self.landmark = l
            self.pose_landmarks = _P(lm)


class _Box:
    def __init__(self, cls, conf):
        self.cls = [cls]
        self.conf = [conf]


class _YoloResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _face(eye_l=0.4, eye_r=0.6, nose_y=0.5):
    lm = [_LM(0.5, 0.5) for _ in range(468)]
    lm[33] = _LM(eye_l, 0.5)
    lm[263] = _LM(eye_r, 0.5)
    lm[1] = _LM(0.5, nose_y)
    return lm


def _make_video(path, n=6, fps=5.0):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert vw.isOpened(), "could not open a VideoWriter for the test clip"
    for i in range(n):
        frame = np.full((H, W, 3), 40 + i * 10, dtype=np.uint8)
        vw.write(frame)
    vw.release()


def _stub_detectors(face_lm, wrist_y, lx, rx, boxes):
    class Face:
        def process(self, _rgb):
            return _FaceMeshResult(face_lm)

    class Hands:
        def process(self, _rgb):
            return _HandsResult(wrist_y)

    class Pose:
        def process(self, _rgb):
            return _PoseResult(lx, rx)

    class Yolo:
        def __call__(self, _frame, verbose=False):
            return [_YoloResult(boxes)]

    return Face(), Hands(), Pose(), Yolo()


def test_extractor_neutral_frame(tmp_path):
    """Centred, upright, no phone, shoulders level -> benign features, label 0."""
    import extract_features as ef
    vid = tmp_path / "S1_p1_r1.mp4"
    _make_video(vid)
    fm, hd, ps, yl = _stub_detectors(_face(), None, 0.3, 0.7, [])
    rows = ef.process_clip(vid, fm, hd, ps, yl)
    assert rows, "expected at least one processed frame"
    r = rows[0]
    assert r["eye_ratio"] == pytest.approx(0.5)
    assert r["head_tilt"] == 0
    assert r["body_tilt"] == 0
    assert r["looking_up"] == 0
    assert r["hand_on_table"] == 0
    assert r["phone_detected"] == 0
    assert r["label"] == 0
    assert r["scenario"] == "S1"
    assert r["session_id"] == "p1"


def test_extractor_phone_frame_is_labelled_positively(tmp_path):
    """A visible phone must be recorded, and the scenario label must be 1."""
    import extract_features as ef
    vid = tmp_path / "S6_p2_r1.mp4"
    _make_video(vid)
    fm, hd, ps, yl = _stub_detectors(_face(), None, 0.3, 0.7, [_Box(67, 0.55)])
    rows = ef.process_clip(vid, fm, hd, ps, yl)
    assert rows[0]["phone_detected"] == 1
    assert rows[0]["label"] == 1
    assert rows[0]["session_id"] == "p2"


def test_extractor_low_confidence_phone_is_ignored(tmp_path):
    """exam.py only counts a phone above confidence 0.1."""
    import extract_features as ef
    vid = tmp_path / "S6_p1_r1.mp4"
    _make_video(vid)
    fm, hd, ps, yl = _stub_detectors(_face(), None, 0.3, 0.7, [_Box(67, 0.05)])
    rows = ef.process_clip(vid, fm, hd, ps, yl)
    assert rows[0]["phone_detected"] == 0


def test_extractor_looking_up_and_head_tilt(tmp_path):
    """Nose high in frame and a vertical eye offset must register."""
    import extract_features as ef
    vid = tmp_path / "S5_p1_r1.mp4"
    _make_video(vid)
    lm = [_LM(0.5, 0.5) for _ in range(468)]
    lm[33] = _LM(0.4, 0.30)
    lm[263] = _LM(0.6, 0.50)
    lm[1] = _LM(0.5, 0.10)  # above h*0.25 = 60px
    fm, hd, ps, yl = _stub_detectors(lm, None, 0.3, 0.7, [])
    rows = ef.process_clip(vid, fm, hd, ps, yl)
    assert rows[0]["looking_up"] == 1
    assert rows[0]["head_tilt"] == abs(72 - 120)
    assert rows[0]["label"] == 1


def test_extractor_hand_on_table(tmp_path):
    import extract_features as ef
    vid = tmp_path / "S2_p1_r1.mp4"
    _make_video(vid)
    fm, hd, ps, yl = _stub_detectors(_face(), 0.85, 0.3, 0.7, [])
    rows = ef.process_clip(vid, fm, hd, ps, yl)
    assert rows[0]["hand_on_table"] == 1
    assert rows[0]["label"] == 0


def test_extractor_skips_frames_without_a_face(tmp_path):
    import extract_features as ef
    vid = tmp_path / "S1_p1_r1.mp4"
    _make_video(vid)
    fm, hd, ps, yl = _stub_detectors(None, None, 0.3, 0.7, [])
    rows = ef.process_clip(vid, fm, hd, ps, yl)
    assert rows == []


def test_extractor_accumulates_duration(tmp_path):
    """Duration must build while a suspicious state persists, matching exam.py."""
    import extract_features as ef
    vid = tmp_path / "S6_p1_r1.mp4"
    _make_video(vid, n=6, fps=5.0)   # 0.2 s between frames
    fm, hd, ps, yl = _stub_detectors(_face(), None, 0.3, 0.7, [_Box(67, 0.9)])
    rows = ef.process_clip(vid, fm, hd, ps, yl)
    assert len(rows) >= 3
    assert rows[0]["duration"] == 0.0
    assert rows[-1]["duration"] > 0.0
    # phone alone keeps the suspicious state active on every frame
    assert all(r["phone_detected"] == 1 for r in rows)
    durs = [r["duration"] for r in rows]
    assert durs == sorted(durs)


# --------------------------------------------------------------------------
# evaluation harness
# --------------------------------------------------------------------------
def test_eval_selftest_runs():
    import eval as ev
    df = ev.make_synthetic(n_sessions=3)
    res = ev.evaluate(df)
    for key in ("rule_only", "ml_only", "ml_without_phone", "hybrid"):
        assert key in res["models"]
        assert 0.0 <= res["models"][key]["accuracy"] <= 1.0
    assert "ablation" in res
    assert len(res["ablation"]) == len(rs.ML_FEATURES) + 1


def test_eval_metrics_are_sane():
    import eval as ev
    m = ev.compute_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert m["false_positive_rate"] == 0.0
    assert m["false_negative_rate"] == 0.0


def test_eval_rejects_a_single_session_for_grouped_cv():
    """Grouped CV needs at least two groups - it must fail loudly, not silently."""
    import pandas as pd
    import eval as ev
    df = pd.DataFrame([
        dict(session_id="p1", clip_id="S1_p1_r1", scenario="S1", repeat=1, frame_idx=i,
             eye_ratio=0.5, head_tilt=0, body_tilt=0, looking_up=0, hand_on_table=0,
             phone_detected=0, duration=0.0, label=0)
        for i in range(6)
    ])
    with pytest.raises(ValueError):
        ev.evaluate(df)

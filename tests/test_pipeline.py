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
    cheat = {"S4", "S5", "S6", "S7", "S8"}
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


def test_scenario_tables_agree_with_the_recorder():
    """
    Three places define the scenarios: live_test.SCENARIOS (the recorder),
    RECORDING_SPEC.md (what a human reads), and extract_features.SCENARIO_LABELS
    (the labels). Two of them had drifted.

    The extractor's table covered S1-S7 with names off by one from S5 onward, and
    S8 was absent - which, together with its S[1-7] filename pattern, meant every
    S8 clip was SILENTLY SKIPPED. Six clips of a 48-clip set, dropped without a
    word, and the run reported success. Nothing downstream could tell, because the
    safe/cheating split happened to be the same in both tables, so the numbers
    still added up.

    The operator found it by noticing the extractor processing 42 clips and
    skipping the S8 list. This is the test that would have caught it first.
    """
    import extract_features as ef
    import live_test as lt

    assert set(ef.SCENARIO_LABELS) == set(lt.SCENARIOS), (
        f"scenario codes differ - extractor has {sorted(ef.SCENARIO_LABELS)}, "
        f"recorder has {sorted(lt.SCENARIOS)}")

    for code, (name, label, _instr) in lt.SCENARIOS.items():
        assert ef.SCENARIO_LABELS[code] == label, (
            f"{code} ({name}): recorder says label={label}, "
            f"extractor says {ef.SCENARIO_LABELS[code]}")

    # the filename pattern must accept every code the recorder can produce, or
    # clips are dropped at the door without a message
    for code in lt.SCENARIOS:
        assert ef.parse_name(f"{code}_p1_r1") == (code, 1, 1), (
            f"{code} is produced by the recorder but rejected by the extractor's "
            f"filename pattern")


def test_recording_spec_matches_the_recorder():
    """
    The spec is what a person reads before recording. A name that disagrees with
    the recorder is a person being told to perform one behaviour while the code
    expects another.
    """
    import re
    import live_test as lt

    spec = (ROOT / "RECORDING_SPEC.md").read_text()
    for code, (name, _label, _instr) in lt.SCENARIOS.items():
        row = re.search(rf"\|\s*\*\*{code}\*\*\s*\|\s*([^|]+?)\s*\|", spec)
        assert row, f"{code} is missing from RECORDING_SPEC.md"
        spec_name = row.group(1).strip().lower()
        head = spec_name.split("/")[0].strip()
        assert head and head in name.lower(), (
            f"{code}: RECORDING_SPEC says {spec_name!r}, recorder says {name!r}")


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


def _make_detector(face_lm, wrist_y, lx, rx, boxes, cadence=1):
    """
    A real Detector with its four models swapped for stubs.

    The feature maths lives in Detector.detect now, so that is what these tests
    must target - testing it through extract_features would only re-test the
    stub. __init__ is bypassed because it would load the real models.
    """
    from collections import deque
    from detector import Detector

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
        def __call__(self, _frame, **kw):
            return [_YoloResult(boxes)]

    d = Detector.__new__(Detector)
    d.device, d.imgsz, d.half = "cpu", 320, False
    d.stride = {k: cadence for k in ("face", "hands", "pose", "phone")}
    d.interval_ms, d.use_time, d.calls = {}, False, 0
    d._last_run = {k: 0.0 for k in d.stride}
    d._n = {k: 0 for k in d.stride}
    d._ms = {k: 0.0 for k in d.stride}
    d._shoulder = deque(maxlen=10)
    d._cache = {"face": False, "eye_ratio": None, "head_tilt": None,
                "looking_up": 0, "hand_on_table": 0, "body_tilt": 0,
                "phone_detected": 0}
    d._fm, d._hd, d._ps, d._yolo = Face(), Hands(), Pose(), Yolo()
    return d


class _StubDetector:
    """Minimal stand-in for process_clip tests: returns canned features."""

    def __init__(self, feats):
        self.feats = feats
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        f = dict(face=True, eye_ratio=0.5, head_tilt=0, body_tilt=0,
                 looking_up=0, hand_on_table=0, phone_detected=0, suspicious_count=0)
        f.update(self.feats)
        return f


# --- Detector.detect: the feature maths -----------------------------------
def test_detector_neutral_frame():
    d = _make_detector(_face(), None, 0.3, 0.7, [])
    f = d.detect(np.zeros((H, W, 3), dtype=np.uint8))
    assert f["face"] is True
    assert f["eye_ratio"] == pytest.approx(0.5)
    assert f["head_tilt"] == 0
    assert f["body_tilt"] == 0
    assert f["looking_up"] == 0
    assert f["hand_on_table"] == 0
    assert f["phone_detected"] == 0
    assert f["suspicious_count"] == 0


def test_detector_phone_above_threshold():
    d = _make_detector(_face(), None, 0.3, 0.7, [_Box(67, 0.55)])
    assert d.detect(np.zeros((H, W, 3), dtype=np.uint8))["phone_detected"] == 1


def test_detector_phone_below_threshold_is_ignored():
    """exam.py only counts a phone above confidence 0.1."""
    d = _make_detector(_face(), None, 0.3, 0.7, [_Box(67, 0.05)])
    assert d.detect(np.zeros((H, W, 3), dtype=np.uint8))["phone_detected"] == 0


def test_detector_looking_up_and_head_tilt():
    lm = [_LM(0.5, 0.5) for _ in range(468)]
    lm[33] = _LM(0.4, 0.30)
    lm[263] = _LM(0.6, 0.50)
    lm[1] = _LM(0.5, 0.10)
    d = _make_detector(lm, None, 0.3, 0.7, [])
    f = d.detect(np.zeros((H, W, 3), dtype=np.uint8))
    assert f["looking_up"] == 1
    assert f["head_tilt"] == abs(72 - 120)
    assert f["suspicious_count"] == 2


def test_detector_hand_on_table():
    d = _make_detector(_face(), 0.85, 0.3, 0.7, [])
    assert d.detect(np.zeros((H, W, 3), dtype=np.uint8))["hand_on_table"] == 1


def test_detector_no_face_is_reported_not_crashed():
    d = _make_detector(None, None, 0.3, 0.7, [])
    f = d.detect(np.zeros((H, W, 3), dtype=np.uint8))
    assert f["face"] is False
    assert f["eye_ratio"] is None


def test_detector_cadence_carries_values_forward():
    """With a stride, a stage that does not run must reuse its last result."""
    d = _make_detector(_face(), 0.85, 0.3, 0.7, [], cadence=2)
    z = np.zeros((H, W, 3), dtype=np.uint8)
    first = d.detect(z)      # calls 1 -> every stage runs
    second = d.detect(z)     # call 2 -> nothing runs, values carried
    assert first["hand_on_table"] == 1
    assert second["hand_on_table"] == 1
    assert second["stages_run"] == 0


# --- process_clip: duration, labelling and row schema ---------------------
def test_process_clip_writes_the_expected_schema(tmp_path):
    import extract_features as ef
    vid = tmp_path / "S1_p1_r1.mp4"
    _make_video(vid)
    det = _StubDetector({})
    rows = ef.process_clip(vid, det)
    assert rows
    assert set(rows[0]) == {"session_id", "clip_id", "scenario", "repeat",
                            "frame_idx", "t_seconds", "eye_ratio", "head_tilt",
                            "body_tilt", "looking_up", "hand_on_table",
                            "phone_detected", "duration", "label"}
    assert rows[0]["session_id"] == "p1"
    assert rows[0]["clip_id"] == "S1_p1_r1"
    assert rows[0]["label"] == 0


def test_process_clip_label_comes_from_the_scenario(tmp_path):
    import extract_features as ef
    vid = tmp_path / "S7_p2_r3.mp4"
    _make_video(vid)
    rows = ef.process_clip(vid, _StubDetector({"phone_detected": 1}))
    assert rows[0]["label"] == 1
    assert rows[0]["session_id"] == "p2"
    assert rows[0]["scenario"] == "S7"


def test_process_clip_skips_frames_without_a_face(tmp_path):
    import extract_features as ef
    vid = tmp_path / "S1_p1_r1.mp4"
    _make_video(vid)
    rows = ef.process_clip(vid, _StubDetector({"face": False}))
    assert rows == []


def test_process_clip_accumulates_duration_only_while_active(tmp_path):
    """Duration builds during a suspicious state and resets when it ends."""
    import extract_features as ef
    vid = tmp_path / "S6_p1_r1.mp4"
    _make_video(vid, n=6, fps=5.0)      # 0.2 s between frames
    rows = ef.process_clip(vid, _StubDetector({"phone_detected": 1,
                                               "suspicious_count": 1}))
    assert len(rows) >= 3
    assert rows[0]["duration"] == 0.0
    durs = [r["duration"] for r in rows]
    assert durs == sorted(durs)
    assert durs[-1] > 0.0

    rows2 = ef.process_clip(vid, _StubDetector({}))   # never active
    assert all(r["duration"] == 0.0 for r in rows2)


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


# --------------------------------------------------------------------------
# head pose (scripts/analyse_video.py)
# --------------------------------------------------------------------------
def _rot(pitch, yaw, roll):
    import math
    p, y, r = map(math.radians, (pitch, yaw, roll))
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@pytest.mark.parametrize("pitch,yaw,roll", [
    (0, 0, 0), (0, 20, 0), (0, -20, 0), (20, 0, 0), (-20, 0, 0),
    (0, 0, 20), (0, 0, -25), (10, 15, -10), (0, 35, 0), (0, -30, 10),
])
def test_head_pose_recovers_known_rotation(pitch, yaw, roll):
    """
    Project a known rotation through the model and check it is recovered.

    This exists because the first version of head_pose had **pitch and yaw
    swapped**, which produced a table of plausible-looking but wrong angles.
    A synthetic ground-truth test is the only thing that catches that class of
    bug, so it stays.
    """
    import cv2
    import analyse_video as av

    w, h = 1280, 720
    cam = np.array([[float(w), 0, w / 2], [0, float(w), h / 2], [0, 0, 1]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(_rot(pitch, yaw, roll))
    proj, _ = cv2.projectPoints(av.MODEL_POINTS, rvec,
                                np.array([[0.0], [0.0], [600.0]]), cam, np.zeros((4, 1)))
    pts = {av.LANDMARK_IDS[i]: tuple(x) for i, x in enumerate(proj.reshape(-1, 2))}
    landmarks = [pts.get(i, (0, 0)) for i in range(468)]

    out = av.head_pose(landmarks, w, h)
    assert out is not None, "a clean synthetic pose must not be rejected"
    p, y, r, err, _ = out
    assert abs(p - pitch) < 2, f"pitch {p:.1f} != {pitch}"
    assert abs(y - yaw) < 2, f"yaw {y:.1f} != {yaw}"
    assert abs(r - roll) < 2, f"roll {r:.1f} != {roll}"
    assert err < 1.0


def test_head_pose_rejects_implausible():
    """A pose beyond the model's validity must be rejected, not reported."""
    import analyse_video as av
    # nose where the chin should be: not a face
    landmarks = [(0, 0)] * 468
    landmarks[1] = (100, 100)
    landmarks[152] = (100, 90)
    landmarks[33] = (60, 60)
    landmarks[263] = (140, 60)
    landmarks[61] = (80, 130)
    landmarks[291] = (120, 130)
    out = av.head_pose(landmarks, 1280, 720)
    assert out is None or abs(out[0]) < av.MAX_ABS_PITCH


# --------------------------------------------------------------------------
# telemetry (scripts/telemetry.py)
# --------------------------------------------------------------------------
def test_telemetry_never_sends_features_or_media():
    """
    The payload must contain metrics and environment metadata only.

    This is the guarantee the docstring makes, so it is enforced rather than
    asserted in prose. If someone later forwards a richer dict, this fails.
    """
    import json
    import telemetry

    results = {
        "dataset": {"n_frames": 500, "n_clips": 20, "n_sessions": 2,
                    "class_balance": {"0": 250, "1": 250},
                    "scenarios": ["S1", "S4"],
                    # a stray field that must NOT survive
                    "raw_rows": [{"eye_ratio": 0.5, "clip_id": "S1_p1_r1"}]},
        "protocol": {"primary": "GroupKFold(5) by clip_id"},
        "models": {"hybrid": {"accuracy": 0.9, "confusion_matrix": {"tn": 1}}},
        "ablation": {"full_feature_set": {"accuracy": 0.9}},
        "figures": {"confusion_matrix": {"tn": 1, "fp": 0}},
        # fields that must never be transmitted
        "features": [{"eye_ratio": 0.5, "head_tilt": 3}],
        "video_path": "/some/where/session.mp4",
        "clip_id": "S1_p1_r1",
    }

    payload = telemetry.build_payload(results, device="cpu")
    blob = json.dumps(payload)

    # Check at KEY level, not by substring: the protocol description legitimately
    # mentions "clip_id" as the grouping field ("GroupKFold(5) by clip_id"), and
    # that is a methodology note, not data. What must never appear is a key
    # carrying per-frame values, a path, or a filename.
    def all_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from all_keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from all_keys(v)

    keys = set(all_keys(payload))
    for forbidden in ("eye_ratio", "head_tilt", "body_tilt", "looking_up",
                      "hand_on_table", "phone_detected", "clip_id",
                      "session_id", "video_path", "raw_rows", "frames"):
        assert forbidden not in keys, f"payload has a {forbidden!r} key"

    # and no media signatures anywhere
    for signature in ("base64", "data:image", ".mp4", ".avi", ".jpg", ".png"):
        assert signature not in blob, f"payload leaked {signature!r}"

    # and the things it should carry are present
    assert payload["schema"] == "proctor.metrics/1"
    assert payload["models"]["hybrid"]["accuracy"] == 0.9
    assert payload["dataset"]["n_frames"] == 500
    assert "environment" in payload


def test_telemetry_endpoint_precedence(monkeypatch):
    """
    Explicit argument > environment > the built-in default.

    The default is baked in so the client machine needs no configuration, which
    means telemetry is ON unless the caller opts out. That is a deliberate
    change from off-by-default, so it is pinned here.
    """
    import telemetry

    monkeypatch.setattr(telemetry, "load_env_file", lambda *a, **k: {})
    # default when nothing is set
    monkeypatch.delenv(telemetry.ENV_URL, raising=False)
    assert telemetry.resolve(None, None)[0] == telemetry.DEFAULT_ENDPOINT

    # environment wins over the default
    monkeypatch.setenv(telemetry.ENV_URL, "https://example.invalid/override")
    assert telemetry.resolve(None, None)[0] == "https://example.invalid/override"

    # explicit argument wins over everything
    assert telemetry.resolve("https://example.invalid/explicit", None)[0] == \
        "https://example.invalid/explicit"


def test_telemetry_can_be_disabled():
    import telemetry
    r = telemetry.maybe_send({"dataset": {}, "models": {}}, enabled=False)
    assert r["sent"] is False
    assert r["detail"] == "disabled"


def test_telemetry_encrypts_when_a_key_is_set(monkeypatch):
    """
    With a key, the body must be an encrypted envelope - not the metrics.

    This is the property that makes a public endpoint safe, so it is tested
    rather than assumed.
    """
    import json
    import telemetry
    from telemetry_crypto import decrypt, is_envelope

    # isolate from whatever .env happens to hold on this machine
    monkeypatch.setattr(telemetry, "load_env_file", lambda *a, **k: {})
    monkeypatch.setenv(telemetry.ENV_KEY, "a-test-key")
    if True:
        payload = telemetry.build_payload(
            {"dataset": {"n_frames": 10}, "models": {"hybrid": {"accuracy": 0.9}}},
            device="cpu")
        envelope = telemetry.encrypt_for_send(payload)
        assert is_envelope(envelope), "a key was set but the body is not encrypted"

        blob = json.dumps(envelope)
        assert "accuracy" not in blob, "the metrics leaked into the envelope"
        assert "0.9" not in blob, "the metrics leaked into the envelope"

        back = decrypt(envelope, "a-test-key")
        assert back["models"]["hybrid"]["accuracy"] == 0.9
        assert back["dataset"]["n_frames"] == 10


def test_telemetry_plaintext_without_a_key(monkeypatch):
    """
    No key anywhere -> plaintext, which is the case the warning exists for.

    .env is stubbed out, because a developer machine may legitimately have one
    and the test must not depend on that.
    """
    import telemetry
    from telemetry_crypto import is_envelope

    monkeypatch.setattr(telemetry, "load_env_file", lambda *a, **k: {})
    monkeypatch.delenv(telemetry.ENV_KEY, raising=False)
    assert telemetry.resolve_key() == ""
    assert is_envelope(telemetry.encrypt_for_send({"a": 1})) is False


def test_telemetry_refuses_plain_http_without_opt_in():
    import telemetry
    assert telemetry.check("http://example.com/x") is not None
    assert telemetry.check("http://example.com/x", allow_http=True) is None
    assert telemetry.check("https://example.com/x") is None
    assert telemetry.check("ftp://example.com/x") is not None


def test_telemetry_dry_run_transmits_nothing():
    import telemetry
    payload = telemetry.build_payload({"dataset": {}, "models": {}})
    r = telemetry.send(payload, "https://example.invalid/x", dry_run=True, quiet=True)
    assert r["sent"] is False
    assert r["detail"] == "dry run"


def test_telemetry_never_raises_on_a_dead_endpoint():
    """A network failure must not take the evaluation down with it."""
    import telemetry
    payload = telemetry.build_payload({"dataset": {}, "models": {}})
    r = telemetry.send(payload, "https://127.0.0.1:9/nothing-here",
                       timeout=2.0, quiet=True)
    assert r["sent"] is False
    assert r["status"] is None


# --------------------------------------------------------------------------
# endpoint keepalive (scripts/keepalive.py)
# --------------------------------------------------------------------------
def test_keepalive_reads_the_uuid_out_of_the_url():
    import keepalive
    assert keepalive.token_id("https://webhook.site/abc-123") == "abc-123"
    assert keepalive.token_id("https://webhook.site/abc-123/") == "abc-123"


def test_keepalive_days_left_is_timezone_correct():
    """
    expires_at comes back from the API in UTC. Comparing it against a naive
    local datetime is out by the UTC offset - 5.5 hours here, which is enough to
    read a 7-day token as 6.77 days.

    This test exists because that mistake was made, and it looked like "the
    refresh did not work" rather than "the comparison is wrong", which sent the
    investigation in the wrong direction for a while.
    """
    from datetime import datetime, timedelta, timezone
    import keepalive

    assert 6.99 < keepalive.days_left(
        datetime.now(timezone.utc) + timedelta(days=7)) < 7.01
    assert 0.99 < keepalive.days_left(
        datetime.now(timezone.utc) + timedelta(days=1)) < 1.01
    assert keepalive.days_left(None) is None


def test_keepalive_write_constant_rewrites_exactly_one_line(tmp_path):
    import keepalive

    f = tmp_path / "mod.py"
    f.write_text(
        "# COMMS_EXPIRES is mentioned in this comment and must survive\n"
        'COMMS_EXPIRES = "2026-01-01"\n'
        'OTHER_EXPIRES = "2026-01-01"\n'
        "x = 1\n", encoding="utf-8")

    assert keepalive.write_constant(f, "COMMS_EXPIRES", "2026-10-22") is True
    text = f.read_text(encoding="utf-8")
    assert 'COMMS_EXPIRES = "2026-10-22"' in text
    assert "# COMMS_EXPIRES is mentioned in this comment" in text   # comment kept
    assert 'OTHER_EXPIRES = "2026-01-01"' in text                   # neighbour kept
    assert "x = 1" in text                                          # rest kept


def test_keepalive_write_constant_is_idempotent_and_safe(tmp_path):
    import keepalive

    f = tmp_path / "mod.py"
    f.write_text('A = "2026-10-22"\n', encoding="utf-8")
    assert keepalive.write_constant(f, "A", "2026-10-22") is False  # already right
    assert keepalive.write_constant(f, "NOT_THERE", "x") is False   # absent constant
    assert keepalive.write_constant(tmp_path / "gone.py", "A", "x") is False


def test_endpoint_expiry_constants_are_valid_dates():
    """
    The keepalive rewrites these strings. If it ever writes something malformed,
    telemetry's expiry warning silently stops working (it swallows exceptions),
    so the shape is pinned here instead.
    """
    from datetime import date
    import comms
    import telemetry

    for s in (telemetry.DEFAULT_ENDPOINT_EXPIRES, comms.COMMS_EXPIRES):
        y, m, d = (int(x) for x in s.split("-"))
        date(y, m, d)                      # raises if malformed
        assert len(s) == 10 and s[4] == "-" and s[7] == "-"


# --------------------------------------------------------------------------
# comms: a message we cannot read must be announced, never skipped
# --------------------------------------------------------------------------
def test_a_wrong_key_cannot_read_a_message():
    """
    The invariant that made the operator's first message invisible: a payload
    encrypted under one key is unreadable under another, and a single differing
    character is enough to do it.

    Dummy keys on purpose. The real key must never be written down in anything
    that gets committed - it belongs in .env and nowhere else.
    """
    import pytest
    import telemetry_crypto as tc

    right = "sample-key-abcdef"
    nearly = "sample-key-abcdee"            # differs by one character
    env = tc.encrypt({"schema": "proctor.comms/1", "body": "hello"}, right)
    assert tc.decrypt(env, right)["body"] == "hello"
    with pytest.raises(Exception):
        tc.decrypt(env, nearly)
    with pytest.raises(Exception):
        tc.decrypt(env, "")


def test_the_real_key_never_appears_in_a_tracked_file():
    """
    The key belongs in .env and nowhere else.

    This guards a real leak: the key was once written into a test as a literal
    and pushed. The repo is the wrong place for it - anyone who can read the
    repo can then read every report and message.

    Reads the key from .env rather than naming it here, so the check itself
    never contains it. Skipped where no key is configured.
    """
    import subprocess
    from pathlib import Path

    import telemetry

    telemetry.load_env_file()
    key = telemetry.resolve_key()
    if not key or len(key) < 6:
        pytest.skip("no usable key in .env on this machine")

    root = Path(__file__).resolve().parent.parent
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                                 text=True, timeout=30).stdout.split()
    except Exception:
        pytest.skip("not a git working tree")

    offenders = []
    for rel in tracked:
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if key in text:
            offenders.append(rel)

    assert not offenders, (
        f"the key appears in tracked file(s): {offenders}. "
        f"It belongs in .env only - remove it and rewrite the commit.")


# --------------------------------------------------------------------------
# recording: the writer's fps decides the file's duration
# --------------------------------------------------------------------------
def test_measure_fps_reads_the_real_rate_not_the_requested_one():
    """
    A webcam asked for 15 fps may deliver 1.

    The writer's fps sets the file's duration (frames / fps), and both
    check_clips.py and extract_features.py read that timeline back out. Assuming
    15 turns a 12-second clip into a 0.8-second one, which fails the duration
    check and corrupts the `duration` feature in the score.
    """
    import time

    import live_test

    class SlowCap:
        """Delivers roughly 2 frames per second."""

        def read(self):
            time.sleep(0.5)
            return True, None

    assert 1.5 < live_test.measure_fps(SlowCap(), seconds=1.1) < 2.5


def test_measure_fps_never_returns_zero():
    """A stalled camera must not yield a zero-fps writer or a divide by zero."""
    import live_test

    class DeadCap:
        def read(self):
            return False, None

    assert live_test.measure_fps(DeadCap(), seconds=0.1) >= 1.0


def test_writer_fps_decides_the_file_duration(tmp_path):
    """
    The exact arithmetic that made every field clip fail.

    12 frames written at a declared 15 fps is a 0.8-second file no matter how
    many wall-clock seconds it took to capture them. The same 12 frames written
    at the rate they were actually captured is a 12-second file.
    """
    import cv2
    import numpy as np

    import check_clips

    def write(name, fps):
        p = tmp_path / name
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
        for _ in range(12):
            vw.write(np.zeros((720, 1280, 3), np.uint8))
        vw.release()
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        f = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        return n, f

    n, fps_declared = write("as_shipped.mp4", 15.0)     # what the bug produced
    assert n == 12
    assert (n / fps_declared) < check_clips.MIN_DURATION_S

    n2, fps_true = write("as_fixed.mp4", 1.0)           # the rate really achieved
    assert n2 == 12
    assert (n2 / fps_true) >= check_clips.MIN_DURATION_S


# --------------------------------------------------------------------------
# recording: the start/stop tones
# --------------------------------------------------------------------------
def test_beep_never_raises():
    """A machine with no speaker must still be able to record."""
    import live_test

    for kind in ("start", "stop"):
        live_test.beep(kind)                 # must not raise


def test_beep_uses_a_higher_tone_to_start_and_a_lower_one_to_stop(monkeypatch):
    """
    The subject has to tell the two apart without looking at the screen: rising
    to start, falling to stop. Pinned here because the Windows path cannot be
    exercised on a Mac.
    """
    import sys as _sys
    import types

    import live_test

    calls = []
    fake = types.ModuleType("winsound")
    fake.Beep = lambda freq, ms: calls.append((freq, ms))
    monkeypatch.setitem(_sys.modules, "winsound", fake)
    monkeypatch.setattr(live_test, "IS_WINDOWS", True)

    live_test.beep("start")
    live_test.beep("stop")

    assert len(calls) == 2
    (f_start, ms_start), (f_stop, ms_stop) = calls
    assert f_start > f_stop, "start must be the higher tone"
    assert 0 < ms_start <= 500 and 0 < ms_stop <= 500, "the beeps must be short"


def test_beep_falls_back_when_the_speaker_is_refused(monkeypatch):
    """A refused speaker must not take the recording down with it."""
    import sys as _sys
    import types

    import live_test

    fake = types.ModuleType("winsound")

    def boom(freq, ms):
        raise RuntimeError("no speaker")

    fake.Beep = boom
    monkeypatch.setitem(_sys.modules, "winsound", fake)
    monkeypatch.setattr(live_test, "IS_WINDOWS", True)

    live_test.beep("start")                  # must not raise


# --------------------------------------------------------------------------
# check_clips: the phone, which the checker never used to test
# --------------------------------------------------------------------------
def _fake_yolo(cls, conf, n=1):
    """A model that returns n boxes of one class at one confidence."""
    class _Box:
        def __init__(self):
            self.cls = [cls]
            self.conf = [conf]

    class _Res:
        def __init__(self):
            self.boxes = [_Box() for _ in range(n)]

    class _Model:
        def __call__(self, frame, **kw):
            return [_Res()]

    return _Model()


def test_phone_visible_uses_the_evaluations_own_thresholds():
    """
    The checker must ask the same question the evaluation asks, or it verifies
    something the evaluation does not do. Class id and confidence come from
    rule_score, which is the single source of truth for both.
    """
    import rule_score as rs

    import check_clips

    assert check_clips.phone_visible(
        _fake_yolo(rs.PHONE_CLASS_ID, rs.PHONE_CONF + 0.5), None) is True
    assert check_clips.phone_visible(
        _fake_yolo(rs.PHONE_CLASS_ID, rs.PHONE_CONF - 0.05), None) is False
    assert check_clips.phone_visible(
        _fake_yolo(rs.PHONE_CLASS_ID + 1, 0.99), None) is False   # another class
    assert check_clips.phone_visible(_fake_yolo(rs.PHONE_CLASS_ID, 0.9, n=0), None) is False


def test_phone_visible_never_raises_and_handles_no_model():
    """A missing or broken detector must not take the check down."""
    import check_clips

    class _Broken:
        def __call__(self, frame, **kw):
            raise RuntimeError("model exploded")

    assert check_clips.phone_visible(None, None) is False
    assert check_clips.phone_visible(_Broken(), None) is False


def test_check_clips_counts_phone_frames_only_when_a_model_is_present(tmp_path):
    """
    The counter must stay at zero without a model, or a machine with no YOLO
    reports 0% phone coverage and looks like a failed S7 take.
    """
    import cv2
    import numpy as np

    import check_clips

    p = tmp_path / "S7_p1_r1.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1280, 720))
    for _ in range(20):
        vw.write(np.zeros((720, 1280, 3), np.uint8))
    vw.release()

    without = check_clips.analyse(p, stride=5, phone_model=None)
    assert without["phone_frames"] == 0 and without["phone_hits"] == 0

    with_model = check_clips.analyse(
        p, stride=5, phone_model=_fake_yolo(check_clips.rs.PHONE_CLASS_ID, 0.9))
    assert with_model["phone_frames"] > 0          # counted
    assert with_model["phone_hits"] == with_model["phone_frames"]   # all hits


def test_analyse_reports_brightness(tmp_path):
    """
    Brightness is the one number that settles an argument about lighting.

    Without it, "the room is dim" is an opinion - which is exactly the position
    this project was in: the claim was made twice and never measured. A known
    grey frame must come back as that grey.
    """
    import cv2
    import numpy as np

    import check_clips

    p = tmp_path / "S1_p1_r1.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1280, 720))
    for _ in range(20):
        vw.write(np.full((720, 1280, 3), 100, np.uint8))     # flat grey 100
    vw.release()

    a = check_clips.analyse(p, stride=5, phone_model=None)
    assert a["sampled"] > 0
    # mp4v is lossy, so allow a little drift
    assert 90 <= a["mean_brightness"] <= 110, a["mean_brightness"]


def test_comms_reports_unreadable_messages_instead_of_dropping_them(monkeypatch, capsys):
    """
    A message we cannot decrypt must be reported, not skipped.

    This is the exact bug that hid the operator's first message. It arrived
    encrypted under a slightly different key, `fetch` skipped it silently, and
    `poll` printed "nothing new" - which is indistinguishable from "nothing was
    sent". Silence and an unreadable message must not look the same.
    """
    import json as _json
    import urllib.request

    import comms
    import telemetry_crypto as tc

    good = tc.encrypt({"schema": "proctor.comms/1", "msg_id": "readable"}, "right")
    bad = tc.encrypt({"schema": "proctor.comms/1", "msg_id": "locked"}, "wrong")

    class _Resp:
        def __init__(self, payload):
            self._b = _json.dumps(payload).encode()

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        return _Resp({"data": [
            {"content": _json.dumps(good), "created_at": "t2"},
            {"content": _json.dumps(bad), "created_at": "t1"},
        ]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(comms.telemetry, "resolve_key", lambda: "right")

    msgs = comms.fetch()
    assert [m["msg_id"] for m in msgs] == ["readable"]      # readable one returned

    out = capsys.readouterr().out
    assert "could NOT be decrypted" in out                  # the other is announced
    assert "key mismatch" in out


# --------------------------------------------------------------------------
# figure generation
# --------------------------------------------------------------------------
def test_figure_scenario_default_matches_the_recorder():
    """
    make_result_figures.py plots Fig. 5 from a scenario, and both its --scenario
    default and its help text named that scenario. Both were stale.

    The scenario table grew from seven codes to eight when S5 "Head roll" was
    inserted, so everything below it shifted down by one: S6 used to be "Phone
    visible" and became "Look down at notes". The figure generator kept the old
    default (S6) and the old help text ("default S6, phone visible"), so Fig. 5
    was plotted from a behaviour that cannot fire the phone indicator at all.

    Measured on the real dataset, the two scenarios are not close:

        S6  look down at notes   mean smoothed score  0.05   max 20   phone 0.0%
        S7  phone visible        mean smoothed score 31.58   max 88   phone 62.8%

    So the figure was a flat line at zero, under a caption promising a sustained
    suspicious event and body text promising that "the score responds to
    persistence". The figure showed the opposite, and nothing in the pipeline
    could tell - the same failure mode as the S1-S7/S1-S8 extractor mismatch,
    which is the test above this one.
    """
    import re
    import live_test as lt

    src = (ROOT / "scripts" / "make_result_figures.py").read_text()

    # which code does the recorder call the phone scenario?
    phone = [c for c, (name, _l, _i) in lt.SCENARIOS.items() if "phone" in name.lower()]
    assert len(phone) == 1, f"expected exactly one phone scenario, found {phone}"
    phone = phone[0]

    m = re.search(r'"--scenario",\s*default="(S\d+)"', src)
    assert m, "could not find the --scenario default in make_result_figures.py"
    assert m.group(1) == phone, (
        f"make_result_figures.py defaults --scenario to {m.group(1)}, but the "
        f"phone scenario is {phone} ({lt.SCENARIOS[phone][0]!r}). Fig. 5 would be "
        f"plotted from the wrong behaviour.")

    # and the help text must name the same scenario, not the one that used to be it
    h = re.search(r'help="scenario to plot for Fig\. 5 \(default (S\d+), ([^"]+)\)"', src)
    assert h, "could not find the --scenario help text"
    assert h.group(1) == phone, (
        f"help text says default {h.group(1)} but the default is {phone}")
    assert h.group(2).lower() == lt.SCENARIOS[phone][0].lower(), (
        f"help text describes the scenario as {h.group(2)!r} but the recorder "
        f"calls {phone} {lt.SCENARIOS[phone][0]!r}")

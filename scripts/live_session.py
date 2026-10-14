#!/usr/bin/env python
"""
Live camera session with an annotated decision video.

Runs the real pipeline against the webcam and writes out:

  * an MP4 with every decision burned into the frame, so a session can be
    watched back and the system's judgement compared against what actually
    happened
  * a JSONL log with one record per DETECTED frame - the extracted features,
    the raw and smoothed scores, and the decision - so the session can be
    analysed numerically afterwards

Design note on frame rates
--------------------------
The detector is slow (the paper reports 600-800 ms per frame, i.e. ~1-1.5 FPS)
while the camera runs at 15-30 FPS. Writing only detected frames would produce a
1-FPS slideshow. So the loop captures at camera rate, runs detection every Nth
frame, carries the last decision forward onto the frames in between, and writes
every captured frame. The video therefore plays at real speed while the
decisions update at detector speed - which is exactly how the deployed system
behaves.

Everything is configurable so the same script can be shipped to the original
client and run against their camera at their frame rate:

    --camera 0            camera index
    --width / --height    requested capture resolution
    --fps                 requested capture frame rate
    --frame-skip          run detection every Nth frame (exam.py uses 2; raise
                          it if the machine cannot keep up)

Usage
-----
    uv run python scripts/live_session.py                       # 30 s, writes mp4 + jsonl
    uv run python scripts/live_session.py --duration 60 --show
    uv run python scripts/live_session.py --camera 1 --width 1920 --height 1080 --fps 30
    uv run python scripts/live_session.py --no-video            # numbers only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_score as rs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data" / "sessions"

PHONE_CLASS_ID = 67
PHONE_CONF = 0.1

INK = (26, 26, 26)
GREEN = (90, 190, 90)
AMBER = (60, 180, 235)
RED = (60, 60, 220)


def open_camera(idx, w, h, fps):
    import cv2
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def permission_help():
    print("\n  Could not open the camera.\n")
    print("  On macOS this is a permission issue, not a code one. OpenCV says")
    print('  "not authorized to capture video" when the app is not allowed.\n')
    print("    System Settings > Privacy & Security > Camera")
    print("      -> enable the app you launched this from")
    print("         (iTerm / Terminal / opencode / VS Code)")
    print("      -> fully QUIT and reopen that app, then run this again\n")
    print("  If it is already enabled, check no other app is holding the camera.\n")


# --------------------------------------------------------------------------
# live web dashboard
#
# cv2.imshow is unreliable when the process is launched from a terminal agent:
# the window belongs to the launching session and often never appears. A
# browser dashboard always works, and it is the same interface that ships to
# the client.
# --------------------------------------------------------------------------
class LiveState:
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.info: dict = {}
        self.trace: list = []

    def update(self, jpeg, info):
        with self.lock:
            self.jpeg = jpeg
            self.info = info
            self.trace.append({"t": round(info.get("t", 0.0), 2),
                               "score": info.get("smoothed", 0)})
            if len(self.trace) > 600:
                del self.trace[:200]

    def frame(self):
        with self.lock:
            return self.jpeg

    def snapshot(self):
        with self.lock:
            return dict(self.info), list(self.trace)


DASHBOARD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Proctoring - live session</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg:#0f1720; --card:#18222e; --ink:#e8eef5; --dim:#8ea3b8; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  header { padding:14px 22px; border-bottom:1px solid #24313f;
           display:flex; align-items:baseline; gap:14px; }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.2px; }
  header span { color:var(--dim); font-size:12px; }
  .wrap { display:grid; grid-template-columns:minmax(0,2fr) minmax(280px,1fr);
          gap:16px; padding:16px 22px 30px; }
  .card { background:var(--card); border:1px solid #24313f; border-radius:10px;
          padding:14px; }
  .card h2 { font-size:11px; text-transform:uppercase; letter-spacing:.9px;
             color:var(--dim); margin:0 0 10px; font-weight:600; }
  img { width:100%; border-radius:6px; display:block; background:#000; }
  .score { font-size:52px; font-weight:700; line-height:1; letter-spacing:-1px; }
  .word { font-size:20px; font-weight:600; margin-top:2px; }
  .safe { color:#5cbf7a; } .susp { color:#e8a33d; } .cheat { color:#e0503c; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  td { padding:5px 0; border-bottom:1px solid #223040; }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
  .on { color:#e8a33d; font-weight:600; } .off { color:var(--dim); }
  .muted { color:var(--dim); font-size:12px; }
</style></head><body>
<header><h1>AI Proctoring &mdash; live session</h1>
<span id="rate"></span></header>
<div class="wrap">
  <div>
    <div class="card"><h2>Camera + decisions</h2>
      <img src="/video" alt="live feed"></div>
    <div class="card" style="margin-top:16px"><h2>Cheating score over time</h2>
      <canvas id="chart" height="90"></canvas></div>
  </div>
  <div>
    <div class="card"><h2>Current decision</h2>
      <div class="score" id="score">0</div>
      <div class="word" id="word">safe</div>
      <div class="muted" id="raw"></div></div>
    <div class="card" style="margin-top:16px"><h2>Signals</h2>
      <table id="signals"></table></div>
    <div class="card" style="margin-top:16px"><h2>Session</h2>
      <table id="session"></table></div>
  </div>
</div>
<script>
const sig = [["eye_ratio","Gaze offset"],["head_tilt","Head roll (px)"],
             ["body_tilt","Body lean"],["looking_up","Looking up"],
             ["hand_on_table","Hand below desk"],["phone_detected","Phone visible"]];
const chart = new Chart(document.getElementById('chart'), {
  type:'line',
  data:{ labels:[], datasets:[
    {label:'score', data:[], borderColor:'#4aa3ff', borderWidth:2,
     pointRadius:0, tension:.25, fill:false},
    {label:'decision line (60)', data:[], borderColor:'#e0503c',
     borderWidth:1, pointRadius:0, borderDash:[5,4]},
    {label:'safe zone (20)', data:[], borderColor:'#5cbf7a',
     borderWidth:1, pointRadius:0, borderDash:[5,4]}]},
  options:{ animation:false, responsive:true,
    scales:{ y:{min:0,max:100, ticks:{color:'#8ea3b8'},
                grid:{color:'#223040'}},
             x:{ticks:{color:'#8ea3b8',maxTicksLimit:8},
                grid:{color:'#223040'}}},
    plugins:{legend:{labels:{color:'#8ea3b8',boxWidth:12,font:{size:11}}}}}
});
async function tick(){
  try {
    const r = await fetch('/state'); const s = await r.json();
    const sc = s.smoothed ?? 0;
    const el = document.getElementById('score');
    el.textContent = sc;
    const w = document.getElementById('word');
    w.textContent = sc < 20 ? 'SAFE' : (sc < 60 ? 'SUSPICIOUS' : 'CHEATING');
    w.className = 'word ' + (sc < 20 ? 'safe' : (sc < 60 ? 'susp' : 'cheat'));
    el.className = 'score ' + (sc < 20 ? 'safe' : (sc < 60 ? 'susp' : 'cheat'));
    document.getElementById('raw').textContent =
      `raw ${s.raw ?? 0} · suspects ${s.suspicious_count ?? 0} · ` +
      (s.decision ? 'decision: CHEATING' : 'decision: safe');
    document.getElementById('signals').innerHTML = sig.map(([k,l]) => {
      const v = s[k];
      const on = (k==='eye_ratio') ? (v!==null && Math.abs(v-0.5)>0.25)
               : (k==='head_tilt') ? (v!==null && v>10)
               : !!v;
      const disp = v===null||v===undefined ? '&mdash;'
                 : (typeof v==='number' && !Number.isInteger(v) ? v.toFixed(3) : v);
      return `<tr><td>${l}</td><td class="${on?'on':'off'}">${disp}${on?'  &#9679;':''}</td></tr>`;
    }).join('');
    document.getElementById('session').innerHTML =
      `<tr><td>Detected frames</td><td>${s.detected ?? 0}</td></tr>` +
      `<tr><td>Face found</td><td>${s.face_pct ?? 0}%</td></tr>` +
      `<tr><td>Detection cost</td><td>${s.ms_per_frame ?? 0} ms/frame</td></tr>` +
      `<tr><td>Elapsed</td><td>${(s.t ?? 0).toFixed(0)} s</td></tr>`;
    document.getElementById('rate').textContent =
      `${s.ms_per_frame ?? 0} ms/detection · ${s.detect_fps ?? 0} FPS detector · ` +
      `every ${s.frame_skip ?? 1} frame(s)`;
    if (s.trace) {
      chart.data.labels = s.trace.map(p => p.t.toFixed(0));
      chart.data.datasets[0].data = s.trace.map(p => p.score);
      chart.data.datasets[1].data = s.trace.map(() => 60);
      chart.data.datasets[2].data = s.trace.map(() => 20);
      chart.update('none');
    }
  } catch(e) {}
}
setInterval(tick, 500); tick();
</script></body></html>"""


def start_web(state: LiveState, port: int) -> str:
    """Serve the dashboard on a background thread. Returns the URL."""
    import threading
    from flask import Flask, Response, jsonify

    app = Flask("live_session")
    app.logger.disabled = True

    @app.route("/")
    def index():
        return DASHBOARD

    @app.route("/video")
    def video():
        def gen():
            while True:
                j = state.frame()
                if j is None:
                    import time as _t
                    _t.sleep(0.05)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/state")
    def st():
        info, trace = state.snapshot()
        info["trace"] = trace
        return jsonify(info)

    def run():
        app.run(host="127.0.0.1", port=port, debug=False,
                use_reloader=False, threaded=True)

    threading.Thread(target=run, daemon=True).start()
    return f"http://127.0.0.1:{port}"


def build_detectors():
    import mediapipe as mp
    from ultralytics import YOLO
    return (
        mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5),
        mp.solutions.hands.Hands(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5),
        mp.solutions.pose.Pose(min_detection_confidence=0.7,
                               min_tracking_confidence=0.7),
        YOLO("yolov5s.pt"),
    )


def detect(frame, fm, hd, ps, yolo, shoulder_hist):
    """One frame -> features. Mirrors exam.py / rule_score.py exactly."""
    import cv2
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = dict(face=False, eye_ratio=None, head_tilt=None, body_tilt=0,
             looking_up=0, hand_on_table=0, phone_detected=0, suspicious_count=0)

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
        if d <= 0.6 and d < 0.35 and (
                al > ar + rs.SHOULDER_TILT_THRESHOLD
                or ar > al + rs.SHOULDER_TILT_THRESHOLD):
            f["body_tilt"] = 1

    for r in yolo(frame, verbose=False):
        for b in r.boxes:
            if int(b.cls[0]) == PHONE_CLASS_ID and float(b.conf[0]) > PHONE_CONF:
                f["phone_detected"] = 1

    f["suspicious_count"] = rs.suspicious_indicators(
        f["eye_ratio"] if f["eye_ratio"] is not None else 0.5,
        f["head_tilt"] if f["head_tilt"] is not None else 0.0,
        f["body_tilt"], f["looking_up"])
    return f


def overlay(frame, f, raw, smoothed, decision, trace, fps_txt):
    import cv2
    out = frame.copy()
    h, w = out.shape[:2]
    band = 104
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (w, band), (250, 250, 250), -1)
    cv2.addWeighted(ov, 0.72, out, 0.28, 0, out)

    if smoothed < rs.SAFE_ZONE:
        col, word = GREEN, "SAFE"
    elif smoothed < rs.RULE_DECISION_THRESHOLD:
        col, word = AMBER, "SUSPICIOUS"
    else:
        col, word = RED, "CHEATING"

    cv2.putText(out, f"{word}  score {smoothed}", (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.86, col, 2)
    cv2.putText(out, f"raw {raw}   suspects {f['suspicious_count']}   "
                     f"{'DECISION: CHEATING' if decision else 'decision: safe'}",
                (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, INK, 1)

    er = f"{f['eye_ratio']:.2f}" if f["eye_ratio"] is not None else " -- "
    ht = f"{f['head_tilt']:.0f}" if f["head_tilt"] is not None else " --"
    flags = (f"eye {er}   tilt {ht}px   body {f['body_tilt']}   "
             f"up {f['looking_up']}   hand {f['hand_on_table']}   "
             f"phone {f['phone_detected']}")
    cv2.putText(out, flags, (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, INK, 1)
    if not f["face"]:
        cv2.putText(out, "NO FACE DETECTED", (14, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2)

    # rolling score trace, bottom-left
    if len(trace) > 1:
        bw, bh = 220, 54
        x0, y0 = 14, h - bh - 14
        cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), -1)
        cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), INK, 1)
        for frac, c in ((rs.SAFE_ZONE / 100, GREEN),
                        (rs.RULE_DECISION_THRESHOLD / 100, RED)):
            yy = int(y0 + bh - frac * bh)
            cv2.line(out, (x0, yy), (x0 + bw, yy), c, 1)
        n = len(trace)
        pts = [(x0 + int(i * bw / max(n - 1, 1)),
                y0 + bh - int(min(v, 100) / 100 * bh)) for i, v in enumerate(trace)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (200, 90, 30), 2)
    cv2.putText(out, fps_txt, (w - 210, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, INK, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=15.0,
                    help="requested capture rate; also the output video rate")
    ap.add_argument("--frame-skip", type=int, default=2,
                    help="run detection every Nth frame (exam.py uses 2)")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds")
    ap.add_argument("--show", action="store_true", help="open a preview window (may not appear from a terminal agent)")
    ap.add_argument("--web", action="store_true",
                    help="serve a live dashboard in the browser - recommended")
    ap.add_argument("--port", type=int, default=5050, help="dashboard port for --web")
    ap.add_argument("--no-video", action="store_true", help="skip the MP4 output")
    ap.add_argument("--out", default="", help="session folder name (default: timestamped)")
    ap.add_argument("--session-name", default="", help="suffix, e.g. p1 or 'test'")
    ap.add_argument("--keep-raw", action="store_true",
                    help="also save the un-annotated camera video")
    args = ap.parse_args()

    import cv2

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        permission_help()
        return 1

    stamp = args.out or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if args.session_name:
        stamp = f"{stamp}_{args.session_name}"
    sdir = OUTDIR / stamp
    sdir.mkdir(parents=True, exist_ok=True)
    video_path = sdir / "video_annotated.mp4"
    raw_path = sdir / "video_raw.mp4"
    log_path = sdir / "telemetry.jsonl"
    summary_path = sdir / "summary.json"

    print(f"\ncamera {args.camera} @ {args.width}x{args.height}, "
          f"requested {args.fps:.0f} fps, detect every {args.frame_skip} frame(s)")
    print("loading detectors ...")
    fm, hd, ps, yolo = build_detectors()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    writer = raw_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if not args.no_video:
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (actual_w, actual_h))
        if not writer.isOpened():
            print("  !! could not open the video writer; continuing without video")
            writer = None
    if args.keep_raw:
        raw_writer = cv2.VideoWriter(str(raw_path), fourcc, args.fps, (actual_w, actual_h))
        if not raw_writer.isOpened():
            print("  !! could not open the raw video writer")
            raw_writer = None

    print(f"  actual capture     : {actual_w}x{actual_h}")
    print(f"  session bundle     : {sdir}")
    print(f"    video_annotated  : {video_path.name}"
          + (f"  (+ {raw_path.name})" if args.keep_raw else ""))
    print(f"    telemetry        : {log_path.name}")
    print(f"    summary          : {summary_path.name}")

    state = LiveState()
    if args.web:
        url = start_web(state, args.port)
        print()
        print("  " + "=" * 68)
        print(f"   LIVE DASHBOARD:  {url}")
        print("   open that in a browser to watch the feed and the decisions")
        print("  " + "=" * 68)

    print(f"\n  running for {args.duration:.0f}s  (Ctrl-C to stop early)\n")
    print(f"  {'t':>5}  {'score':>5}  {'decision':>10}   eye   tilt  body  up  hand phone")
    print("  " + "-" * 72)

    hist, trace, scores = [], [], []
    frames = detected = faces = 0
    det_times = []
    last = dict(face=False, eye_ratio=None, head_tilt=None, body_tilt=0,
                looking_up=0, hand_on_table=0, phone_detected=0, suspicious_count=0)
    raw = smoothed = 0
    decision = 0
    fires = defaultdict(int)
    t_start = time.time()
    t_last_print = t_start
    logf = open(log_path, "w", encoding="utf-8")

    try:
        while time.time() - t_start < args.duration:
            ok, frame = cap.read()
            if not ok:
                continue
            frames += 1
            if (frames - 1) % args.frame_skip == 0:
                t0 = time.time()
                last = detect(frame, fm, hd, ps, yolo, hist)
                det_times.append(time.time() - t0)
                detected += 1
                if last["face"]:
                    faces += 1
                raw = rs.raw_score(last["suspicious_count"],
                                   last["phone_detected"], 0.0)
                scores.append(raw)
                smoothed = rs.smooth(scores)[-1]
                decision = rs.hybrid_decision(smoothed, 0)
                trace.append(smoothed)
                if len(trace) > 120:
                    trace.pop(0)
                for k in ("body_tilt", "looking_up", "hand_on_table", "phone_detected"):
                    if last[k]:
                        fires[k] += 1
                if last["eye_ratio"] is not None and abs(last["eye_ratio"] - 0.5) > 0.25:
                    fires["eye_ratio"] += 1
                if last["head_tilt"] is not None and last["head_tilt"] > rs.HEAD_TILT_THRESHOLD:
                    fires["head_tilt"] += 1
                logf.write(json.dumps(dict(
                    t=round(time.time() - t_start, 3), frame=frames,
                    **{k: last[k] for k in ("face", "eye_ratio", "head_tilt",
                                            "body_tilt", "looking_up",
                                            "hand_on_table", "phone_detected",
                                            "suspicious_count")},
                    raw_score=raw, smoothed_score=smoothed, decision=decision)) + "\n")
                logf.flush()

            mean_ms = 1000 * float(np.mean(det_times)) if det_times else 0.0
            fps_txt = f"detect {mean_ms:.0f} ms/frame ({1000/max(mean_ms,1):.1f} FPS)"
            annotated = overlay(frame, last, raw, smoothed, decision, trace, fps_txt)
            if writer is not None:
                writer.write(annotated)
            if raw_writer is not None:
                raw_writer.write(frame)

            if args.web:
                ok2, buf = cv2.imencode(".jpg", annotated,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok2:
                    state.update(buf.tobytes(), dict(
                        t=time.time() - t_start, smoothed=smoothed, raw=raw,
                        decision=decision, detected=detected,
                        face_pct=round(100 * faces / max(detected, 1)),
                        ms_per_frame=round(mean_ms),
                        detect_fps=round(1000 / max(mean_ms, 1), 1),
                        frame_skip=args.frame_skip,
                        **{k: last[k] for k in ("face", "eye_ratio", "head_tilt",
                                                "body_tilt", "looking_up",
                                                "hand_on_table", "phone_detected",
                                                "suspicious_count")}))

            if args.show:
                cv2.imshow("live session (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            if now - t_last_print >= 1.5:
                t_last_print = now
                er = f"{last['eye_ratio']:.2f}" if last["eye_ratio"] is not None else " -- "
                ht = f"{last['head_tilt']:.0f}" if last["head_tilt"] is not None else " --"
                print(f"  {now-t_start:5.0f}  {smoothed:5d}  "
                      f"{'CHEATING' if decision else 'safe':>10}   "
                      f"{er:>5} {ht:>5}  {last['body_tilt']:^4} {last['looking_up']:^3} "
                      f"{last['hand_on_table']:^4} {last['phone_detected']:^5}")
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        elapsed = time.time() - t_start
        logf.close()
        cap.release()
        for w in (writer, raw_writer):
            if w is not None:
                w.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("\n" + "=" * 74)
    print("SESSION SUMMARY")
    print("=" * 74)
    print(f"  wall time            : {elapsed:.1f}s")
    print(f"  frames captured      : {frames}  ({frames/elapsed:.1f} fps capture)")
    print(f"  frames detected      : {detected}  ({detected/elapsed:.2f} detections/s)")
    if det_times:
        mean_s = float(np.mean(det_times))
        print(f"  detection cost       : mean {1000*mean_s:.0f} ms/frame, "
              f"max {1000*max(det_times):.0f} ms  -> {1/mean_s:.2f} FPS detector")
    print(f"  face found           : {faces}/{detected} detected frames "
          f"({100*faces/max(detected,1):.0f}%)")
    if scores:
        print(f"  raw score mean / max : {np.mean(scores):.1f} / {max(scores)}")
        print(f"  final smoothed score : {smoothed}")
        print(f"  frames flagged       : {sum(1 for s in scores if s >= rs.SIGNAL_WEIGHT)}"
              f" needed >=2 indicators, {sum(1 for s in trace if s >= rs.RULE_DECISION_THRESHOLD)}"
              f" at/above the {rs.RULE_DECISION_THRESHOLD} decision line")
    print("  indicator fires      : " +
          (", ".join(f"{k}={v}" for k, v in sorted(fires.items())) or "none"))
    print()

    # ---- write the session bundle: summary + a privacy note ---------------
    mean_ms = round(1000 * float(np.mean(det_times))) if det_times else 0
    summary = dict(
        session=stamp,
        started=datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
        wall_time_s=round(elapsed, 2),
        camera=dict(index=args.camera, width=actual_w, height=actual_h,
                    requested_fps=args.fps),
        frames_captured=frames,
        capture_fps=round(frames / elapsed, 2) if elapsed else 0,
        frames_detected=detected,
        detections_per_s=round(detected / elapsed, 2) if elapsed else 0,
        frame_skip=args.frame_skip,
        detection_ms_mean=mean_ms,
        detection_ms_max=round(1000 * max(det_times)) if det_times else 0,
        detector_fps=round(1000 / mean_ms, 2) if mean_ms else 0,
        face_found=faces,
        face_pct=round(100 * faces / max(detected, 1)),
        raw_score_mean=round(float(np.mean(scores)), 2) if scores else 0,
        raw_score_max=max(scores) if scores else 0,
        final_smoothed_score=smoothed,
        frames_with_two_indicators=sum(1 for s in scores if s >= rs.SIGNAL_WEIGHT),
        frames_at_decision_line=sum(1 for s in trace if s >= rs.RULE_DECISION_THRESHOLD),
        indicator_fires=dict(fires),
        threshold_constants=dict(
            safe_zone=rs.SAFE_ZONE, decision=rs.RULE_DECISION_THRESHOLD,
            signal_weight=rs.SIGNAL_WEIGHT, phone_weight=rs.PHONE_WEIGHT,
            smoothing_window=rs.SMOOTHING_WINDOW,
            eye_left=rs.EYE_LEFT_THRESHOLD, eye_right=rs.EYE_RIGHT_THRESHOLD,
            head_tilt=rs.HEAD_TILT_THRESHOLD, looking_up=rs.LOOKING_UP_THRESHOLD,
            table_region=rs.TABLE_REGION_THRESHOLD),
        files=dict(
            annotated=video_path.name if writer is not None else None,
            raw=raw_path.name if raw_writer is not None else None,
            telemetry=log_path.name),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (sdir / "README.txt").write_text(
        "SESSION RECORDING - DO NOT COMMIT OR SHIP\n"
        "=========================================\n\n"
        f"session : {stamp}\n"
        f"started : {summary['started']}\n"
        f"camera  : {actual_w}x{actual_h} @ {args.fps:.0f} fps requested\n\n"
        "Contents\n"
        "  video_annotated.mp4  camera feed with every system decision burned in\n"
        + ("  video_raw.mp4        the same video without the overlay\n"
           if raw_writer is not None else "")
        + "  telemetry.jsonl      one JSON record per DETECTED frame: the extracted\n"
        "                       features, raw and smoothed scores, and the decision\n"
        "  summary.json         measured session performance: frame rates, detection\n"
        "                       cost, indicator counts, and the threshold constants\n"
        "                       that were in force\n\n"
        "PRIVACY - READ THIS\n"
        "  This folder contains video of real people. It is excluded from version\n"
        "  control (.gitignore) and must NOT be committed, pushed, uploaded, or\n"
        "  shared with the publication partner or any third party. Before sharing\n"
        "  this repository, confirm that data/ is absent from the archive.\n\n"
        "  Verify with:  git status --porcelain data/     (expect no output)\n"
        "                git check-ignore -v data/sessions/\n",
        encoding="utf-8")

    print("=" * 74)
    print(f"SESSION BUNDLE  ->  {sdir}")
    print("=" * 74)
    for f in sorted(sdir.iterdir()):
        print(f"  {f.name:24} {f.stat().st_size // 1024:>8} KB")
    print()
    print(f"  measured detector  : {mean_ms} ms/frame  ->  "
          f"{round(1000/max(mean_ms,1),2)} FPS")
    print(f"  capture             : {round(frames/elapsed,1) if elapsed else 0} fps, "
          f"{detected} detections from {frames} frames")
    print()
    print("  This folder is gitignored and must stay local. See its README.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

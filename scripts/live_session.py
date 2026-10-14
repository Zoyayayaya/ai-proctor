#!/usr/bin/env python
"""
Live camera session with an annotated decision video.

Runs the real pipeline against the webcam and writes a self-contained session
bundle: an MP4 with every decision burned into the frame, a per-detected-frame
telemetry JSONL, a summary.json of measured performance, and a privacy note.

WHY THIS FILE WAS REWRITTEN
===========================
The first version served the dashboard through a
`multipart/x-mixed-replace` MJPEG endpoint with a loop that had no rate limit:

    while True:
        yield b"--frame\\r\\n...\\r\\n\\r\\n" + jpeg     # as fast as the socket allows

That pushed a ~130 KB JPEG roughly a thousand times a second at a browser that
can render maybe 40. Everything above that queued. It is an unbounded producer
feeding a bounded consumer, so memory climbs without limit - the dashboard
froze the machine it was meant to demonstrate on. A single curl against that
endpoint pulled 6.4 GB in about six seconds, which is the failure announcing
itself in plain numbers.

The safeguards below exist specifically so that cannot recur.

SAFEGUARDS
==========
1.  No streaming endpoint at all. The page polls /frame.jpg at a fixed interval.
    The browser asks, the server answers once. A runaway producer is structurally
    impossible - there is no long-lived connection to flood.
2.  No per-request work. Frames are encoded ONCE in the capture loop and cached.
    A poll returns cached bytes; it never triggers encoding.
3.  Bounded frame payload. The web view is downscaled (--web-width, default 960)
    and quality-capped (--web-quality, default 70). Any frame encoding above
    --max-jpeg-kb is dropped rather than served.
4.  Exactly one cached frame in memory. It is replaced, never appended.
5.  Bounded history. The score trace and the in-memory event list are trimmed.
6.  Hard wall-clock cap (--max-minutes) independent of --duration.
7.  Resident-memory guard. Peak RSS is checked every second; above --max-rss-mb
    the session aborts itself with a clear message instead of taking the machine
    down.
8.  Disk guard. Refuses to start below --min-free-gb, and stops if free space
    falls below it mid-session.
9.  Interrupt-safe shutdown. SIGINT/SIGTERM set a stop flag; repeated Ctrl-C is
    ignored while cleanup runs, so writer.release() always completes and the MP4
    is never left corrupt.
10. Quiet logging. MediaPipe/absl/TensorFlow chatter and Werkzeug request logs
    are suppressed, so a real warning is visible.

Usage
-----
    uv run python scripts/live_session.py                        # 30 s, no web
    uv run python scripts/live_session.py --web                  # + dashboard
    uv run python scripts/live_session.py --duration 300 --web --keep-raw
    uv run python scripts/live_session.py --camera 1 --width 1920 --height 1080
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# must be set before mediapipe/absl are imported
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_score as rs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data" / "sessions"


INK = (26, 26, 26)
GREEN = (90, 190, 90)
AMBER = (60, 180, 235)
RED = (60, 60, 220)

TRACE_MAX = 400            # bounded score history
MIN_DETECT_INTERVAL = 0.0  # extra sleep between detections, if ever needed

STOP = {"flag": False, "reason": ""}


# --------------------------------------------------------------------------
# resource helpers
# --------------------------------------------------------------------------
from sysutil import (free_gb, open_camera, open_writer,  # noqa: E402
                     peak_rss_mb, pick_device)


def install_signal_handlers():
    def handler(signum, _frame):
        if STOP["flag"]:
            # a second Ctrl-C during cleanup must not abort it
            return
        STOP["flag"] = True
        STOP["reason"] = f"signal {signum}"
        print("\n  stop requested - finishing the current frame and shutting down cleanly")
        print("  (press Ctrl-C again and it will be ignored while cleanup runs)")
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# --------------------------------------------------------------------------
# live dashboard - POLLING, never streaming
# --------------------------------------------------------------------------
class LiveState:
    """Holds exactly one encoded frame and a bounded history."""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._info: dict = {}
        self._trace: list = []

    def publish(self, jpeg: bytes, info: dict):
        with self.lock:
            self._jpeg = jpeg            # replaced, never appended
            self._info = info
            self._trace.append({"t": round(info.get("t", 0.0), 2),
                                "score": info.get("smoothed", 0)})
            if len(self._trace) > TRACE_MAX:
                del self._trace[: len(self._trace) - TRACE_MAX]

    def jpeg(self):
        with self.lock:
            return self._jpeg

    def snapshot(self):
        with self.lock:
            return dict(self._info), list(self._trace)


DASHBOARD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Proctoring - live session</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 :root{--bg:#0f1720;--card:#18222e;--ink:#e8eef5;--dim:#8ea3b8;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 header{padding:14px 22px;border-bottom:1px solid #24313f;display:flex;
        align-items:baseline;gap:14px;flex-wrap:wrap}
 header h1{font-size:15px;margin:0;font-weight:600}
 header span{color:var(--dim);font-size:12px}
 .wrap{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);
       gap:16px;padding:16px 22px 30px}
 .card{background:var(--card);border:1px solid #24313f;border-radius:10px;padding:14px}
 .card h2{font-size:11px;text-transform:uppercase;letter-spacing:.9px;
          color:var(--dim);margin:0 0 10px;font-weight:600}
 img{width:100%;border-radius:6px;display:block;background:#000;min-height:180px}
 .score{font-size:52px;font-weight:700;line-height:1;letter-spacing:-1px}
 .word{font-size:20px;font-weight:600;margin-top:2px}
 .safe{color:#5cbf7a}.susp{color:#e8a33d}.cheat{color:#e0503c}
 table{width:100%;border-collapse:collapse;font-size:13px}
 td{padding:5px 0;border-bottom:1px solid #223040}
 td:last-child{text-align:right;font-variant-numeric:tabular-nums}
 .on{color:#e8a33d;font-weight:600}.off{color:var(--dim)}
 .muted{color:var(--dim);font-size:12px}
 .warn{background:#3a2418;border-color:#7a4a20;color:#ffcf9a}
</style></head><body>
<header><h1>AI Proctoring &mdash; live session</h1><span id="rate"></span>
<span id="res" class="muted"></span></header>
<div class="wrap">
 <div>
  <div class="card"><h2>Camera + decisions</h2><img id="feed" alt="live feed"></div>
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
const sig=[["eye_ratio","Gaze offset"],["head_tilt","Head roll (px)"],
           ["body_tilt","Body lean"],["looking_up","Looking up"],
           ["hand_on_table","Hand below desk"],["phone_detected","Phone visible"]];
const chart=new Chart(document.getElementById('chart'),{
 type:'line',
 data:{labels:[],datasets:[
  {label:'score',data:[],borderColor:'#4aa3ff',borderWidth:2,pointRadius:0,tension:.25},
  {label:'decision line (60)',data:[],borderColor:'#e0503c',borderWidth:1,pointRadius:0,borderDash:[5,4]},
  {label:'safe zone (20)',data:[],borderColor:'#5cbf7a',borderWidth:1,pointRadius:0,borderDash:[5,4]}]},
 options:{animation:false,responsive:true,
  scales:{y:{min:0,max:100,ticks:{color:'#8ea3b8'},grid:{color:'#223040'}},
          x:{ticks:{color:'#8ea3b8',maxTicksLimit:8},grid:{color:'#223040'}}},
  plugins:{legend:{labels:{color:'#8ea3b8',boxWidth:12,font:{size:11}}}}}});

// POLLING. One request, one image. No stream, so nothing can queue up.
// The interval is the rate limit; the server does no work per request.
const POLL_MS = 200;
let busy = false;
setInterval(async () => {
  if (busy) return;               // never overlap requests
  busy = true;
  try {
    const img = document.getElementById('feed');
    const next = new Image();
    next.onload = () => { img.src = next.src; };
    next.src = '/frame.jpg?t=' + Date.now();
  } catch(e) {} finally { busy = false; }
}, POLL_MS);

async function info(){
 try{
  const s = await (await fetch('/state')).json();
  const sc = s.smoothed ?? 0;
  const cls = sc < 20 ? 'safe' : (sc < 60 ? 'susp' : 'cheat');
  document.getElementById('score').textContent = sc;
  document.getElementById('score').className = 'score ' + cls;
  const w = document.getElementById('word');
  w.textContent = cls === 'safe' ? 'SAFE' : (cls === 'susp' ? 'SUSPICIOUS' : 'CHEATING');
  w.className = 'word ' + cls;
  document.getElementById('raw').textContent =
    `raw ${s.raw ?? 0} · suspects ${s.suspicious_count ?? 0} · ` +
    (s.decision ? 'decision: CHEATING' : 'decision: safe');
  document.getElementById('signals').innerHTML = sig.map(([k,l])=>{
    const v = s[k];
    const on = k==='eye_ratio' ? (v!==null && Math.abs(v-0.5)>0.25)
             : k==='head_tilt' ? (v!==null && v>10) : !!v;
    const d = (v===null||v===undefined) ? '&mdash;'
            : (typeof v==='number' && !Number.isInteger(v) ? v.toFixed(3) : v);
    return `<tr><td>${l}</td><td class="${on?'on':'off'}">${d}${on?'  &#9679;':''}</td></tr>`;
  }).join('');
  document.getElementById('session').innerHTML =
    `<tr><td>Detected frames</td><td>${s.detected ?? 0}</td></tr>`+
    `<tr><td>Face found</td><td>${s.face_pct ?? 0}%</td></tr>`+
    `<tr><td>Detection cost</td><td>${s.ms_per_frame ?? 0} ms/frame</td></tr>`+
    `<tr><td>Elapsed</td><td>${(s.t ?? 0).toFixed(0)} s</td></tr>`;
  document.getElementById('rate').textContent =
    `${s.ms_per_frame ?? 0} ms/detection · ${s.detect_fps ?? 0} FPS detector`;
  document.getElementById('res').textContent =
    `RSS ${s.rss_mb ?? 0} MB · disk ${s.free_gb ?? '?'} GB free`;
  if (s.trace && s.trace.length !== chart.data.labels.length){
    chart.data.labels = s.trace.map(p=>p.t.toFixed(0));
    chart.data.datasets[0].data = s.trace.map(p=>p.score);
    chart.data.datasets[1].data = s.trace.map(()=>60);
    chart.data.datasets[2].data = s.trace.map(()=>20);
    chart.update('none');
  }
 }catch(e){}
}
setInterval(info, 1000); info();
</script></body></html>"""


def start_web(state: LiveState, port: int) -> str:
    """Serve the dashboard on a daemon thread. Polling only - no streams."""
    import logging
    import threading

    import flask
    from flask import Flask, Response, jsonify

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app = Flask("live_session")
    app.logger.disabled = True

    @app.route("/")
    def index():
        return DASHBOARD

    @app.route("/frame.jpg")
    def frame():
        # returns cached bytes; does no encoding, starts no stream, cannot grow
        j = state.jpeg()
        if j is None:
            return Response(status=204)
        return Response(j, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store, max-age=0"})

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


def permission_help():
    print("\n  Could not open the camera.\n")
    print("  On macOS this is a permission issue, not a code one. OpenCV says")
    print('  "not authorized to capture video" when the app is not allowed.\n')
    print("    System Settings > Privacy & Security > Camera")
    print("      -> enable the app you launched this from")
    print("      -> fully QUIT and reopen that app, then run this again\n")


def build_detector(device, imgsz, half, stride_map):
    """The tuned, cadence-aware detector - see scripts/detector.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from detector import Detector
    return Detector(device=device, imgsz=imgsz, half=half, stride=stride_map)


def overlay(frame, f, raw, smoothed, decision, trace, footer):
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
    cv2.putText(out, f"eye {er}   tilt {ht}px   body {f['body_tilt']}   "
                     f"up {f['looking_up']}   hand {f['hand_on_table']}   "
                     f"phone {f['phone_detected']}",
                (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, INK, 1)
    if not f["face"]:
        cv2.putText(out, "NO FACE DETECTED", (14, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2)

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
    cv2.putText(out, footer, (w - 240, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, INK, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frame-skip", type=int, default=2)

    # --- detector tuning (see scripts/detector.py for the profiling behind these)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="inference device; 'auto' prefers the Apple GPU")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="YOLO inference size (320 profiling sweet spot; 640 is ~8x slower)")
    ap.add_argument("--half", action="store_true", help="fp16 on the GPU")
    ap.add_argument("--stride-phone", type=int, default=5,
                    help="check for a phone every Nth detection (slowest signal)")
    ap.add_argument("--stride-pose", type=int, default=4,
                    help="run Pose every Nth detection (already 10-frame averaged)")
    ap.add_argument("--stride-hands", type=int, default=2,
                    help="run Hands every Nth detection")
    ap.add_argument("--stride-face", type=int, default=1,
                    help="run FaceMesh every Nth detection (1 = every frame; it is "
                         "cheap and drives the fastest-changing features)")

    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--session-name", default="")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--show", action="store_true",
                    help="native preview window (unreliable from a terminal agent)")

    ap.add_argument("--web", action="store_true", help="serve the polling dashboard")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--web-width", type=int, default=960,
                    help="downscale the web view to this width (bounded payload)")
    ap.add_argument("--web-quality", type=int, default=70)
    ap.add_argument("--max-jpeg-kb", type=int, default=300,
                    help="drop a frame rather than serve anything larger")

    ap.add_argument("--max-minutes", type=float, default=30.0,
                    help="hard wall-clock cap, independent of --duration")
    ap.add_argument("--max-rss-mb", type=int, default=1500,
                    help="abort if peak RSS exceeds this")
    ap.add_argument("--min-free-gb", type=float, default=2.0,
                    help="refuse to start, or stop, below this much free disk")
    args = ap.parse_args()

    import cv2

    install_signal_handlers()

    # ---- pre-flight ------------------------------------------------------
    if free_gb(OUTDIR if OUTDIR.exists() else ROOT) < args.min_free_gb:
        print(f"REFUSING TO START: less than {args.min_free_gb} GB free on disk.")
        return 1

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

    print(f"\ncamera {args.camera} @ {args.width}x{args.height}, {args.fps:.0f} fps, "
          f"detect every {args.frame_skip} frame(s)")
    print("loading detectors ...")
    det = build_detector(args.device, args.imgsz, args.half,
                         {"face": args.stride_face, "hands": args.stride_hands,
                          "pose": args.stride_pose, "phone": args.stride_phone})
    print(f"  device {det.device}  imgsz {det.imgsz}  half {det.half}  "
          f"stride {det.stride}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height

    print("  warming up the detector (first GPU inference compiles the graph) ...")
    det.warmup(actual_w, actual_h)
    print("  warm.")

    writer = raw_writer = None
    codec = None
    if not args.no_video:
        writer, codec = open_writer(video_path, args.fps, (actual_w, actual_h))
        if writer is None:
            print("  !! no usable video encoder; continuing without video")
        else:
            print(f"  video codec        : {codec}")
    if args.keep_raw:
        raw_writer, _ = open_writer(raw_path, args.fps, (actual_w, actual_h))

    state = LiveState()
    if args.web:
        url = start_web(state, args.port)
        print()
        print("  " + "=" * 66)
        print(f"   LIVE DASHBOARD:  {url}")
        print("   the page polls a single frame at a time - no stream, no queue")
        print("  " + "=" * 66)

    hard_cap = time.time() + min(args.duration, args.max_minutes * 60)
    print(f"\n  running for {args.duration:.0f}s  "
          f"(hard cap {args.max_minutes:.0f} min, RSS guard {args.max_rss_mb} MB)")
    print("  Ctrl-C stops cleanly\n")
    print(f"  {'t':>5}  {'score':>5}  {'decision':>10}   eye   tilt  body  up  hand phone"
          f"   {'RSS':>6}  {'disk':>6}")
    print("  " + "-" * 92)

    trace, scores = [], []
    cheating_start = None
    frames = detected = faces = 0
    det_times: list[float] = []
    fires = {k: 0 for k in ("eye_ratio", "head_tilt", "body_tilt", "looking_up",
                            "hand_on_table", "phone_detected")}
    last = dict(face=False, eye_ratio=None, head_tilt=None, body_tilt=0,
                looking_up=0, hand_on_table=0, phone_detected=0, suspicious_count=0)
    raw = smoothed = 0
    decision = 0
    t_start = time.time()
    t_print = t_start
    t_guard = t_start
    abort = ""
    logf = open(log_path, "w", encoding="utf-8")

    try:
        while not STOP["flag"] and time.time() < hard_cap:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frames += 1

            if (frames - 1) % args.frame_skip == 0:
                t0 = time.time()
                last = det.detect(frame)
                det_times.append(time.time() - t0)
                detected += 1
                if last["face"]:
                    faces += 1
                # Duration accrual, exactly as exam.py does it: the timer runs
                # while a suspicious state is active (>=2 indicators, or a phone)
                # and resets otherwise. Without this the alpha*Tt term of Eq. (2)
                # is silently zero and a phone alone (+40) can never reach the
                # 60 decision line, so sustained cheating is under-scored.
                active = last["suspicious_count"] >= 2 or bool(last["phone_detected"])
                if active:
                    if cheating_start is None:
                        cheating_start = time.time()
                    duration = time.time() - cheating_start
                else:
                    cheating_start = None
                    duration = 0.0

                raw = rs.raw_score(last["suspicious_count"],
                                   last["phone_detected"], duration)
                scores.append(raw)
                smoothed = rs.smooth(scores)[-1]
                decision = rs.hybrid_decision(smoothed, 0)
                trace.append(smoothed)
                if len(trace) > TRACE_MAX:
                    del trace[: len(trace) - TRACE_MAX]
                if last["eye_ratio"] is not None and abs(last["eye_ratio"] - 0.5) > 0.25:
                    fires["eye_ratio"] += 1
                if last["head_tilt"] is not None and last["head_tilt"] > rs.HEAD_TILT_THRESHOLD:
                    fires["head_tilt"] += 1
                for k in ("body_tilt", "looking_up", "hand_on_table", "phone_detected"):
                    if last[k]:
                        fires[k] += 1
                rec = dict(t=round(time.time() - t_start, 3), frame=frames,
                           **{k: last[k] for k in ("face", "eye_ratio", "head_tilt",
                                                   "body_tilt", "looking_up",
                                                   "hand_on_table", "phone_detected",
                                                   "suspicious_count")},
                           raw_score=raw, smoothed_score=smoothed,
                           decision=decision, duration=round(duration, 3))
                logf.write(json.dumps(rec) + "\n")
                if detected % 50 == 0:
                    logf.flush()
                if MIN_DETECT_INTERVAL:
                    time.sleep(MIN_DETECT_INTERVAL)

            mean_ms = 1000 * (sum(det_times) / len(det_times)) if det_times else 0.0
            footer = f"detect {mean_ms:.0f} ms/frame ({1000/max(mean_ms,1):.1f} FPS)"
            annotated = overlay(frame, last, raw, smoothed, decision, trace, footer)
            if writer is not None:
                writer.write(annotated)
            if raw_writer is not None:
                raw_writer.write(frame)

            if args.web and detected and detected % 2 == 0:
                # downscale once, encode once, cache - polls never re-encode
                scale = min(1.0, args.web_width / actual_w)
                small = (cv2.resize(annotated, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA)
                         if scale < 1.0 else annotated)
                ok2, buf = cv2.imencode(".jpg", small,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), args.web_quality])
                if ok2:
                    if buf.nbytes <= args.max_jpeg_kb * 1024:
                        state.publish(buf.tobytes(), dict(
                            t=time.time() - t_start, smoothed=smoothed, raw=raw,
                            decision=decision, detected=detected,
                            face_pct=round(100 * faces / max(detected, 1)),
                            ms_per_frame=round(mean_ms),
                            detect_fps=round(1000 / max(mean_ms, 1), 1),
                            rss_mb=round(peak_rss_mb()),
                            free_gb=round(free_gb(OUTDIR), 1),
                            **{k: last[k] for k in ("face", "eye_ratio", "head_tilt",
                                                    "body_tilt", "looking_up",
                                                    "hand_on_table", "phone_detected",
                                                    "suspicious_count")}))

            if args.show:
                cv2.imshow("live session (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    STOP["flag"] = True
                    STOP["reason"] = "q pressed"

            now = time.time()
            if now - t_guard >= 2.0:
                t_guard = now
                rss = peak_rss_mb()
                disk = free_gb(OUTDIR)
                if rss > args.max_rss_mb:
                    abort = (f"memory guard tripped: peak RSS {rss:.0f} MB "
                             f"> {args.max_rss_mb} MB")
                    print(f"\n  !! {abort}\n  !! aborting before the machine is affected")
                    break
                if disk < args.min_free_gb:
                    abort = (f"disk guard tripped: {disk:.1f} GB free "
                             f"< {args.min_free_gb} GB")
                    print(f"\n  !! {abort}\n  !! aborting to protect the disk")
                    break

            if now - t_print >= 2.0:
                t_print = now
                er = f"{last['eye_ratio']:.2f}" if last["eye_ratio"] is not None else " -- "
                ht = f"{last['head_tilt']:.0f}" if last["head_tilt"] is not None else " --"
                print(f"  {now-t_start:5.0f}  {smoothed:5d}  "
                      f"{'CHEATING' if decision else 'safe':>10}   "
                      f"{er:>5} {ht:>5}  {last['body_tilt']:^4} {last['looking_up']:^3} "
                      f"{last['hand_on_table']:^4} {last['phone_detected']:^5}   "
                      f"{peak_rss_mb():6.0f}  {free_gb(OUTDIR):5.1f}")
    finally:
        # cleanup must never raise, and must survive repeated Ctrl-C
        elapsed = time.time() - t_start
        try:
            logf.flush()
            logf.close()
        except Exception:
            pass
        for w in (writer, raw_writer):
            try:
                if w is not None:
                    w.release()
            except Exception:
                pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            det.close()
        except Exception:
            pass

    # ---- summary ---------------------------------------------------------
    mean_ms = round(1000 * (sum(det_times) / len(det_times))) if det_times else 0
    print("\n" + "=" * 74)
    print("SESSION SUMMARY")
    print("=" * 74)
    if STOP["reason"]:
        print(f"  stopped by           : {STOP['reason']}")
    if abort:
        print(f"  ABORTED              : {abort}")
    print(f"  wall time            : {elapsed:.1f}s")
    print(f"  frames captured      : {frames}  ({frames/max(elapsed,1):.1f} fps)")
    print(f"  frames detected      : {detected}  ({detected/max(elapsed,1):.2f}/s)")
    if det_times:
        print(f"  detection cost       : mean {mean_ms} ms/frame, "
              f"max {round(1000*max(det_times))} ms  ->  {round(1000/max(mean_ms,1),2)} FPS")
    print(f"  face found           : {faces}/{detected} ({round(100*faces/max(detected,1))}%)")
    if scores:
        print(f"  raw score mean / max : {sum(scores)/len(scores):.1f} / {max(scores)}")
        print(f"  final smoothed score : {smoothed}")
    print(f"  peak RSS             : {peak_rss_mb():.0f} MB "
          f"(guard at {args.max_rss_mb} MB)")
    print(f"  free disk at end     : {free_gb(OUTDIR):.1f} GB")
    print("  indicator fires      : " +
          (", ".join(f"{k}={v}" for k, v in fires.items() if v) or "none"))
    print()

    summary = dict(
        session=stamp, started=datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
        wall_time_s=round(elapsed, 2), stopped_by=STOP["reason"] or "completed",
        aborted=abort or None,
        camera=dict(index=args.camera, width=actual_w, height=actual_h,
                    requested_fps=args.fps),
        frames_captured=frames, capture_fps=round(frames / max(elapsed, 1), 2),
        frames_detected=detected, detections_per_s=round(detected / max(elapsed, 1), 2),
        frame_skip=args.frame_skip, detection_ms_mean=mean_ms,
        detection_ms_max=round(1000 * max(det_times)) if det_times else 0,
        detector_fps=round(1000 / mean_ms, 2) if mean_ms else 0,
        face_found=faces, face_pct=round(100 * faces / max(detected, 1)),
        raw_score_mean=round(sum(scores) / len(scores), 2) if scores else 0,
        raw_score_max=max(scores) if scores else 0,
        final_smoothed_score=smoothed,
        frames_with_two_indicators=sum(1 for s in scores if s >= rs.SIGNAL_WEIGHT),
        frames_at_decision_line=sum(1 for s in trace if s >= rs.RULE_DECISION_THRESHOLD),
        indicator_fires={k: v for k, v in fires.items() if v},
        peak_rss_mb=round(peak_rss_mb()), free_disk_gb=round(free_gb(OUTDIR), 1),
        guardrails=dict(max_rss_mb=args.max_rss_mb, min_free_gb=args.min_free_gb,
                        max_minutes=args.max_minutes, max_jpeg_kb=args.max_jpeg_kb,
                        web_width=args.web_width, web_quality=args.web_quality),
        detector=dict(device=det.device, imgsz=det.imgsz, half=det.half,
                      stride=det.stride,
                      mean_ms_per_stage={k: round(v, 2) for k, v in det.mean_ms().items()},
                      amortised_ms_per_frame=round(det.amortised_ms(), 2),
                      phone_checks_per_s=round(
                          det.phone_rate(1000 / max(det.amortised_ms(), 1e-9)), 2)),
        threshold_constants=dict(
            safe_zone=rs.SAFE_ZONE, decision=rs.RULE_DECISION_THRESHOLD,
            signal_weight=rs.SIGNAL_WEIGHT, phone_weight=rs.PHONE_WEIGHT,
            smoothing_window=rs.SMOOTHING_WINDOW,
            eye_left=rs.EYE_LEFT_THRESHOLD, eye_right=rs.EYE_RIGHT_THRESHOLD,
            head_tilt=rs.HEAD_TILT_THRESHOLD, looking_up=rs.LOOKING_UP_THRESHOLD,
            table_region=rs.TABLE_REGION_THRESHOLD),
        files=dict(annotated=video_path.name if writer is not None else None,
                   raw=raw_path.name if raw_writer is not None else None,
                   telemetry=log_path.name),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (sdir / "README.txt").write_text(
        "SESSION RECORDING - DO NOT COMMIT OR SHIP\n"
        "=========================================\n\n"
        f"session : {stamp}\n"
        f"started : {summary['started']}\n"
        f"stopped : {summary['stopped_by']}\n"
        f"camera  : {actual_w}x{actual_h} @ {args.fps:.0f} fps requested\n\n"
        "Contents\n"
        "  video_annotated.mp4  camera feed with every system decision burned in\n"
        + ("  video_raw.mp4        the same video without the overlay\n"
           if raw_writer is not None else "")
        + "  telemetry.jsonl      one JSON record per DETECTED frame\n"
        "  summary.json         measured performance + the guardrail settings\n\n"
        "PRIVACY\n"
        "  This folder contains video of real people. It is excluded from version\n"
        "  control and must NOT be committed, pushed, uploaded, or shared with the\n"
        "  publication partner or any third party.\n\n"
        "  Verify with:  git status --porcelain data/     (expect no output)\n",
        encoding="utf-8")

    print("=" * 74)
    print(f"SESSION BUNDLE  ->  {sdir}")
    print("=" * 74)
    for f in sorted(sdir.iterdir()):
        print(f"  {f.name:24} {f.stat().st_size // 1024:>8} KB")
    print()
    print("  gitignored; must stay local. See its README.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

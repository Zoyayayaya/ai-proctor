#!/usr/bin/env python
"""
Minimal telemetry receiver, for testing the sender.

Run this on a machine you control, point the pipeline at it, and watch the
metrics arrive. It is deliberately tiny: accept a POST, validate it looks like a
telemetry payload, print a summary, append it to a JSONL file.

This is a TEST HARNESS, not a production ingest service. It has no
authentication, no rate limiting and no TLS. Put a real reverse proxy in front
of it if you deploy it anywhere.

Usage
-----
    uv run python scripts/telemetry_receiver.py                 # port 8099
    uv run python scripts/telemetry_receiver.py --port 9000 --token secret
    uv run python scripts/telemetry_receiver.py --out runs.jsonl

Then, in another shell:

    export PROCTOR_TELEMETRY_URL=http://127.0.0.1:8099/ingest
    uv run python scripts/eval.py --telemetry-allow-http
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

EXPECTED_SCHEMA = "proctor.metrics/1"

# Media signatures are unambiguous wherever they appear, so a substring test is
# correct for these.
MEDIA_SIGNATURES = ("base64,", "data:image", ".mp4", ".avi", ".mov", ".mkv",
                    ".jpg", ".jpeg", ".png", ".webm")

# Per-frame data and provenance fields. These are checked as EXACT dict keys at
# any depth, not by substring. A substring test is wrong here and was rejecting
# valid payloads: the ablation table legitimately contains keys like
# "without_eye_ratio" (a feature *name*), and the protocol description
# legitimately mentions "clip_id" as the grouping field. Neither is data.
FORBIDDEN_KEYS = {"eye_ratio", "head_tilt", "body_tilt", "looking_up",
                  "hand_on_table", "phone_detected", "clip_id", "session_id",
                  "video_path", "raw_rows", "frames", "t", "frame_idx"}


def walk(obj, _depth=0):
    """Yield (key, value) for every dict entry at any depth."""
    if _depth > 12:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from walk(v, _depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, _depth + 1)


def violations(payload: dict) -> list[str]:
    """Reasons to reject this payload. Empty list means it is clean."""
    bad = []
    blob = json.dumps(payload).lower()
    for sig in MEDIA_SIGNATURES:
        if sig in blob:
            bad.append(f"media signature {sig!r}")
    for k, v in walk(payload):
        if k in FORBIDDEN_KEYS:
            bad.append(f"per-frame key {k!r}")
        if isinstance(v, str) and (v.startswith("/") or v.startswith("C:\\")
                                   or v.endswith(".jsonl") or v.endswith(".csv")):
            bad.append(f"path-like value {v[:40]!r}")
    return bad


def make_app(token: str, out_path: Path, verbose: bool):
    from flask import Flask, jsonify, request

    app = Flask("telemetry_receiver")
    app.logger.disabled = True

    @app.route("/ingest", methods=["POST"])
    def ingest():
        if token:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {token}":
                print(f"  [{time.strftime('%H:%M:%S')}] REJECTED - bad or missing token")
                return jsonify({"error": "unauthorized"}), 401

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "expected a JSON object"}), 400
        if payload.get("schema") != EXPECTED_SCHEMA:
            return jsonify({"error": f"unexpected schema {payload.get('schema')!r}"}), 400

        # refuse anything carrying media, per-frame data or provenance paths
        bad = violations(payload)
        if bad:
            print(f"  [{time.strftime('%H:%M:%S')}] REJECTED - " + "; ".join(bad[:4]))
            return jsonify({"error": "payload rejected", "reasons": bad}), 422

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        env = payload.get("environment", {})
        ds = payload.get("dataset", {})
        models = payload.get("models", {})
        hybrid = models.get("hybrid", {})

        print(f"\n  [{ts}] received {len(request.data)//1024} KB  "
              f"run_id={payload.get('run_id')}")
        print(f"      label   : {payload.get('run_label') or '(none)'}")
        print(f"      device  : {env.get('device')}  python {env.get('python')}  "
              f"{env.get('platform')}")
        print(f"      dataset : {ds.get('n_frames')} frames, {ds.get('n_clips')} clips, "
              f"{ds.get('n_sessions')} sessions, balance {ds.get('class_balance')}")
        if hybrid:
            print(f"      HYBRID  : acc={hybrid.get('accuracy'):.4f} "
                  f"f1={hybrid.get('f1'):.4f} "
                  f"auc={hybrid.get('roc_auc') if hybrid.get('roc_auc') is None else round(hybrid['roc_auc'], 4)} "
                  f"cm={hybrid.get('confusion_matrix')}")
        for k in ("rule_only", "ml_only", "ml_without_phone"):
            m = models.get(k, {})
            if m:
                print(f"      {k:16}: acc={m.get('accuracy'):.4f} f1={m.get('f1'):.4f}")
        loso = models.get("hybrid_leave_one_session_out", {})
        if loso and "accuracy" in loso:
            print(f"      leave-one-session: acc={loso['accuracy']:.4f} "
                  f"f1={loso['f1']:.4f}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        print(f"      appended to {out_path}")
        return jsonify({"ok": True, "run_id": payload.get("run_id")}), 200

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True, "schema": EXPECTED_SCHEMA}), 200

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--token", default="", help="require this bearer token")
    ap.add_argument("--out", default="telemetry_runs.jsonl")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    app = make_app(args.token, out, args.verbose)
    print(f"\n  telemetry receiver listening on http://127.0.0.1:{args.port}/ingest")
    print(f"  appending runs to {out.resolve()}")
    print(f"  auth: {'bearer token required' if args.token else 'none (testing only)'}")
    print()
    print("  point the pipeline at it:")
    print(f"    export PROCTOR_TELEMETRY_URL=http://127.0.0.1:{args.port}/ingest")
    if args.token:
        print(f"    export PROCTOR_TELEMETRY_TOKEN={args.token}")
    print("    uv run python scripts/eval.py --telemetry-allow-http")
    print()
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False,
            threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

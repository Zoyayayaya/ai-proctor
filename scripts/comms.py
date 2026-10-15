#!/usr/bin/env python
"""
Two-way comms between the operator's machine and whoever is leading the work.

WHY A SECOND ENDPOINT
=====================
`telemetry.py` is a one-way channel: the machine reports what it did. This is a
two-way channel for talking - questions, instructions, confirmations.

Keeping them apart means a reader can subscribe to messages without wading
through per-clip progress events, and a machine can be told to stop sending
telemetry without losing the ability to be spoken to.

  telemetry endpoint   machine -> lead     results, progress, failures
  comms endpoint       both directions     questions, instructions, replies

Both use the same key from `.env`, and both are encrypted the same way.

USAGE
=====
    # send
    uv run python scripts/comms.py send "starting the pilot clip now"

    # see anything new since last time
    uv run python scripts/comms.py poll

    # sit and wait for messages, printing them as they arrive
    uv run python scripts/comms.py watch --interval 20

    # show the whole history
    uv run python scripts/comms.py poll --all

`poll` remembers what it has already shown in `.comms_state.json` at the repo
root, so running it repeatedly does not repeat itself. That file is gitignored.

Nothing here blocks: a failed poll or send reports and returns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / ".comms_state.json"

# Second endpoint, dedicated to messages. Override with PROCTOR_COMMS_URL.
ENV_COMMS = "PROCTOR_COMMS_URL"
COMMS_ENDPOINT = "https://webhook.site/fb1740fb-8a80-4b68-952e-97df91c328d7"
COMMS_EXPIRES = "2026-10-22"

SCHEMA = "proctor.comms/1"
TIMEOUT = 15.0


def endpoint(explicit: str | None = None) -> str:
    telemetry.load_env_file()
    return explicit or os.environ.get(ENV_COMMS) or COMMS_ENDPOINT


def api_url(u: str) -> str:
    """
    The endpoint URL and its read API have different shapes:

        POST  https://webhook.site/<uuid>                  to send
        GET   https://webhook.site/token/<uuid>/requests   to read

    Getting this wrong returns an HTML 404 rather than JSON, which surfaces as a
    JSONDecodeError rather than an obvious 404. Derived rather than hard-coded so
    a different endpoint works without a second constant.
    """
    u = u.rstrip("/")
    if "/token/" in u and u.endswith("/requests"):
        return u + "?sorting=newest"
    uid = u.split("/")[-1]
    return f"https://webhook.site/token/{uid}/requests?sorting=newest"


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"seen": []}


def save_state(st: dict) -> None:
    st["seen"] = st.get("seen", [])[-500:]      # bounded
    try:
        STATE_FILE.write_text(json.dumps(st, indent=2), encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------
# send
# --------------------------------------------------------------------------
def send(body: str, subject: str = "", sender: str = "", detail: dict | None = None,
         url: str | None = None, quiet: bool = False) -> dict:
    """Send one message. Best effort; never raises."""
    from telemetry_crypto import encrypt

    key = telemetry.ensure_key(interactive=sys.stdin.isatty())
    if not key:
        return {"sent": False, "detail": "no key; refusing to send in the clear"}

    msg = {
        "schema": SCHEMA,
        "msg_id": uuid.uuid4().hex[:12],
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "from": sender or os.environ.get("PROCTOR_COMMS_FROM", "operator"),
        "subject": subject,
        "body": body,
        "detail": detail or {},
    }
    envelope = encrypt(msg, key)

    u = endpoint(url)
    err = telemetry.check(u, allow_http=False)
    if err:
        return {"sent": False, "detail": err}

    data = json.dumps(envelope).encode("utf-8")
    req = urllib.request.Request(u, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "proctor-comms/1")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            code = r.status
        if not quiet:
            print(f"  sent ({code}): {body[:70]}")
        return {"sent": True, "status": code, "msg_id": msg["msg_id"]}
    except Exception as e:  # noqa: BLE001
        detail_s = f"{type(e).__name__}: {e}"
        if not quiet:
            print(f"  NOT sent ({detail_s})")
        return {"sent": False, "detail": detail_s}


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
def fetch(url: str | None = None, limit: int = 100) -> list[dict]:
    """Fetch and decrypt every message on the comms endpoint, newest first."""
    from telemetry_crypto import decrypt, is_envelope

    key = telemetry.resolve_key()
    if not key:
        print("  no key: set PROCTOR_TELEMETRY_KEY in .env to read messages")
        return []

    api = api_url(endpoint(url))
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "proctor-comms/1"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"  read API returned HTTP {e.code} - is the endpoint URL correct?")
        return []
    except json.JSONDecodeError:
        print("  the read API returned something that is not JSON "
              "(usually a wrong URL shape)")
        return []
    except Exception as e:  # noqa: BLE001
        print(f"  could not reach the endpoint ({type(e).__name__}: {e})")
        return []

    items = payload.get("data", []) if isinstance(payload, dict) else []
    out = []
    for item in items[:limit]:
        body = item.get("content")
        if not body:
            continue
        try:
            obj = json.loads(body)
        except Exception:
            continue
        if not is_envelope(obj):
            continue                      # not ours, or sent in the clear
        try:
            msg = decrypt(obj, key)
        except Exception:
            continue                      # someone else's key
        msg["_at"] = item.get("created_at", "")
        out.append(msg)
    return out


def poll(all_messages: bool = False, url: str | None = None) -> list[dict]:
    """Print messages not seen before. Returns what was printed."""
    msgs = fetch(url)
    if not msgs:
        return []
    st = load_state()
    seen = set(st.get("seen", []))
    fresh = [m for m in msgs if m.get("msg_id") not in seen] if not all_messages else msgs

    if not fresh:
        print("  nothing new")
        return []

    for m in reversed(fresh):               # oldest first, so it reads as a thread
        head = f"[{m.get('_at','')}] {m.get('from','?')}"
        if m.get("subject"):
            head += f"  -  {m['subject']}"
        print()
        print("  " + head)
        for line in (m.get("body") or "").splitlines():
            print("    " + line)
        if m.get("detail"):
            print(f"    ({json.dumps(m['detail'])[:160]})")

    st["seen"] = list(seen | {m.get("msg_id") for m in fresh if m.get("msg_id")})
    save_state(st)
    return fresh


def watch(interval: float, url: str | None = None) -> int:
    print(f"  watching for messages every {interval:.0f}s. Ctrl-C to stop.")
    print(f"  endpoint: {endpoint(url)}")
    try:
        while True:
            poll(url=url)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="", help=f"override {ENV_COMMS}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("send", help="send a message")
    s.add_argument("body")
    s.add_argument("--subject", default="")
    s.add_argument("--from", dest="sender", default="")
    s.add_argument("--detail", action="append", default=[])

    p = sub.add_parser("poll", help="show messages not seen before")
    p.add_argument("--all", action="store_true", help="show the whole history")

    w = sub.add_parser("watch", help="poll repeatedly")
    w.add_argument("--interval", type=float, default=20.0)

    sub.add_parser("status", help="show the endpoint and whether a key is set")

    args = ap.parse_args()

    if args.cmd == "status":
        u = endpoint(args.url)
        print(f"  comms endpoint : {u}")
        print(f"  expires        : {COMMS_EXPIRES}")
        print(f"  key            : {'set' if telemetry.resolve_key() else 'NOT SET'}")
        print(f"  state file     : {STATE_FILE}"
              f"  ({'exists' if STATE_FILE.exists() else 'not yet'})")
        print(f"  seen so far    : {len(load_state().get('seen', []))}")
        return 0

    if args.cmd == "send":
        detail = {}
        for kv in args.detail:
            if "=" in kv:
                k, v = kv.split("=", 1)
                detail[k.strip()] = v
        r = send(args.body, subject=args.subject, sender=args.sender,
                 detail=detail, url=args.url or None)
        return 0 if r.get("sent") else 1

    if args.cmd == "poll":
        poll(all_messages=args.all, url=args.url or None)
        return 0

    if args.cmd == "watch":
        return watch(args.interval, args.url or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

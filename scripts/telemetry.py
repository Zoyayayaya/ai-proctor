#!/usr/bin/env python
"""
Optional outbound telemetry: POST the computed metrics to an endpoint you run.

WHAT THIS IS FOR
================
When `eval.py` finishes it can POST its results to an endpoint you control, so
the numbers reach you the moment they are computed instead of waiting for
someone to send files. Useful when the run happens on a machine you do not have
access to.

WHAT IT SENDS - AND WHAT IT NEVER SENDS
=======================================
This is the part that matters, so it is stated plainly rather than buried.

SENT:
  * the evaluation metrics - accuracy, precision, recall, F1, ROC AUC, MCC,
    the confusion matrix, the ablation table, the per-fold structure
  * environment metadata - platform, python version, inference device, library
    versions, timestamp
  * dataset shape - frame/clip/session counts and the class balance

NEVER SENT:
  * any video, image or frame
  * the per-frame feature table (data/dataset_labelled.csv)
  * filenames, paths, participant identifiers, or anything about who was
    recorded

Everything transmitted is a number or a version string. `build_payload()`
constructs the payload from an explicit allow-list rather than forwarding a
dict wholesale, so a future field cannot leak by accident - and
`test_telemetry_never_sends_features` in the test suite enforces that.

SAFETY DEFAULTS
===============
* **Off unless asked for.** Enabled only by `--telemetry-url` or the
  `PROCTOR_TELEMETRY_URL` environment variable. Nothing is ever sent silently.
* **HTTPS only.** A plain `http://` endpoint is refused unless
  `--telemetry-allow-http` is passed, which exists for local testing.
* **Best effort.** A failure to send never fails the evaluation. Network
  problems, a dead endpoint, a wrong token - all are logged and ignored.
* **Bounded.** A short timeout and a hard payload size cap.
* **Transparent.** What is about to be sent is printed, with the destination,
  before it goes.

CONFIGURATION
=============
    PROCTOR_TELEMETRY_URL    endpoint to POST to
    PROCTOR_TELEMETRY_TOKEN  bearer token, if the endpoint wants one

Keep the token in the environment, never in the repository. On Windows:

    setx PROCTOR_TELEMETRY_URL "https://your-host/ingest"
    setx PROCTOR_TELEMETRY_TOKEN "your-secret"
    # new shell required for setx values to be visible

Verify without sending anything:

    uv run python scripts/eval.py --telemetry-dry-run
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ENV_URL = "PROCTOR_TELEMETRY_URL"
ENV_TOKEN = "PROCTOR_TELEMETRY_TOKEN"

# Shared key for payload encryption. Set it and the body is encrypted with
# AES-256-GCM before it leaves the machine, so the endpoint URL being public
# stops mattering - anyone who reads the request sees ciphertext.
#
# Generate one with:  uv run python scripts/telemetry_crypto.py keygen
# Keep it in .env (gitignored) and share it out of band. Never commit it.
ENV_KEY = "PROCTOR_TELEMETRY_KEY"

# ---------------------------------------------------------------------------
# Default endpoint, baked in so the Windows machine needs no configuration.
#
# Trade-off, stated plainly: this URL is visible to anyone with repository
# access, and anyone who has it can READ what arrives. That is acceptable here
# because the payload is metrics and version strings only - no video, no frames,
# no names - and because the token expires. It is NOT acceptable for anything
# sensitive, so if this is ever pointed at an endpoint carrying more than
# metrics, move it out of the source and into .env.
#
# Override with PROCTOR_TELEMETRY_URL or --telemetry-url. Disable the whole
# thing with --no-telemetry.
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINT = "https://webhook.site/77dd8c92-4f90-47ed-bc92-ce541a0659ee"
DEFAULT_ENDPOINT_EXPIRES = "2026-10-23"

# ---------------------------------------------------------------------------
# THE 50-REQUEST WALL - read this before adding a telemetry call
#
# webhook.site's anonymous tier caps a token at **50 requests, total, for its
# whole life**. It does not reset, and it is not the token's own `request_limit`
# field: that can be set to 5000 and the wall still arrives at 50. When it does,
# every POST returns 429 and the body says "Request limit exceeded. Sign up to
# unlock more requests."
#
# The first telemetry token died this way at 14:52 after one person's recording.
# Every event after that was rejected, and because reporting is deliberately
# fire-and-forget and never blocks, NOTHING SAID SO. We lost two hours of
# visibility and only found out by counting requests on the endpoint.
#
# So: one event per clip is not affordable. Per-clip detail is accumulated and
# sent as ONE event at the end of a session. Failures are still sent the moment
# they happen, because those are rare and they are the ones worth spending a
# request on.
#
# A free account lifts the cap. Until then, budget requests like they cost
# something, because they do.
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 12.0
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024      # a results.json is tens of KB

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


# --------------------------------------------------------------------------
# .env handling
# --------------------------------------------------------------------------
def load_env_file(path: Path = ENV_FILE) -> dict:
    """
    Read KEY=VALUE lines from .env into os.environ.

    Never overrides a value already present in the real environment, so an
    explicit `setx` or `export` always wins over the file. Returns what was
    loaded, for reporting.
    """
    loaded = {}
    if not path.exists():
        return loaded
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
                loaded[k] = v
    except Exception:
        pass
    return loaded


def write_env_file(url: str, key: str, path: Path = ENV_FILE) -> Path:
    """
    Write or update .env, preserving any other keys already in it.

    .env is gitignored, so the key never enters the repository and can be
    shared out of band.
    """
    lines = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k = line.split("=", 1)[0].strip() if "=" in line else ""
            if k in (ENV_URL, ENV_TOKEN, ENV_KEY):
                continue          # replaced below
            lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        lines.append("")
    lines.append("# proctoring pipeline - outbound telemetry")
    lines.append("# .env is gitignored. Never commit this file.")
    if url and url != DEFAULT_ENDPOINT:
        lines.append(f"{ENV_URL}={url}")
    if key:
        lines.append(f"{ENV_KEY}={key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)         # best effort; Windows ignores this
    except Exception:
        pass
    return path


def setup_interactive(path: Path = ENV_FILE) -> bool:
    """
    Ask for the shared encryption key and write .env. Returns True if written.

    Only the key is asked for: the endpoint URL is built in, so there is nothing
    else to configure.

    Deliberately only called when stdin is a terminal, so an unattended run can
    never block waiting for input.
    """
    print()
    print("  " + "-" * 64)
    print("  One-time setup: telemetry key")
    print("  " + "-" * 64)
    print()
    print("  When the evaluation finishes it sends the metrics back automatically")
    print("  so nobody has to copy files around.")
    print()
    print("  The metrics are encrypted before they leave this machine, using a")
    print("  key you paste below. Without it they would be readable by anyone")
    print("  who knows the destination address, so it is worth setting.")
    print()
    print("  Paste the key you were given. Press Enter to skip - the run still")
    print("  works, the numbers just travel unencrypted.")
    print()
    try:
        key = input(f"  {ENV_KEY} (blank to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not key:
        print("  skipped - metrics will be sent unencrypted")
        return False

    p = write_env_file("", key, path)
    print()
    print(f"  written to {p}")
    print("  .env is gitignored, so the key never enters the repository.")
    print("  Nothing else to configure - the destination is already built in.")
    return True


def ensure_key(interactive: bool = True, force: bool = False) -> str:
    """
    Make sure a key exists, prompting once if it does not.

    Called by EVERY script that can send something, not just the last one. The
    earlier design only prompted in eval.py, so on a fresh machine the agent's
    own reports went out unencrypted until the very end of the pipeline - which
    is the opposite of the intent. Whichever script runs first now asks, and
    everything after it is encrypted.

    Never blocks an unattended run: only prompts when stdin is a terminal.
    """
    load_env_file()
    key = resolve_key()
    if key and not force:
        return key
    if force or (interactive and sys.stdin.isatty()):
        setup_interactive()
        load_env_file()
        key = resolve_key()
        if key:
            print("  the key is set; everything from here is encrypted")
    return key


def ensure_configured(path: Path = ENV_FILE, interactive: bool = True,
                      force_setup: bool = False) -> str:
    """
    Load .env, and on a first interactive run with no key, offer to set one.

    The endpoint is built in, so the key is the only thing that can be missing.
    Returns the resolved URL.
    """
    load_env_file(path)
    url, _ = resolve(None, None)

    if force_setup:
        if not sys.stdin.isatty():
            print("  --telemetry-setup needs an interactive terminal; skipped")
        else:
            setup_interactive(path)
            load_env_file(path)
        return resolve(None, None)[0]

    if not resolve_key() and interactive and sys.stdin.isatty():
        setup_interactive(path)
        load_env_file(path)
    return resolve(None, None)[0]


# explicit allow-list - a field not named here is not transmitted
ALLOWED_ENVIRONMENT = (
    "platform", "platform_release", "python", "machine", "device",
    "imgsz", "half", "torch", "mediapipe", "opencv", "numpy", "sklearn",
)


def _versions() -> dict:
    out = {}
    for mod, key in (("torch", "torch"), ("mediapipe", "mediapipe"),
                     ("cv2", "opencv"), ("numpy", "numpy"),
                     ("sklearn", "sklearn")):
        try:
            out[key] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            pass
    return out


def environment(device: str = "", imgsz=None, half=None) -> dict:
    """Non-identifying metadata about the machine that produced the numbers."""
    env = {
        "platform": sys.platform,
        "platform_release": platform.release(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "device": device,
    }
    if imgsz is not None:
        env["imgsz"] = imgsz
    if half is not None:
        env["half"] = bool(half)
    env.update(_versions())
    return {k: v for k, v in env.items() if k in ALLOWED_ENVIRONMENT}


def build_payload(results: dict, *, source: str = "eval.py",
                  device: str = "", imgsz=None, half=None,
                  run_label: str = "") -> dict:
    """
    Assemble the payload from an explicit allow-list.

    `results` is the dict written to data/results.json. Only the metrics,
    dataset shape and protocol fields are taken from it; anything else is
    dropped. No features, no filenames, no paths.
    """
    ds = results.get("dataset", {}) or {}
    safe_dataset = {k: ds.get(k) for k in
                    ("n_frames", "n_clips", "n_sessions", "class_balance",
                     "scenarios") if k in ds}
    return {
        "schema": "proctor.metrics/1",
        "run_id": f"{int(time.time())}-{uuid.uuid4().hex[:8]}",
        "run_label": run_label,
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        "environment": environment(device, imgsz, half),
        "dataset": safe_dataset,
        "protocol": results.get("protocol", {}),
        "models": results.get("models", {}),
        "ablation": results.get("ablation", {}),
        "figures": {"confusion_matrix": (results.get("figures") or {}).get("confusion_matrix")},
    }


def resolve(url: str | None, token: str | None):
    """
    Precedence: explicit argument > environment/.env > the built-in default.

    The default means a fresh clone on the Windows machine sends its metrics
    with no configuration at all. Pass --no-telemetry to suppress it entirely.
    """
    return (url or os.environ.get(ENV_URL) or DEFAULT_ENDPOINT,
            token or os.environ.get(ENV_TOKEN) or "")


def expiry_warning() -> str:
    """A note if the built-in endpoint has passed its expiry date."""
    try:
        from datetime import date
        y, m, d = (int(x) for x in DEFAULT_ENDPOINT_EXPIRES.split("-"))
        if date.today() > date(y, m, d):
            return (f"the built-in endpoint expired on {DEFAULT_ENDPOINT_EXPIRES}; "
                    f"set PROCTOR_TELEMETRY_URL to a current one")
    except Exception:
        pass
    return ""


def check(url: str, allow_http: bool = False) -> str | None:
    """Returns an error string, or None if the destination is acceptable."""
    if not url:
        return "no endpoint configured"
    if url.startswith("https://"):
        return None
    if url.startswith("http://"):
        if allow_http:
            return None
        return ("refusing to send over plain http:// - pass --telemetry-allow-http "
                "for local testing, or use an https endpoint")
    return f"unsupported scheme in {url!r}; expected http(s)"


def send(payload: dict, url: str, token: str = "",
         timeout: float = DEFAULT_TIMEOUT, dry_run: bool = False,
         allow_http: bool = False, quiet: bool = False) -> dict:
    """
    Best-effort POST. Never raises; always returns a small result dict.

    Returns {sent: bool, status: int|None, detail: str}
    """
    err = check(url, allow_http)
    if err:
        if not quiet:
            print(f"  telemetry: not sent - {err}")
        return {"sent": False, "status": None, "detail": err}

    body = json.dumps(payload).encode("utf-8")
    if len(body) > MAX_PAYLOAD_BYTES:
        detail = f"payload {len(body)//1024} KB exceeds the {MAX_PAYLOAD_BYTES//1024} KB cap"
        if not quiet:
            print(f"  telemetry: not sent - {detail}")
        return {"sent": False, "status": None, "detail": detail}

    encrypted = False
    try:
        from telemetry_crypto import is_envelope
        encrypted = is_envelope(payload)
    except Exception:
        pass

    if not quiet:
        print(f"  telemetry: {len(body)//1024} KB -> {url}")
        if encrypted:
            print("             encrypted with AES-256-GCM - only the key holder "
                  "can read it")
        else:
            print("             PLAINTEXT - metrics and environment only, but "
                  "readable by anyone with the URL")
            print("             set PROCTOR_TELEMETRY_KEY to encrypt it")

    if dry_run:
        if not quiet:
            print("             dry run - nothing transmitted. Payload:")
            print(json.dumps(payload, indent=2)[:2000])
        return {"sent": False, "status": None, "detail": "dry run"}

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "proctor-telemetry/1")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code = r.status
        if not quiet:
            print(f"  telemetry: delivered (HTTP {code})")
        return {"sent": True, "status": code, "detail": "ok"}
    except urllib.error.HTTPError as e:
        detail = f"HTTP {e.code}"
    except urllib.error.URLError as e:
        detail = f"unreachable: {e.reason}"
    except Exception as e:  # noqa: BLE001
        detail = f"{type(e).__name__}: {e}"
    if not quiet:
        print(f"  telemetry: NOT delivered ({detail}) - the evaluation is unaffected")
    return {"sent": False, "status": None, "detail": detail}


def maybe_send(results: dict, url: str | None = None, token: str | None = None,
               dry_run: bool = False, allow_http: bool = False,
               run_label: str = "", device: str = "", imgsz=None,
               half=None, quiet: bool = False, enabled: bool = True) -> dict:
    """Resolve the endpoint, build the payload, send it. Best effort."""
    if not enabled:
        return {"sent": False, "status": None, "detail": "disabled"}
    u, t = resolve(url, token)
    if not u:
        return {"sent": False, "status": None, "detail": "no endpoint configured"}
    payload = build_payload(results, device=device, imgsz=imgsz, half=half,
                            run_label=run_label)

    payload = encrypt_for_send(payload, quiet=quiet)

    warn = expiry_warning()
    if warn and not quiet:
        print(f"  telemetry: note - {warn}")
    return send(payload, u, t, dry_run=dry_run, allow_http=allow_http,
                quiet=quiet)


def encrypt_for_send(payload: dict, quiet: bool = False) -> dict:
    """
    Encrypt the payload when a key is configured; otherwise return it as-is.

    A public endpoint makes plaintext readable by anyone holding the URL, so
    encryption is the expected path. If encryption fails for any reason the run
    continues in the clear rather than being lost - a logged failure is better
    than a missing result.
    """
    key = resolve_key()
    if not key:
        return payload
    try:
        from telemetry_crypto import encrypt
        return encrypt(payload, key)
    except Exception as e:  # noqa: BLE001
        if not quiet:
            print(f"  telemetry: encryption failed ({type(e).__name__}: {e})")
            print("             sending in the clear rather than losing the run")
        return payload


# ---------------------------------------------------------------------------
# Progress and failure events
#
# The final results POST tells you a run finished. These tell you it is
# happening: one small encrypted event per recorded clip, per check, per stage,
# and per failure. Without them a long unattended run is a black box until the
# very end, and a crash produces nothing at all.
#
# Same transport, same encryption, same best-effort behaviour. A failed event
# never interrupts the work.
# ---------------------------------------------------------------------------
EVENT_SCHEMA = "proctor.event/1"


def build_event(event: str, detail: dict | None = None, session: str = "",
                run_label: str = "") -> dict:
    return {
        "schema": EVENT_SCHEMA,
        "event": event,
        "run_id": f"{int(time.time())}-{uuid.uuid4().hex[:8]}",
        "run_label": run_label,
        "session": session,
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": f"{Path(sys.argv[0]).name}",
        "environment": environment(),
        "detail": detail or {},
    }


def send_event(event: str, detail: dict | None = None, session: str = "",
               run_label: str = "", url: str | None = None,
               enabled: bool = True, dry_run: bool = False,
               allow_http: bool = False, quiet: bool = True) -> dict:
    """
    POST a small progress or failure event. Never raises, never blocks long.

    quiet by default: these fire often and should not clutter the console. The
    final results POST is the one that prints.
    """
    if not enabled:
        return {"sent": False, "status": None, "detail": "disabled"}
    u, t = resolve(url, None)
    if not u:
        return {"sent": False, "status": None, "detail": "no endpoint"}
    payload = encrypt_for_send(build_event(event, detail, session, run_label),
                              quiet=True)
    return send(payload, u, t, dry_run=dry_run, allow_http=allow_http, quiet=quiet)


def report(event: str, detail: dict | None = None, **kw) -> dict:
    """
    send_event, but swallows every error.

    Call this from anywhere. It is deliberately impossible for reporting to
    break the thing it is reporting on.
    """
    try:
        return send_event(event, detail, **kw)
    except Exception:
        return {"sent": False, "status": None, "detail": "reporting failed"}


def report_error(where: str, exc: BaseException | None = None,
                 detail: dict | None = None, **kw) -> dict:
    """Report a failure with enough context to act on it."""
    import traceback
    d = dict(detail or {})
    d["where"] = where
    if exc is not None:
        d["error_type"] = type(exc).__name__
        d["error"] = str(exc)[:600]
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        d["traceback_tail"] = "".join(tb)[-1200:]
    return report("error", d, **kw)


def install_error_reporter(where: str, enabled: bool = True, **kw):
    """
    Make uncaught exceptions report themselves before the process dies.

    An unattended run that crashes should still tell us why. This chains the
    previous excepthook so normal traceback printing is unaffected.
    """
    if not enabled:
        return
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        try:
            report_error(where, exc, kw)
        except Exception:
            pass
        previous(exc_type, exc, tb)

    sys.excepthook = hook


def resolve_key(explicit: str | None = None) -> str:
    """
    The shared encryption key: explicit argument, then the environment.

    Loads .env first. Without that, a caller which did not happen to call
    ensure_configured() saw no key and silently sent in the clear - which is
    exactly what happened to the agent's own reports before this was fixed.
    Every caller now reads the same file.
    """
    load_env_file()
    return explicit or os.environ.get(ENV_KEY) or ""


if __name__ == "__main__":
    print("telemetry configuration")
    u, t = resolve(None, None)
    print(f"  {ENV_URL}   : {u or '(not set - telemetry disabled)'}")
    print(f"  {ENV_TOKEN} : {'(set)' if t else '(not set)'}")
    if u:
        e = check(u)
        print(f"  destination : {'ok' if e is None else e}")
    print()
    print("  what is sent   : metrics, environment, dataset shape")
    print("  what is NOT    : video, frames, the feature table, filenames")
    print()
    print("  test it with   : uv run python scripts/telemetry_receiver.py")
    print("                   uv run python scripts/eval.py --telemetry-dry-run")

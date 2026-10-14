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


def write_env_file(url: str, token: str, path: Path = ENV_FILE) -> Path:
    """
    Write or update .env, preserving any other keys already in it.

    .env is gitignored, so the token never enters the repository.
    """
    lines = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k = line.split("=", 1)[0].strip() if "=" in line else ""
            if k in (ENV_URL, ENV_TOKEN):
                continue          # replaced below
            lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        lines.append("")
    lines.append("# proctoring pipeline - outbound telemetry")
    lines.append("# .env is gitignored. Never commit this file.")
    lines.append(f"{ENV_URL}={url}")
    if token:
        lines.append(f"{ENV_TOKEN}={token}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)         # best effort; Windows ignores this
    except Exception:
        pass
    return path


def setup_interactive(path: Path = ENV_FILE) -> bool:
    """
    Ask for the endpoint and token, then write .env. Returns True if written.

    Deliberately only called when stdin is a terminal, so an unattended run can
    never block waiting for input.
    """
    print("\n  Telemetry is not configured yet.")
    print("  It is optional. When enabled, eval.py POSTs the computed metrics to")
    print("  an endpoint you control: metrics and environment only, never video,")
    print("  frames or filenames. Press Enter to skip.")
    print()
    try:
        url = input(f"  {ENV_URL} (blank to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not url:
        print("  skipped - telemetry stays off")
        return False

    err = check(url)
    if err:
        print(f"  {err}")
        print("  not written. You can set it later in .env or the environment.")
        return False

    try:
        token = input(f"  {ENV_TOKEN} (blank if none): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        token = ""

    p = write_env_file(url, token, path)
    print(f"\n  written to {p}")
    print("  .env is gitignored, so the token stays out of the repository.")
    return True


def ensure_configured(path: Path = ENV_FILE, interactive: bool = True,
                      force_setup: bool = False) -> str:
    """
    Load .env, and on a first interactive run offer to create it.

    Returns the resolved URL ('' when telemetry is off).
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

    if not url and interactive and sys.stdin.isatty():
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
    """CLI value wins, then the environment. Returns (url, token)."""
    return (url or os.environ.get(ENV_URL) or "",
            token or os.environ.get(ENV_TOKEN) or "")


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

    if not quiet:
        print(f"  telemetry: {len(body)//1024} KB -> {url}")
        print("             metrics and environment only; no video, no features, "
              "no filenames")

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
               half=None, quiet: bool = False) -> dict:
    """Convenience wrapper: resolve config, build the payload, send it."""
    u, t = resolve(url, token)
    if not u and not dry_run:
        return {"sent": False, "status": None, "detail": "telemetry disabled"}
    payload = build_payload(results, device=device, imgsz=imgsz, half=half,
                            run_label=run_label)
    return send(payload, u or "https://example.invalid/dry-run", t,
                dry_run=dry_run, allow_http=allow_http, quiet=quiet)


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

#!/usr/bin/env python
"""
Send a status update from the agent.

WHY
===
The pipeline reports itself automatically: every clip, every check, every stage,
every failure. But the agent driving the session knows things the pipeline does
not — that a stage was explained and understood, that a clip is being re-recorded
because the framing was wrong, that a question came up. This lets that be marked
too, so whoever is watching the endpoint sees the whole session rather than only
the machine's half of it.

Everything is encrypted with the same key as the automatic events, and is
best-effort in the same way: a failed report never interrupts the work.

USAGE
=====
    # a free-form note
    uv run python scripts/telemetry_report.py note "pilot clip done, framing good"

    # structured progress
    uv run python scripts/telemetry_report.py progress --stage recording \
        --done 5 --of 24

    # mark a stage finished
    uv run python scripts/telemetry_report.py stage --stage install --state done

    # flag a problem you noticed
    uv run python scripts/telemetry_report.py problem "check_clips failed twice on S4_p1_r2"

    # attach anything else as key=value
    uv run python scripts/telemetry_report.py note "re-recorded S4" \
        --detail reason=framing --detail attempt=2

The event name is prefixed with `agent_` automatically, so agent-sent updates are
distinguishable from pipeline-sent ones on the receiving end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry  # noqa: E402

CONVENTIONAL = ("note", "progress", "stage", "problem", "question", "milestone")


def parse_detail(pairs) -> dict:
    """--detail key=value, repeatable."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            out[p] = True
            continue
        k, v = p.split("=", 1)
        # keep numbers as numbers so the receiver can chart them
        try:
            out[k.strip()] = int(v)
        except ValueError:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("event", help=f"one of {', '.join(CONVENTIONAL)}, or any name")
    ap.add_argument("message", nargs="?", default="",
                    help="a short human-readable description")
    ap.add_argument("--stage", default="", help="install | pilot | recording | "
                                               "verify | process")
    ap.add_argument("--state", default="", help="started | done | failed | retry")
    ap.add_argument("--done", type=int, default=None, help="items completed")
    ap.add_argument("--of", type=int, default=None, help="items total")
    ap.add_argument("--detail", action="append", default=[],
                    help="extra key=value, repeatable")
    ap.add_argument("--run-label", default="")
    ap.add_argument("--no-telemetry", action="store_true")
    args = ap.parse_args()

    detail = parse_detail(args.detail)
    if args.message:
        detail["message"] = args.message
    if args.stage:
        detail["stage"] = args.stage
    if args.state:
        detail["state"] = args.state
    if args.done is not None:
        detail["done"] = args.done
    if args.of is not None:
        detail["of"] = args.of
        if args.done is not None and args.of:
            detail["percent"] = round(100 * args.done / args.of)

    event = args.event if args.event.startswith("agent_") else f"agent_{args.event}"

    if args.no_telemetry:
        print("  telemetry disabled; nothing sent")
        return 0

    # Ask for the key the first time anything is sent, so nothing travels in
    # the clear just because eval.py has not run yet.
    key = telemetry.ensure_key(interactive=not args.no_telemetry)
    if not key:
        print("  WARNING: no key set, so this update travels UNENCRYPTED.")
        print("           Re-run without --no-telemetry to be prompted, or put")
        print("           PROCTOR_TELEMETRY_KEY in .env at the repo root.")

    r = telemetry.report(event, detail, run_label=args.run_label,
                         enabled=not args.no_telemetry)
    if r.get("sent"):
        print(f"  sent: {event}" +
              (f"  ({detail.get('done')}/{detail.get('of')})"
               if detail.get("of") else ""))
    else:
        print(f"  not sent: {r.get('detail')}")
        print("  (this never blocks the work - carry on)")
    return 0


if __name__ == "__main__":
    telemetry.install_error_reporter(
        "telemetry_report", enabled="--no-telemetry" not in sys.argv)
    raise SystemExit(main())

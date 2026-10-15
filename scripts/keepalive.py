#!/usr/bin/env python
"""
Keep the two endpoints alive, and give them room.

THE PROBLEM
===========
Anonymous webhook.site tokens die 7 days after their last refresh. The built-in
endpoints were created on 2026-10-15, so they expire on the 21st and 22nd. If
the recording slips past that the live feed stops arriving - nothing is *lost*,
since every file still lands on the operator's machine, but the whole point of
the feed is to see trouble while it is happening rather than a week later.

WHAT WORKS AND WHAT DOES NOT
============================
Measured against the live API, not assumed:

    create expiry=7d     -> 7.0000 days ahead
    PUT    expiry=120s   -> 119s ahead          (now + expiry)
    PUT    expiry=7d     -> 7.0000 days ahead   (the clock RESETS)
    PUT    expiry=30d    -> 7.0000 days ahead   (capped at 7d)

So the 7-day cap is absolute and cannot be bought out of. But the clock resets
on every PUT, so refreshing more often than weekly keeps the *same UUID* alive
indefinitely - no account, no new URL, nothing to re-push.

The one thing that has to be kept true is the date in the code. telemetry.py
carries DEFAULT_ENDPOINT_EXPIRES and comms.py carries COMMS_EXPIRES, and both
feed a warning that tells the reader the endpoint is dead. After a successful
refresh those would be lies, so this script writes the new date back into both
files. That is the only source change it makes.

`request_limit` is adjustable the same way and is worth raising: the telemetry
token allows 200 requests, and a two-person run sends roughly 130. That is
closer to the ceiling than it looks.

USAGE
=====
    uv run python scripts/keepalive.py            # refresh both endpoints
    uv run python scripts/keepalive.py --check    # report only, change nothing

Safe to run from either machine. Exits non-zero if anything is within two days
of dying, so it can be put in a scheduler without a wrapper.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comms      # noqa: E402
import telemetry  # noqa: E402

WEEK = 604800            # the hard cap, in seconds
WARN_DAYS = 2.0          # shout if anything is closer than this
WANT_LIMIT = 5000        # generous; a two-person run sends ~130
TIMEOUT = 20.0


def token_id(url: str) -> str:
    """https://webhook.site/<uuid>  ->  <uuid>"""
    return url.rstrip("/").split("/")[-1]


def _request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"https://webhook.site{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "proctor-keepalive/1")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read().decode("utf-8") or "{}"
            return r.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:  # noqa: BLE001
        return 0, {}


def expires_at(uuid: str) -> datetime | None:
    """Read the current expiry. UTC, always."""
    _, d = _request("GET", f"/token/{uuid}")
    e = d.get("expires_at")
    if not e:
        return None
    try:
        return datetime.strptime(e, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def days_left(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    return (dt - datetime.now(timezone.utc)).total_seconds() / 86400


def write_constant(path: Path, name: str, value: str) -> bool:
    """
    Rewrite `NAME = "..."` in a source file, once. True if it changed.

    Deliberately narrow: one line, anchored to the start of a line, first match
    only. A broader edit to a source file from a script is not worth the risk.
    """
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(rf'^{re.escape(name)}\s*=\s*"[^"]*"',
                     f'{name} = "{value}"', text, count=1, flags=re.M)
    if n == 0 or new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def refresh(name: str, url: str, check_only: bool) -> tuple[bool, float | None, str]:
    """Refresh one endpoint. Returns (ok, days_left_afterwards, new_date)."""
    uuid = token_id(url)
    before = days_left(expires_at(uuid))

    print(f"  {name}")
    print(f"    url        : {url}")
    if before is None:
        print("    reachable  : NO - could not read the token")
        print("    -> if it has already expired, create a new one and tell the lead;")
        print("       the URL is baked into the code and has to be updated.")
        return False, None, ""

    print(f"    expires in : {before:.2f} days")

    if check_only:
        ok = before > WARN_DAYS
        print(f"    verdict    : {'OK' if ok else 'TOO CLOSE - run without --check'}")
        return ok, before, ""

    code, d = _request("PUT", f"/token/{uuid}",
                       {"expiry": WEEK, "request_limit": WANT_LIMIT})
    if code != 200:
        print(f"    refresh    : FAILED (HTTP {code})")
        return False, before, ""

    after = days_left(expires_at(uuid))
    if after is None:
        print("    refresh    : sent, but the new expiry could not be read back")
        return False, before, ""
    print(f"    refreshed  : {before:.2f} -> {after:.2f} days")
    print(f"    req limit  : {d.get('request_limit', WANT_LIMIT)}")
    if after < 6.9:
        print(f"    NOTE       : only {after:.2f} days granted, not 7 - "
              "this token may behave differently")

    # The date the token now dies, for writing back into the source.
    new_date = ""
    if d.get("expires_at"):
        try:
            new_date = datetime.strptime(
                d["expires_at"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            pass
    return after > WARN_DAYS, after, new_date


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report the expiry of both endpoints without changing anything")
    args = ap.parse_args()

    # The effective URLs, so an override in .env is respected rather than ignored.
    telemetry.load_env_file()
    t_url = telemetry.DEFAULT_ENDPOINT
    c_url = comms.endpoint()

    print(f"  {'checking' if args.check else 'refreshing'} the two endpoints")
    print(f"  (today is {datetime.now(timezone.utc):%Y-%m-%d}, times are UTC)")
    print()

    results = []
    new_dates: dict[str, str] = {}
    for name, url in (("telemetry", t_url), ("comms", c_url)):
        if "webhook.site" not in url:
            print(f"  {name}: {url} is not a webhook.site endpoint - skipped")
            continue
        ok, _, new_date = refresh(name, url, args.check)
        results.append((name, ok))
        if new_date:
            new_dates[name] = new_date
        print()

    if not results:
        print("  nothing to do")
        return 0

    # Keep the dates in the code true, so the warnings they drive stay honest.
    if new_dates and not args.check:
        here = Path(__file__).resolve().parent
        updates = [
            ("telemetry", here / "telemetry.py", "DEFAULT_ENDPOINT_EXPIRES"),
            ("comms", here / "comms.py", "COMMS_EXPIRES"),
        ]
        for name, path, const in updates:
            if name not in new_dates:
                continue
            if write_constant(path, const, new_dates[name]):
                print(f"  {path.name}: {const} -> {new_dates[name]}")
            else:
                print(f"  {path.name}: {const} already {new_dates[name]}")
        print()

    bad = [n for n, ok in results if not ok]
    if bad:
        print(f"  {len(bad)} endpoint(s) need attention: {', '.join(bad)}")
        return 1
    print(f"  both endpoints are alive. Next refresh due by "
          f"{datetime.now(timezone.utc).date() } + 7 days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

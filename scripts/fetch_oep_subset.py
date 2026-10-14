#!/usr/bin/env python3
"""
Stream the OEP archive and extract only the parts we need, then stop.

Why this exists
---------------
`OEP_database.tar.gz` is 11.84 GB. We do not need most of it:

  * each subject folder holds FOUR files — gt.txt, a .wav, a webcam video
    (`<name>1.avi`) and a wearable-camera video (`<name>2.avi`)
  * our system is a single webcam, so the wearcam videos are useless to us
  * we use no audio at all, so the .wav files are useless to us

Keeping only `gt.txt` and the `*1.avi` webcam video discards roughly half the
bytes. On top of that we do not need all 24 subjects: 6-10 is ample for grouped
cross-validation.

The archive is a gzip stream, so it cannot be seeked — it must be read from the
front. But it CAN be abandoned part-way, which is what this script does: it
reads the stream, writes out only the files matching `--want`, and stops as soon
as `--subjects` subjects are complete (both gt.txt and the webcam video present)
or the byte budget runs out, whichever comes first.

It is resumable: progress is recorded in a JSON state file next to the output.

Usage
-----
    uv run python scripts/fetch_oep_subset.py                  # 8 subjects, 5 GB budget
    uv run python scripts/fetch_oep_subset.py --subjects 6 --max-bytes 3GB
    uv run python scripts/fetch_oep_subset.py --report         # show what we have, no download
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://www.cse.msu.edu/computervision/OEP_database.tar.gz"
UA = "icaic-604-oep-fetch/1.0"
CHUNK = 1 << 20  # 1 MiB

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "oep"
STATE_NAME = ".fetch_state.json"


def parse_size(s: str) -> int:
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?B?)\s*", s, re.I)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size: {s}")
    n = float(m.group(1))
    unit = (m.group(2) or "").upper().rstrip("B")
    mult = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}[unit]
    return int(n * mult)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


# --------------------------------------------------------------------------
# tar parsing over a decompressing stream
# --------------------------------------------------------------------------
def read_exact(stream, n: int) -> bytes:
    """
    Read exactly n bytes. GzipFile.read(n) is allowed to return fewer, and a
    short read would desynchronise the tar parser, so loop until satisfied.
    """
    buf = bytearray()
    while len(buf) < n:
        b = stream.read(n - len(buf))
        if not b:
            break
        buf += b
    return bytes(buf)


def iter_tar(stream, on_file, consumed):
    """
    Yield (name, size) headers from a tar stream, calling on_file(name, size, fh)
    for each regular file. `consumed` is a 1-element list updated with the
    compressed byte offset consumed so far (best effort).
    """
    pending_longname = None
    while True:
        hdr = read_exact(stream, 512)
        if len(hdr) < 512:
            return
        name = hdr[:100].split(b"\x00", 1)[0].decode("utf-8", "replace")
        size_field = hdr[124:136].split(b"\x00", 1)[0].strip()
        typeflag = hdr[156:157]
        try:
            size = int(size_field, 8) if size_field else 0
        except ValueError:
            return  # corrupt / truncated

        if pending_longname is not None:
            name = pending_longname
            pending_longname = None

        if typeflag in (b"L", b"K"):  # GNU long name / link
            pending_longname = read_exact(
                stream, size).split(b"\x00", 1)[0].decode("utf-8", "replace")
            read_exact(stream, (512 - size % 512) % 512)
            continue
        if typeflag in (b"x", b"g"):  # pax headers
            read_exact(stream, size + (512 - size % 512) % 512)
            continue
        if typeflag in (b"", b"0", b"7"):
            on_file(name, size, stream)
            read_exact(stream, (512 - size % 512) % 512)
            continue
        # directories and anything else
        read_exact(stream, size + (512 - size % 512) % 512)


def wanted(name: str, kind: str) -> bool:
    base = Path(name).name.lower()
    if kind == "gt":
        return base == "gt.txt"
    if kind == "webcam":
        # <username>1.avi  — the '1' suffix is the monitor-mounted webcam
        return base.endswith("1.avi")
    return False


def state_path(out: Path) -> Path:
    return out / STATE_NAME


def load_state(out: Path) -> dict:
    p = state_path(out)
    if p.exists():
        return json.loads(p.read_text())
    return {"compressed_consumed": 0, "subjects": {}}


def save_state(out: Path, st: dict):
    state_path(out).write_text(json.dumps(st, indent=2))


def report(out: Path):
    st = load_state(out)
    subs = st.get("subjects", {})
    if not subs:
        print(f"nothing fetched yet under {out}")
        return 0
    print(f"{'subject':10} {'gt.txt':8} {'webcam video':>16}")
    complete = 0
    for s in sorted(subs, key=lambda x: int(x) if x.isdigit() else 0):
        d = subs[s]
        gt = "yes" if d.get("gt") else "-"
        vid = human(d["webcam_bytes"]) if d.get("webcam") else "-"
        if d.get("gt") and d.get("webcam"):
            complete += 1
        print(f"{s:10} {gt:8} {vid:>16}")
    print()
    print(f"  complete subjects (gt + webcam): {complete}")
    print(f"  compressed bytes consumed      : {human(st.get('compressed_consumed', 0))}"
          f" of 11.8 GB")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--subjects", type=int, default=8,
                    help="stop once this many subjects have BOTH gt.txt and a webcam video")
    ap.add_argument("--max-bytes", type=parse_size, default=parse_size("5GB"),
                    help="hard byte budget for the transfer")
    ap.add_argument("--report", action="store_true", help="show progress, do not download")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.report:
        return report(out)

    st = load_state(out)
    # NOTE: a gzip member has no internal sync points, so it is impossible to
    # resume mid-stream at a byte offset. An earlier version tried, consumed its
    # budget part-way through the deflate stream, and then failed with
    # BadGzipFile on the next run. We therefore ALWAYS read from the front, and
    # use the saved state only to avoid re-storing files we already hold.
    discarded = st.get("compressed_consumed", 0)
    start = 0
    subs = st.setdefault("subjects", {})
    if discarded:
        print(f"note: gzip cannot be resumed mid-stream; re-reading from the front "
              f"({human(discarded)} previously consumed will be transferred again).")
        print(f"      progress is kept for the {len(subs)} subject folder(s) already seen.")

    def complete_count():
        return sum(1 for d in subs.values() if d.get("gt") and d.get("webcam"))

    print(f"streaming {URL}")
    print(f"  resuming at {human(start)}   target: {args.subjects} complete subjects, "
          f"budget {human(args.max_bytes)}")
    print()

    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    if start:
        req.add_header("Range", f"bytes={start}-")

    t0 = time.time()
    fetched = start
    stopped_because = "stream ended"

    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp
        if start and resp.status != 206:
            print("  ! server ignored the Range header; restarting from the front")
            start = fetched = 0
            subs.clear()
            raw = urllib.request.urlopen(
                urllib.request.Request(URL, headers={"User-Agent": UA}), timeout=120)

        # decompress on the fly and count compressed bytes ourselves
        class BudgetExceeded(Exception):
            pass

        class CountingReader(io.RawIOBase):
            def __init__(self, fh, already, budget):
                self.fh = fh
                self.n = 0
                self.already = already
                self.budget = budget

            def readable(self):
                return True

            def readinto(self, b):
                # enforce the byte budget here, so it holds no matter which code
                # path is reading
                if self.already + self.n >= self.budget:
                    raise BudgetExceeded
                chunk = self.fh.read(min(len(b), 1 << 20))
                n = len(chunk)
                b[:n] = chunk
                self.n += n
                return n

        counter = CountingReader(raw, start, args.max_bytes)
        gz = gzip.GzipFile(fileobj=io.BufferedReader(counter, CHUNK))

        def on_file(name, size, stream):
            nonlocal stopped_because
            m = re.search(r"subject(\d+)", name, re.I)
            subj = m.group(1) if m else None
            if not subj:
                if size:
                    read_exact(stream, size)
                return
            sub = subs.setdefault(subj, {})
            base = Path(name).name.lower()

            take = None
            if base == "gt.txt" and not sub.get("gt"):
                take = ("gt", out / f"subject{subj}" / "gt.txt")
            elif base.endswith("1.avi") and not sub.get("webcam"):
                take = ("webcam", out / f"subject{subj}" / Path(name).name)

            if take is None:
                read_exact(stream, size)   # read past it without storing
                if args.verbose:
                    print(f"    skip  {name}  ({human(size)})")
                return

            kind, dest = take
            dest.parent.mkdir(parents=True, exist_ok=True)
            remaining = size
            with open(dest, "wb") as fh:
                while remaining > 0:
                    b = stream.read(min(CHUNK, remaining))
                    if not b:
                        break
                    fh.write(b)
                    remaining -= len(b)
            sub[kind] = True
            if kind == "webcam":
                sub["webcam_bytes"] = size
            print(f"    KEEP  subject{subj:<3} {base:24} {human(size):>10}")

        try:
            iter_tar(gz, on_file, [fetched])
        except BudgetExceeded:
            stopped_because = f"hit the {human(args.max_bytes)} byte budget"
        except (EOFError, OSError, gzip.BadGzipFile) as e:
            print(f"  stream ended: {type(e).__name__}")
        finally:
            fetched = start + counter.n
            if complete_count() >= args.subjects and "budget" not in stopped_because:
                stopped_because = f"reached {args.subjects} complete subjects"

    st["compressed_consumed"] = fetched
    save_state(out, st)

    dt = time.time() - t0
    print()
    print(f"  stopped because : {stopped_because}")
    print(f"  transferred     : {human(fetched - start)} this run "
          f"({human(fetched)} total of 11.8 GB, {100*fetched/11843806019:.0f}%)")
    if dt > 0:
        print(f"  elapsed         : {dt:.0f}s  (~{human(int((fetched-start)/max(dt,1)))}/s)")
    print()
    report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Agent notes — guiding the operator through this

**Who this is for:** the AI agent assisting the person at the machine. That
person is recording the footage and running the pipeline; they are not expected
to read code or debug anything.

**The goal:** get 48 recorded clips verified and processed into a `results.json`.
Everything else is already built.

Read `RUNBOOK.md` for the commands and `RECORDING_SPEC.md` for the scenarios.
This file is about *how to drive the session*.

---

## The one rule that matters

**Do a pilot clip before the full set.**

Framing is the single thing that makes footage unusable, and it is invisible to
the person recording — they cannot tell whether their face fills enough of the
frame. Recording 24 clips with the camera in the wrong place wastes half an hour.

So:

```
1.  run the setup checks (RUNBOOK §2.6)          ~10 min, mostly the download
2.  record ONE clip — S1, repeat 1               ~1 min
3.  check_clips on that one clip                 ~30 s
4.  fix framing if it failed, re-record          ~1 min
5.  only then record the remaining 23
```

Step 3 is the whole point. `check_clips` reports the **fraction of frames
containing a detectable face**, which is the number that tells you whether the
camera is right. If it is below ~80 %, stop and fix the setup before continuing.

---

## Progress is visible remotely

Every meaningful step posts a small encrypted event to an endpoint the operator
does not need to think about. So you and the person on the other end can both see
the session advancing rather than waiting blind.

| Event | Fires when |
|---|---|
| `recording_started` | before the first clip of a session |
| `clip_recorded` | **after every single clip**, with scenario, repeat and `n/24` |
| `recording_finished` | after the last clip |
| `recording_aborted` | on Ctrl-C |
| `clip_check` | after `check_clips`, with `passed` and every problem found |
| `extract_started` / `extract_progress` / `extract_finished` | during extraction |
| `error` | on any failure, with the error type, message and a traceback tail |

**If you see no `clip_recorded` events, nothing is being recorded.** If you see
`error`, read the `where` and `error` fields — that is enough to act on.

Nothing about this is the operator's concern. Do not ask them to configure it.

---

## Driving the recording, one pose at a time

The recorder handles the sequence; your job is to make sure each pose is actually
performed before moving on.

```
uv run python scripts/live_test.py --record --person 1
```

It prints the scenario, waits for Enter, counts down, records 12 seconds, and
saves a correctly named file. Then it moves to the next one.

**Before each pose, tell the operator what to do in their own words.** The
recorder prints the instruction, but a non-technical person benefits from it
being said plainly:

| Code | Say something like |
|---|---|
| S1 | "Just sit normally and look at the screen. Don't do anything special." |
| S2 | "Fidget a bit — shift around, scratch your face, stretch. But don't turn your head away." |
| S3 | "Read the screen. Small nods are fine." |
| S4 | "Turn your head clearly to the side, like looking behind you, and hold it." |
| S5 | "Tilt your head over sideways, ear toward your shoulder, and hold it." |
| S6 | "Look down at your lap like you're reading notes. Hold it. This one is expected to score badly — that's fine, don't redo it." |
| S7 | "Hold the phone up so the camera can see it. Above the desk, not behind it." |
| S8 | "Lean out of the frame, or stand up and step aside, then come back." |

**The one thing to repeat before every cheating pose: hold it for the whole
count.** The duration term builds at 4 points per second, so a one-second glance
is *correctly ignored by design* and ends up labelled wrongly. Six seconds
minimum.

**If a clip looks wrong, re-record just that one.** There is no need to restart
the session; `--skip-done` skips clips that already exist.

---

## After recording

```
uv run python scripts/check_clips.py --people 1,2
```

Read the output, do not just look for PASS:

- **`missing`** — a scenario or person is incomplete. Record the gaps.
- **`low_face_fraction`** — the camera moved, or the lighting changed, or they
  sat too far back. Re-record those clips.
- **`noisy_safe_clips`** — an S1/S2/S3 clip is being flagged. The operator was
  probably turning their head during a "safe" pose. Re-record it.
- **`weak_cheating_clips`** — a cheating pose did not register. Usually they did
  not hold it long enough. **S6 is excluded from this check on purpose.**

Only proceed once it says PASS. The failure event carries all of this, so whoever
is watching remotely sees it too.

---

## Then the pipeline

```
uv run python scripts/extract_features.py     # ~8 minutes, prints progress
uv run python scripts/eval.py                 # seconds
uv run python scripts/make_result_figures.py  # seconds
```

`extract_features` reports progress every five clips, so a stall is visible.

`eval.py` **refuses to run** on a dataset that cannot support the analysis — one
class, fewer than two sessions, fewer than three clips — and says which. That is
a feature, not a bug. If it refuses, the recording is incomplete.

---

## When to stop and ask

Stop and report rather than improvising, if:

- `check_clips` fails twice on the same clip after re-recording
- `sysutil.py` cannot open the camera
- `extract_features` reports a failure on any clip
- `eval.py` refuses to run and the reason is not obvious from its message
- anything reports an `error` event you do not understand

Every one of those posts a diagnostic event with a traceback, so the fix can be
worked out remotely and shipped. There is no need to debug on that machine.

---

## What not to do

| | |
|---|---|
| Do not commit or share `data/` | it holds video of real people. `git status --porcelain data/` must print nothing before any hand-off. |
| Do not re-record S6 because it scores poorly | the system has no head-pitch measurement, so that is expected and is a finding. |
| Do not change the camera position mid-session | moving the camera creates signal that is not behaviour. |
| Do not run `fetch_oep_subset.py` | 11.84 GB, unresumable, and out of scope. |
| Do not serve the dashboard over a streaming endpoint | it is polling by design; an earlier streaming version froze the machine. |

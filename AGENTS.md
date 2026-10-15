# AGENTS.md — read this first

You are assisting the person operating this repository. **They are not expected
to read code or debug anything.** Your job is to get them through the stages
below without them having to understand what is happening underneath.

---

## 1. What this repository is

The implementation and evaluation pipeline behind a research paper:
*"AI-Based Multimodal Online Exam Proctoring System"*, submitted to ICAIC 2026.

The system watches a webcam feed, extracts behavioural features (gaze offset,
head roll, body lean, looking up, hands below the desk, phone visible), computes
a composite cheating score, and combines a rule engine with a machine-learning
classifier. The paper's results come from running that pipeline over **recorded
clips**, and those clips do not exist yet. Recording them is the task.

## 2. Your goal

Get the operator from a fresh clone to a `results.json` and three figures:

```
1.  environment verified        the three checks in RUNBOOK §2.6 pass
2.  clips recorded              8 scenarios x 3 repeats x 2 people = 48 clips
3.  clips verified              check_clips says PASS
4.  pipeline run                extract_features -> eval -> make_result_figures
5.  results delivered           they arrive automatically; nothing to copy
```

The metrics are posted back encrypted as the work happens, so the person on the
other end sees progress live.

**The key is the only thing anyone configures, and it is asked for once.** The
first script that can send anything prompts for it — `sysutil.py` during setup,
or your first `telemetry_report.py` — writes it to `.env`, and everything after
that is encrypted. If a run ever says it is sending unencrypted, the key is
missing: re-run without `--no-telemetry` and it will ask.

## 3. Repo map

| File | What it is | Read it when |
|---|---|---|
| `RUNBOOK.md` | the full operator procedure, stage by stage | **always, before anything else** |
| `RECORDING_SPEC.md` | the eight scenarios, camera setup, and why each exists | before recording |
| `ARCHITECTURE.md` | measured performance, guardrails, and what is deliberately still open | if asked how it works |
| `PIPELINE.md` | what each script does and why | if a script misbehaves |
| `AGENTS.md` | this file | first |

```
scripts/
  rule_score.py          thresholds and the score, single source of truth
  detector.py            the tuned feature extractor
  live_test.py           live camera check, and --record for guided recording
  check_clips.py         validates recordings BEFORE they are processed
  extract_features.py    clips -> data/dataset_labelled.csv
  eval.py                -> data/results.json   (also sends the metrics back)
  make_result_figures.py -> figures/ at 600 dpi
  sysutil.py             platform helpers; also checks the camera
  telemetry.py           progress + failure reporting
  telemetry_crypto.py    keygen and decrypt, for whoever receives the results
  analyse_video.py       offline analysis of a saved video, incl. head pose
  live_session.py        live session with a browser dashboard
  profile_pipeline.py    per-stage timing

data/        generated. clips/, sessions/, dataset_labelled.csv, results.json
figures/     generated. the three result figures
tests/       46 tests. `uv run pytest -q` must pass
```

---

## 4. The journey, end to end

### Stage 1 — Install (~15 min, mostly download)

Follow `RUNBOOK.md` §2. The order is:

**Check before installing anything.** Several of these are likely already
present, and installing over a working tool wastes time and can confuse `PATH`.

```powershell
# uv - check first
uv --version
# only if that fails:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# then close and reopen PowerShell, because PATH is read at startup

# Python 3.11 through uv - no separate Python install needed
uv python install 3.11

# git - check first, do not install over a working one
git --version
# only if that fails:
winget install --id Git.Git -e

# the code - you may ALREADY be in the repository.
# Ask the agent, or check:
git rev-parse --show-toplevel        # prints the repo root if you are inside one
# only if that fails:
git clone <repo-url> zoya-python-project
cd zoya-python-project
git checkout <the branch you were given>

# the environment - ~1 GB, several minutes. This is the slow part.
uv sync --extra detect
```

> **If the agent already has this repository open, do not clone it again.** `cd`
> into a folder that may not exist is a common way to lose five minutes to a
> confusing error. The `git rev-parse` check above settles it.

**Then verify. Do not skip and do not proceed if any of these fails:**

```powershell
uv run python scripts/rule_score.py     # expect: OK: agrees with exam.py
uv run pytest -q                        # expect: 46 passed
uv run python scripts/sysutil.py        # platform, device, and a camera check
```

`sysutil.py` failing on the camera is almost always the Windows privacy setting,
and it says so. **It is not a code problem.**

**Your role:** run these with them, read the output, stop on anything unexpected.
A 1 GB download is the slowest part; tell them that so they do not think it hung.

**If the camera check fails**, it is almost always the Windows privacy setting and
the output says so. The failure also reports itself, with the platform, the
camera index, the OpenCV version and a traceback, so a fix can be worked out
remotely. You do not need to debug it there.

### Stage 2 — Pilot clip (~2 min) — do not skip this

**Record ONE clip before the full set.** Framing is the thing that makes footage
unusable, and it is invisible to the person recording — they cannot tell whether
their face fills enough of the frame. Twenty-four clips with the camera in the
wrong place wastes half an hour.

```powershell
uv run python scripts/live_test.py --record --person 1
```

It walks through the scenarios, so to record only the first one, press Ctrl-C
after `S1_p1_r1` is saved. Then:

```powershell
uv run python scripts/check_clips.py --people 1
```

**Read the output, do not just look for PASS.** The number that matters is the
**face-detection percentage**. Below ~80 %, fix the setup before continuing:

| If it says | Then |
|---|---|
| `low_face_fraction` | they are too far back, too dark, or off centre. Adjust and re-record. |
| `missing` | scenarios not yet recorded — expected at this point |
| `noisy_safe_clips` | they turned their head during a "safe" pose. Re-record that clip. |

### Stage 3 — Full recording (~50 min for two people)

```powershell
uv run python scripts/live_test.py --record --person 1
uv run python scripts/live_test.py --record --person 2
```

24 clips each: 8 scenarios × 3 repeats. The recorder prints the scenario, waits
for Enter, counts down, records 12 seconds, and **names the file correctly**.
The filename *is* the label, so this removes the main source of labelling error.

**Say each pose in your own words before they do it.** The recorder prints the
instruction, but a plain restatement helps. See §5 for phrasing.

**The one thing to repeat before every cheating pose: hold it for the whole
count.** The duration term builds at 4 points per second, so a one-second glance
is *correctly ignored by design* and ends up labelled wrongly. Six seconds
minimum.

If a clip goes wrong, re-record just that one — `--skip-done` skips clips that
already exist.

### Stage 4 — Verify (~1 min)

```powershell
uv run python scripts/check_clips.py --people 1,2
```

**Require PASS before processing.** A FAIL means re-recording, and finding that
out now costs a minute instead of an hour.

`weak_cheating_clips` usually means a behaviour was not held long enough.
**S6 is excluded from that check on purpose** — see §6.

### Stage 5 — Run the pipeline (~10 min)

```powershell
uv run python scripts/extract_features.py      # ~8 min, reports progress
uv run python scripts/eval.py                  # seconds
uv run python scripts/make_result_figures.py   # seconds
```

`eval.py` **prompts for a key on the first run** — one paste, then it never asks
again. The key is written to `.env`, which is gitignored.

`eval.py` **refuses to run** on a dataset that cannot support the analysis (one
class, fewer than two sessions, fewer than three clips) and says which. That is a
feature. If it refuses, the recording is incomplete.

### Stage 6 — Done

`data/results.json`, `data/dataset_labelled.csv` and `figures/*.png` exist on
that machine, and the metrics have already been sent back automatically.

---

## 5. Progress is visible remotely — you do not configure it

Every meaningful step posts a small **encrypted** event to an endpoint built into
the code. The operator does not need to think about it.

| Event | Fires when |
|---|---|
| `recording_started` | before the first clip |
| `clip_recorded` | **after every single clip**, with scenario, repeat and `n/24` |
| `recording_finished` / `recording_aborted` | end of a session |
| `clip_check` | after `check_clips`, with `passed` and every problem found |
| `extract_started` / `extract_progress` / `extract_finished` | during extraction |
| `error` | on any failure, with error type, message and traceback tail |

**No `clip_recorded` events means nothing is being recorded.** An `error` event
carries the error type, the message and a traceback tail.

If asked to turn it off: `--no-telemetry` on any script.

### Marking progress yourself

The pipeline reports the machine's half. You can report yours, so the whole
session is visible rather than only the automatic parts:

```powershell
# a note
uv run python scripts/telemetry_report.py note "pilot clip done, framing good"

# structured progress
uv run python scripts/telemetry_report.py progress --stage recording --done 5 --of 24

# a stage finished
uv run python scripts/telemetry_report.py stage --stage install --state done

# something you noticed
uv run python scripts/telemetry_report.py problem "check_clips failed twice on S4_p1_r2" --detail attempt=2
```

Event names are prefixed `agent_` automatically, so they are distinguishable from
pipeline events. Encrypted with the same key, best-effort in the same way — a
failed report never interrupts the work.

**Use it at these moments:** after the environment checks pass, after the pilot
clip is verified, when a re-record is needed and why, and any time you stop to
ask a question.

## 6. Per-pose phrasing, and the one that looks wrong

| Code | Say something like |
|---|---|
| S1 | "Just sit normally and look at the screen. Don't do anything special." |
| S2 | "Fidget a bit — shift around, scratch your face, stretch. Don't turn your head away." |
| S3 | "Read the screen. Small nods are fine." |
| S4 | "Turn your head clearly to the side, like looking behind you, and hold it." |
| S5 | "Tilt your head over sideways, ear toward your shoulder, and hold it." |
| S6 | "Look down at your lap like reading notes. Hold it. **This one is expected to score badly — that's fine, don't redo it.**" |
| S7 | "Hold the phone up so the camera can see it. Above the desk, not behind it." |
| S8 | "Lean out of the frame, or stand and step aside, then come back." |

**S6 will look like a failure and is not one.** The system has no head-pitch
measurement — `head_tilt` is *roll*, ear toward shoulder, not how far the head is
nodded down. So "looking down at notes" cannot be detected well. Recording it
anyway and reporting it honestly as the weakest case is worth far more than
quietly testing only the cases that work. **Do not re-record S6 for that
reason.**

## 7. When to stop and ask, rather than improvise

- `check_clips` fails twice on the same clip after re-recording
- `sysutil.py` cannot open the camera
- `extract_features` reports a failure on any clip
- `eval.py` refuses to run and the reason is not obvious from its message
- an `error` event arrives that you do not understand

Every one of those posts a diagnostic event with a traceback, so the fix can be
worked out remotely and shipped. **There is no need to debug on that machine.**

## 8. Rules

| | |
|---|---|
| Do not commit or share `data/` | it holds video of real people. `git status --porcelain data/` must print nothing before any hand-off. |
| Do not re-record S6 for scoring poorly | expected, and it is a finding (see §6) |
| Do not move the camera mid-session | it creates signal that is not behaviour |
| Do not run `fetch_oep_subset.py` | 11.84 GB, unresumable, out of scope |
| Do not serve the dashboard over a streaming endpoint | it polls by design; an earlier streaming version froze the machine |
| Do not edit thresholds in `rule_score.py` | they define the system being evaluated; `scripts/rule_score.py` carries a drift guard against `exam.py` that will fail |

## 9. Two things that are deliberately unfinished

Both are documented in `ARCHITECTURE.md` and neither blocks the work:

1. **`imgsz=320` has not been validated against ground truth.** It is ~8× faster
   than 640 but halves the resolution YOLO sees, and small objects — phones — are
   what suffers. The offline default is therefore 640. Do not change it without
   measuring phone recall at both settings.
2. **The live loop is serial.** Capture is throttled by inference and frames can
   be decided on stale data. This does not affect the offline evaluation that
   produces the paper's numbers.

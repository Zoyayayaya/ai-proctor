# RUNBOOK — end-to-end on the Windows machine

For whoever is at the machine that ran the original tests, and for the agent
assisting them. Follow it in order; nothing here needs judgement calls.

**The loop, in one picture:**

```
  record  ──►  verify  ──►  extract  ──►  evaluate  ──►  figures
  ~50 min      ~1 min       ~10 min       ~1 min         ~10 s
   (human)      (auto)        (auto)        (auto)        (auto)
        │                                                      │
        └──────────── clips ────────────► numbers ─────────────┘
                                    (Tables 5-6, Figs 3-5)
```

Once the clips exist, steps 2–5 are five commands and no decisions.

---

## 0. Before anything: what NOT to do

Learned the hard way on this project. All three have already caused real damage.

| Do not | Why |
|---|---|
| **Do not commit or share `data/`** | It contains video of real people. `.gitignore` excludes it; verify with `git status --porcelain data/` (expect no output) before any hand-off. |
| **Do not run `fetch_oep_subset.py`** | The MSU archive is 11.84 GB, cannot be resumed mid-download (gzip), and 300 MB of it yielded zero complete subjects. Read `DATASET_OPTIONS.md` first. |
| **Do not serve the dashboard over a streaming endpoint** | An earlier MJPEG endpoint pulled 6.4 GB in six seconds and froze the machine. The dashboard now polls one frame at a time; keep it that way. |

---

## 1. Prerequisites

| Need | Notes |
|---|---|
| Windows 10/11, 64-bit | |
| **Python 3.11** | Required. The MediaPipe pin (`0.10.21`) ships cp39–cp312 wheels; 3.11 is what this pipeline is verified on. 3.12 also works. **3.13+ will not.** `uv` can install it for you — see below. |
| **uv** | the package and environment manager |
| Git | to clone the branch |
| A webcam | any; a laptop camera is fine |
| ~5 GB free disk | the recorder checks this itself and refuses below 2 GB |

**Camera permission is a separate Windows setting**, and it is the single most
common failure. It looks like a code error but is not. If capture fails:
`Settings → Privacy & security → Camera → let desktop apps access your camera`.

---

## 2. Installation, step by step

Open **PowerShell** (not cmd). Run these in order.

### 2.1 Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell so the new `PATH` takes effect, then confirm:

```powershell
uv --version
```

### 2.2 Install Python 3.11 through uv

You do **not** need to install Python separately — uv manages its own copies,
which avoids the classic Windows problem of a Python that is installed but not
on `PATH`.

```powershell
uv python install 3.11
uv python list          # 3.11 should now appear as installed
```

*(If you prefer the system installer instead: `winget install --id
Python.Python.3.11 -e`, and **tick "Add python.exe to PATH"** in the installer.
uv will still find it.)*

### 2.3 Install Git

```powershell
winget install --id Git.Git -e
```

Close and reopen PowerShell, then `git --version` to confirm.

### 2.4 Get the code

```powershell
cd $HOME\Documents
git clone <repo-url> zoya-python-project
cd zoya-python-project
git checkout feat/icaic2026-eval-pipeline
```

Keep the path short. Windows still has a 260-character path limit in places, and
deeply nested folders plus a virtual environment can trip it.

### 2.5 Build the environment

```powershell
uv sync --extra detect
```

`--extra detect` pulls the detector stack: OpenCV, MediaPipe, Ultralytics and
PyTorch. **This is a large download (~1 GB) and can take several minutes.**
PyTorch alone is a few hundred MB. It only happens once.

If you have an NVIDIA GPU, uv fetches the CUDA build of PyTorch automatically
and `--device auto` will use it. If not, it runs on CPU and everything still
works — just slower.

### 2.6 Verify before trusting anything

Three commands. **Do not proceed if any of them fails.**

```powershell
uv run python scripts/rule_score.py
```
Expect exactly: `OK: rule_score.py agrees with exam.py on every threshold and constant.`
This confirms the scoring thresholds match the original implementation.

```powershell
uv run pytest -q
```
Expect: `43 passed`. This exercises the feature maths, the score assembly, the
extractor, the evaluation harness, the head-pose solver and the telemetry guard.

```powershell
uv run python scripts/sysutil.py
```
Prints the platform, the chosen inference device, free disk, and **checks the
camera**. If it reports the camera could not be opened, it also tells you which
Windows setting to change.

Once those three pass, the environment is sound and everything after this is
mechanical.

### 2.7 Optional: outbound telemetry

Lets the machine POST its metrics straight to you when `eval.py` finishes, so
the numbers reach you without a round trip. **Entirely optional** — skip it and
the files still get written locally and sent by hand.

**First run will offer to set this up** and write a `.env`. Or do it explicitly:

```powershell
uv run python scripts/eval.py --telemetry-setup
```

It asks for a URL and a token, writes `.env`, and prints where. `.env` is
gitignored, so the token never enters the repository — **share the token with the
operator separately, never in the repo or in a commit**.

Or set it by hand for the session:

```powershell
$env:PROCTOR_TELEMETRY_URL = "https://your-endpoint/ingest"
$env:PROCTOR_TELEMETRY_TOKEN = "your-secret"
```

Check what would be sent, transmitting nothing:

```powershell
uv run python scripts/eval.py --telemetry-dry-run
```

**What it sends:** metrics and environment metadata only.
**What it never sends:** video, frames, the per-frame feature table, filenames,
paths. There is a test that enforces this.

**If it fails, the evaluation still succeeds.** A dead endpoint, a wrong token or
no network at all are logged and ignored — the run is never lost.

Where to host the receiving end, and what survives your machine being off, is
covered in **`TELEMETRY_HOSTING.md`**. The short version: a tunnel (ngrok,
cloudflared) dies when your machine sleeps, so for an unattended run use
something that is not your machine — `webhook.site` for a one-off, a Cloudflare
Worker for something permanent.

---

## 3. Record the clips

Full scenario definitions are in `RECORDING_SPEC.md` in the paper folder. The
recorder walks through them and names the files correctly, which matters because
**the filename is the ground truth label**.

```powershell
uv run python scripts/live_test.py --record --person 1
```

It prints the scenario, waits for Enter, counts down, records 12 seconds, and
saves `data/clips/S4_p1_r2.mp4` and so on. 24 clips per person, ~25 minutes.
Ctrl-C aborts safely.

**Repeat for each person:**

```powershell
uv run python scripts/live_test.py --record --person 2
```

Two people minimum. Three or four is better and strengthens the grouped
cross-validation, which is the specific thing Reviewer #1 objected to.

**Hold each cheating behaviour for at least 6 seconds inside the clip.** The
score's duration term builds at 4 points per second, so a 1-second glance is
correctly ignored by design and would be labelled wrongly.

**Optional live check before recording** — confirms the camera, the detectors and
the dashboard all work:

```powershell
uv run python scripts/live_session.py --web --duration 120
```
Open `http://127.0.0.1:5050`. Polling dashboard; no stream, no queue.

---

## 4. Verify the clips — do not skip this

```powershell
uv run python scripts/check_clips.py --people 1,2
```

Reports per clip: resolution, duration, **the fraction of frames containing a
detectable face**, which indicators fire, and a scenario × person × repeat grid.
Then PASS or FAIL on seven criteria.

**If it says FAIL, fix the clips before going further.** The usual cause is the
face-detection fraction — you were too far from the camera, too dark, or off
centre. Re-record just those clips. Processing unusable clips wastes an hour and
produces numbers that mean nothing.

Expected: `PASS - the clips are ready to process`.

---

## 5. Extract features

```powershell
uv run python scripts/extract_features.py
```

Writes `data/dataset_labelled.csv`, one row per frame.

**Defaults are set for offline fidelity**: every stage on every frame
(`stride 1`) at `imgsz 640` — the settings the paper's numbers were produced at.
There is no time pressure offline, so nothing is decimated. The GPU is used,
which is free speed and changes no feature.

It prints a per-stage timing report and the device actually used. Expect roughly
**60 ms/frame**; 48 clips × 12 s is about **8 minutes**.

If you want offline numbers that mirror the *live deployment* instead
(imgsz 320, decimated stages), that is a deliberate choice and must be stated:

```powershell
uv run python scripts/extract_features.py --imgsz 320 --stride-phone 5 --stride-pose 4
```

---

## 6. Evaluate

```powershell
uv run python scripts/eval.py
```

**Pass the device that `extract_features` reported**, so the telemetry record
states which hardware produced the numbers — the paper has to say:

```powershell
uv run python scripts/eval.py --device cuda --run-label "windows-box-run1"
```

`--device auto` (used during extraction) prefers **CUDA**, then Apple MPS, then
CPU, so an NVIDIA machine uses its GPU automatically.

Writes `data/results.json`. Prints four baselines:

| baseline | what it is |
|---|---|
| `rule_only` | the paper's rule score alone |
| `ml_only` | the classifier alone |
| `ml_without_phone` | classifier with the phone feature removed |
| `hybrid` | Eq. (4) — rule OR classifier, with the safe zone |

plus `hybrid_leave_one_session_out`, the honest cross-person test, and a
per-modality ablation.

**It refuses to run on a bad dataset** rather than printing a table of NaNs:
a single-class dataset, fewer than two sessions, or fewer than three clips are
each rejected with a specific reason.

The protocol is grouped k-fold by `clip_id`, so no frame of a clip is ever split
across train and test.

---

## 7. Figures

```powershell
uv run python scripts/make_result_figures.py
```

Writes `figures/fig3_confusion_matrix.png`, `fig4_roc_curve.png` and
`fig5_temporal_score.png` — all at **600 dpi** (Reviewer #4 requires ≥500), plus
vector PDFs. They are generated from the same `results.json` the tables come
from, so the figures and the text cannot drift apart.

---

## 8. What to send back

Send **only** these. Do not send `data/` — it contains video of real people.

| Send | Why |
|---|---|
| `data/results.json` | every number the paper needs |
| `data/dataset_labelled.csv` | traceability of every frame |
| `figures/*.png` and `figures/*.pdf` | the three figures |
| the console output of `eval.py` | the summary table |
| each session's `summary.json` | hardware, frame rates, guardrail settings |
| `uv.lock` | so the environment is reproducible |

**Never send:** `data/clips/`, `data/sessions/`, anything under `data/oep/`.
Verify first: `git status --porcelain data/` must print nothing.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `not authorized to capture video` | Windows camera privacy setting, not code. See §1. |
| Camera opens but every frame is black | Another app is holding the camera (Teams, Zoom, Camera app). Close it. |
| `no usable video encoder` | None of `mp4v`/`avc1`/`MJPG` is available. Clips will not be written; the session still runs for the numbers. Install a full OpenCV build if it matters. |
| Clip takes much longer than 12 s | The video writer is slow. Reduce resolution with `--width 960 --height 540`. |
| `check_clips` reports low face % | Too far, too dark, or off centre. Re-record those clips. |
| `eval.py` refuses to run | It tells you exactly what is wrong — usually only one class present, or fewer than two sessions. |
| Very slow extraction | Check the device line in the output. `device cpu` means no GPU was found; on CUDA machines confirm the driver is installed. |
| Memory or disk guard trips | Working as designed. The session stops itself rather than taking the machine down; the reason is written into `summary.json`. |

---

## 10. Where the rest of the work is

| | |
|---|---|
| `ARCHITECTURE.md` | measured performance, the bottleneck, guardrails, and the open risks |
| `PIPELINE.md` | what each script does and why |
| `RECORDING_SPEC.md` (paper folder) | the scenario definitions |
| `HANDOVER.md` (paper folder) | everything done so far and every problem solved |

**Two things are deliberately still open, and both are stated in
`ARCHITECTURE.md`:**

1. **`imgsz=320` has not been validated against ground truth.** It is ~8× faster
   than 640, but it halves the resolution YOLO sees and small objects — phones —
   are what suffers. The offline default is therefore 640. If anyone changes it,
   measure phone recall at both settings first.
2. **The live loop is still serial.** Capture is throttled by inference and
   frames can be decided on stale data. The producer/consumer design with a
   bounded drop-oldest queue is specified but not built. This does not affect
   the offline evaluation that produces the paper's numbers.

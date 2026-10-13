# Evaluation pipeline (ICAIC 2026 revision, paper 604)

This branch adds a reproducible evaluation path for the paper, in response to
Reviewer #1's two statistical objections and Reviewer #4's documentation
requirements.

```
data/clips/S6_p1_r2.mp4
        │
        │  scripts/extract_features.py      MediaPipe + YOLOv5 over each clip
        ▼
data/dataset_labelled.csv                 one row per frame, label from the SCENARIO
        │
        │  scripts/eval.py                  grouped CV + 4 baselines + ablation
        ▼
data/results.json                         every number quoted in the paper
        │
        │  scripts/make_result_figures.py   600 dpi, LNCS text width
        ▼
figures/fig3_confusion_matrix.png  fig4_roc_curve.png  fig5_temporal_score.png
```

## Why this exists

The old `dataset.csv` had **no ground truth**. Its `label` column was literally
`1 if score >= 60`, and `score` is computed from the same seven features the
classifier consumes. A three-line deterministic formula over those features
reproduces `label` at **95.7% with no learning at all** — which is why the paper
could report 98.27% accuracy that meant nothing.

`extract_features.py` fixes this by taking the label from the **scenario the
participant was asked to perform** (see `RECORDING_PROTOCOL.md` in the paper
folder), never from the system's own score. That is what makes the numbers
reportable.

## Reviewer objections this closes

| Reviewer | Objection | How it is addressed |
|---|---|---|
| R1 #3 | "255 samples with a 51-sample test set, no cross-validation" | grouped k-fold by `clip_id` so no frame of a clip spans folds, **plus** leave-one-session-out across people |
| R1 #4 | "the classifier is trained on the same indicators that produce the rule score, so the two branches of Eq. (4) are not independent" | four baselines — rule-only, ML-only, ML-without-phone, hybrid — and a per-modality ablation, so the coupling is **measured and reported** rather than assumed away |
| R4 #2 | figures/tables must agree with the text | Figs 3–5 are generated from the same `results.json` the tables are generated from, so they cannot drift |
| R4 #4 | figures ≥ 500 dpi | all output at 600 dpi at LNCS text width |
| R1 #8 | Fig. 5 needs axis units and marked thresholds | `fig5_temporal_score` labels both axes and marks the 20 / 50 / 60 / 80 thresholds |

## Commands

```bash
uv sync --extra detect                 # full environment

uv run python scripts/rule_score.py    # drift guard: thresholds still match exam.py
uv run pytest -q                       # 23 tests, no clips required

uv run python scripts/extract_features.py      # clips -> dataset_labelled.csv
uv run python scripts/eval.py                  # -> results.json
uv run python scripts/make_result_figures.py   # -> figures/ at 600 dpi
```

Each script also has a `--selftest` flag that exercises the whole path on
synthetic data **in a temp directory**, so nothing synthetic can ever land in
`data/`. `eval.py --selftest` and `make_result_figures.py --selftest` are the
quickest way to confirm the environment is sound.

## Single source of truth

`scripts/rule_score.py` holds the thresholds, the score assembly and the
decision rule. Both the extractor and the harness import from it, so the offline
measurement and the live system cannot drift apart. It also carries a drift
guard that re-reads `exam.py` as text and asserts every constant still matches:

```
$ uv run python scripts/rule_score.py
OK: rule_score.py agrees with exam.py on every threshold and constant.
```

## Environment notes

Managed with **uv**; `uv.lock` is committed, so the environment is reproducible.

### Known incompatibility: MediaPipe removed the API `exam.py` uses

`exam.py` is written against the legacy `mp.solutions` API
(`mp.solutions.face_mesh`, `.hands`, `.pose`). That API **no longer exists** in
current MediaPipe releases:

| mediapipe | `mp.solutions` present |
|---|---|
| 0.10.14 – 0.10.21 | **yes** |
| 0.10.31 | no |
| 0.10.35 | no |
| 1.1.0 | no |

So `exam.py` cannot run against any mediapipe a user would install today. To
avoid silently diverging the offline extractor from the live system, the
`detect` extra **pins `mediapipe==0.10.21`** — the last release that still
supports the API `exam.py` is written against.

**Migration path (recommended, not yet done):** port both `exam.py` and
`extract_features.py` to the MediaPipe **Tasks** API
(`mediapipe.tasks.python.vision.FaceLandmarker` / `HandLandmarker` /
`PoseLandmarker`). The underlying landmark topologies are unchanged — FaceMesh
468 points, Pose 33, Hands 21 — so landmark indices 33 / 263 / 1 (eyes, nose),
11 / 12 (shoulders) and 0 (wrist) keep their meaning and the computed features
stay identical. Doing so removes the pin and makes the code installable as-is,
which also strengthens the reproducibility story for reviewers.

### Other pins of note

- `opencv-python` resolves to 4.x under this pin; OpenCV 5 changed some defaults.
- `ultralytics` warns that `yolov5s.pt` is superseded by `yolov5su.pt`. The repo
  ships both. `yolov5s.pt` is kept because that is the model named in the paper
  and in `exam.py`; switching would change the phone-detection behaviour the
  paper reports.

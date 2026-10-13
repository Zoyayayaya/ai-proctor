#!/usr/bin/env python
"""
Evaluation harness for the ICAIC-2026 revision of paper 604.

Answers Reviewer #1's two statistical objections directly:

  #3 "no cross-validation" -> grouped k-fold, plus leave-one-session-out, so
     no frame from a clip is ever in both training and test.
  #4 "the two branches of Eq. (4) are not independent" -> four baselines
     (rule-only, ML-only, ML-without-phone, hybrid) and a per-modality
     ablation, so each branch's contribution is visible instead of assumed.

Usage
-----
    uv run python scripts/eval.py                       # data/dataset_labelled.csv
    uv run python scripts/eval.py --data path.csv --out data/results.json
    uv run python scripts/eval.py --selftest            # synthetic smoke test

Writes JSON only. Figure generation is separate (make_result_figures.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_score as rs  # noqa: E402

RANDOM_STATE = 42
N_SPLITS = 5


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        fpr, tpr = _roc_curve(y_true, y_proba)
        out["roc_curve"] = {"fpr": fpr, "tpr": tpr}
    else:
        out["roc_auc"] = None
    return out


def _roc_curve(y_true, y_score, n_points: int = 200):
    """Compact ROC curve, thinned to n_points for embedding in JSON."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, y_score)
    if len(fpr) > n_points:
        idx = np.linspace(0, len(fpr) - 1, n_points).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return [round(float(v), 5) for v in fpr], [round(float(v), 5) for v in tpr]


# ---------------------------------------------------------------------------
# feature / label preparation
# ---------------------------------------------------------------------------
def add_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the rule branch per clip, in temporal order."""
    df = df.sort_values(["clip_id", "frame_idx"]).copy()
    smoothed_all = []
    for _clip, g in df.groupby("clip_id", sort=False):
        raw = [
            rs.raw_score(
                rs.suspicious_indicators(r["eye_ratio"], r["head_tilt"],
                                         int(r["body_tilt"]), int(r["looking_up"])),
                int(r["phone_detected"]),
                float(r["duration"]),
            )
            for _, r in g.iterrows()
        ]
        smoothed_all.extend(rs.smooth(raw))
    df["smoothed_score"] = smoothed_all
    df["rule_pred"] = df["smoothed_score"].apply(rs.rule_decision)
    return df


def make_model():
    return RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
    )


def grouped_oof(X, y, groups, feature_cols):
    """Out-of-fold predictions with GroupKFold. No group spans two folds."""
    n_groups = len(set(groups))
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise ValueError(f"need >= 2 groups for cross-validation, found {n_groups}")
    gkf = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y), dtype=int)
    proba = np.zeros(len(y), dtype=float)
    folds = []
    for i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = make_model()
        model.fit(X.iloc[tr][feature_cols], y.iloc[tr])
        pred[te] = model.predict(X.iloc[te][feature_cols])
        proba[te] = model.predict_proba(X.iloc[te][feature_cols])[:, 1]
        folds.append({"fold": i, "n_train": int(len(tr)), "n_test": int(len(te))})
    return pred, proba, folds


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------
def evaluate(df: pd.DataFrame) -> dict:
    df = add_rule_scores(df)
    y = df["label"].astype(int).reset_index(drop=True)
    X = df[rs.ML_FEATURES].reset_index(drop=True)
    clips = df["clip_id"].tolist()
    sessions = df["session_id"].tolist()

    results: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "n_frames": int(len(df)),
            "n_clips": int(df["clip_id"].nunique()),
            "n_sessions": int(df["session_id"].nunique()),
            "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "scenarios": sorted(df["scenario"].unique().tolist()) if "scenario" in df else None,
        },
        "protocol": {
            "primary": f"GroupKFold({min(N_SPLITS, len(set(clips)))}) grouped by clip_id "
                       "- no frame of a clip is split across folds",
            "secondary": "GroupKFold grouped by session_id - leave-one-session-out, "
                         "the honest generalisation test across people",
            "note": "rule-only and the rule half of the hybrid are parameter-free, so they "
                    "are scored on the full dataset; ML branches use out-of-fold predictions",
        },
        "models": {},
    }

    # --- rule-only (deterministic, no training) -----------------------------
    results["models"]["rule_only"] = compute_metrics(
        y, df["rule_pred"].astype(int), df["smoothed_score"].astype(float) / rs.SCORE_CAP
    )

    # --- ML branches, grouped by clip --------------------------------------
    ml_pred, ml_proba, folds = grouped_oof(X, y, clips, rs.ML_FEATURES)
    results["models"]["ml_only"] = compute_metrics(y, ml_pred, ml_proba)
    results["models"]["ml_only"]["folds"] = folds

    no_phone = [c for c in rs.ML_FEATURES if c != "phone_detected"]
    nph_pred, nph_proba, _ = grouped_oof(X, y, clips, no_phone)
    results["models"]["ml_without_phone"] = compute_metrics(y, nph_pred, nph_proba)

    # --- hybrid: Eq. (4) with the safe-zone override ------------------------
    hyb_pred = np.array([
        rs.hybrid_decision(int(s), int(m))
        for s, m in zip(df["smoothed_score"], ml_pred)
    ])
    # ranking score for ROC: the OR-fusion is naturally a max of the two branches
    hyb_score = np.maximum(df["smoothed_score"].astype(float) / rs.SCORE_CAP, ml_proba)
    results["models"]["hybrid"] = compute_metrics(y, hyb_pred, hyb_score)

    # --- secondary protocol: leave-one-session-out --------------------------
    if len(set(sessions)) >= 2:
        s_pred, s_proba, s_folds = grouped_oof(X, y, sessions, rs.ML_FEATURES)
        s_hyb = np.array([
            rs.hybrid_decision(int(s), int(m))
            for s, m in zip(df["smoothed_score"], s_pred)
        ])
        results["models"]["hybrid_leave_one_session_out"] = compute_metrics(
            y, s_hyb, np.maximum(df["smoothed_score"].astype(float) / rs.SCORE_CAP, s_proba)
        )
        results["models"]["hybrid_leave_one_session_out"]["folds"] = s_folds
    else:
        results["models"]["hybrid_leave_one_session_out"] = {
            "skipped": "only one session in the dataset - record a second person to "
                       "make leave-one-session-out meaningful"
        }

    # --- ablation: drop one modality at a time ------------------------------
    ablation = {}
    for drop in rs.ML_FEATURES:
        keep = [c for c in rs.ML_FEATURES if c != drop]
        p, pr, _ = grouped_oof(X, y, clips, keep)
        m = compute_metrics(y, p, pr)
        ablation[f"without_{drop}"] = {
            "accuracy": m["accuracy"], "f1": m["f1"], "roc_auc": m["roc_auc"]
        }
    full = results["models"]["ml_only"]
    ablation["full_feature_set"] = {
        "accuracy": full["accuracy"], "f1": full["f1"], "roc_auc": full["roc_auc"]
    }
    for k, v in ablation.items():
        if k != "full_feature_set":
            v["delta_accuracy"] = v["accuracy"] - full["accuracy"]
            v["delta_roc_auc"] = (
                (v["roc_auc"] - full["roc_auc"])
                if v["roc_auc"] is not None and full["roc_auc"] is not None else None
            )
    results["ablation"] = ablation

    # --- confusion matrices for the figure pass -----------------------------
    results["figures"] = {
        "confusion_matrix": results["models"]["hybrid"]["confusion_matrix"],
        "roc_curves": {
            k: results["models"][k].get("roc_curve")
            for k in ("rule_only", "ml_only", "ml_without_phone", "hybrid")
        },
    }
    return results


# ---------------------------------------------------------------------------
# self test - never writes into data/
# ---------------------------------------------------------------------------
def make_synthetic(n_sessions: int = 3, n_clips_per_session: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    scenarios = {0: ["S1", "S2", "S3"], 1: ["S4", "S5", "S6", "S7"]}
    for s in range(1, n_sessions + 1):
        for c in range(n_clips_per_session):
            label = int(c % 2 == 1)
            scen = scenarios[label][c % len(scenarios[label])]
            clip = f"{scen}_p{s}_r{c % 3 + 1}"
            dur = 0.0
            for f in range(12):
                cheat = label
                eye = float(np.clip(rng.normal(0.5 + (0.22 if cheat else 0.02), 0.05), 0, 1))
                tilt = float(max(0, rng.normal(16 if cheat else 6, 4)))
                body = int(cheat and rng.random() < 0.6)
                up = int(cheat and rng.random() < 0.3)
                phone = int(cheat and rng.random() < 0.45)
                hand = int(rng.random() < 0.5)
                dur = dur + 1.6 if cheat else 0.0
                rows.append(dict(
                    session_id=f"p{s}", clip_id=clip, scenario=scen, repeat=c % 3 + 1,
                    frame_idx=f, eye_ratio=eye, head_tilt=tilt, body_tilt=body,
                    looking_up=up, phone_detected=phone, hand_on_table=hand,
                    duration=round(dur, 3), label=label,
                ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dataset_labelled.csv")
    ap.add_argument("--out", default="data/results.json")
    ap.add_argument("--selftest", action="store_true",
                    help="run on synthetic data in a temp dir; proves the pipeline works "
                         "and never writes into data/")
    args = ap.parse_args()

    if args.selftest:
        with tempfile.TemporaryDirectory() as td:
            tmp_out = Path(td) / "results.json"
            df = make_synthetic()
            res = evaluate(df)
            tmp_out.write_text(json.dumps(res, indent=2))
            print("SELFTEST on synthetic data (nothing written to data/)")
            print(f"  frames={res['dataset']['n_frames']} "
                  f"clips={res['dataset']['n_clips']} "
                  f"sessions={res['dataset']['n_sessions']}")
            for k in ("rule_only", "ml_only", "ml_without_phone", "hybrid"):
                m = res["models"][k]
                auc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "n/a"
                print(f"  {k:20} acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={auc}")
            print(f"  ablation keys: {len(res['ablation'])}")
            print("  OK")
        return 0

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: {data_path} not found.", file=sys.stderr)
        print("   Record the scenario clips first - see the paper folder's "
              "RECORDING_PROTOCOL.md - then run scripts/extract_features.py.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(data_path)
    required = {"session_id", "clip_id", "frame_idx", "label", *rs.ML_FEATURES}
    missing = required - set(df.columns)
    if missing:
        print(f"error: {data_path} is missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    res = evaluate(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    print(f"  n={res['dataset']['n_frames']} frames, "
          f"{res['dataset']['n_clips']} clips, {res['dataset']['n_sessions']} sessions")
    for k in ("rule_only", "ml_only", "ml_without_phone", "hybrid",
              "hybrid_leave_one_session_out"):
        m = res["models"].get(k, {})
        if "accuracy" in m:
            auc = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a"
            print(f"  {k:32} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={auc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

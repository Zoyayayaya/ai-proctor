#!/usr/bin/env python
"""
Generate result figures (3, 4, 5) for the ICAIC-2026 revision.

All figures are emitted at 600 dpi, above the 500 dpi minimum Reviewer #4
requires, and sized to the LNCS text block (4.8 in full width).

Fixes three defects in the original figures:
  * a stray viridis gradient background and an orphan colourbar, which
    appeared in the original Figs 3, 4 AND 5
  * Fig. 3 originally disagreed with Table 6 (it reported n = 57 and 96.5%
    against the table's n = 51), and Fig. 4 reported AUC = 1.00 against the
    text's 0.98 - both are now generated from the same results.json the tables
    are generated from, so they cannot drift
  * Fig. 5 had no axis units and no marked thresholds (Reviewer #1, item 8)

Usage
-----
    uv run python scripts/make_result_figures.py --results data/results.json
    uv run python scripts/make_result_figures.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_score as rs  # noqa: E402

DPI = 600
FULL_W = 4.8
HALF_W = 3.0
OUT_DEFAULT = "figures"

INK = "#1a1a1a"
ACCENT = "#1f4e79"
SERIES = ["#1f4e79", "#c0392b", "#2e7d32", "#b58900"]
# thresholds marked on the temporal plot
THRESHOLDS = [
    (rs.SAFE_ZONE, "safe zone", "#2e7d32"),
    (50, "alert / warning", "#b58900"),
    (rs.RULE_DECISION_THRESHOLD, "rule decision", "#c0392b"),
    (80, "critical", "#7b1fa2"),
]


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=DPI, facecolor="white")
    fig.savefig(outdir / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.png at {DPI} dpi + {name}.pdf (vector)")


# ---------------------------------------------------------------- Figure 3
def fig3_confusion(cm: dict, outdir: Path):
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    m = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(HALF_W, 2.7))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.86, bottom=0.20)

    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.15)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            v = m[i, j]
            shade = v > (m.max() * 0.55)
            ax.text(j, i - 0.09, str(v), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if shade else INK)
            ax.text(j, i + 0.17, labels[i][j], ha="center", va="center",
                    fontsize=7.5, color="white" if shade else "#666666")

    ax.set_xticks([0, 1], ["Predicted\nsafe", "Predicted\ncheating"], fontsize=7)
    ax.set_yticks([0, 1], ["Actual\nsafe", "Actual\ncheating"], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # crisp cell separators instead of the original's floating grid
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    total = int(m.sum())
    acc = (tn + tp) / total if total else float("nan")
    ax.set_title(f"Confusion matrix (n = {total}, accuracy {acc*100:.2f}%)",
                 fontsize=8, color=INK, pad=7)
    _save(fig, outdir, "fig3_confusion_matrix")


# ---------------------------------------------------------------- Figure 4
def fig4_roc(curves: dict, outdir: Path, aucs: dict):
    fig, ax = plt.subplots(figsize=(HALF_W, 2.7))
    fig.subplots_adjust(left=0.155, right=0.97, top=0.93, bottom=0.165)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8,
            color="#999999", zorder=1, label="chance")

    nice = {
        "rule_only": "rule only",
        "ml_only": "ML only",
        "ml_without_phone": "ML without phone",
        "hybrid": "hybrid (Eq. 4)",
    }
    for i, key in enumerate(["rule_only", "ml_without_phone", "ml_only", "hybrid"]):
        c = curves.get(key)
        if not c:
            continue
        a = aucs.get(key)
        lab = nice[key] + (f" (AUC = {a:.3f})" if a is not None else "")
        ax.plot(c["fpr"], c["tpr"], linewidth=1.4, color=SERIES[i],
                zorder=3, label=lab)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate", fontsize=7.5)
    ax.set_ylabel("True positive rate", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6.2, loc="lower right", frameon=False)
    ax.set_title("ROC curves, grouped cross-validation", fontsize=8, color=INK, pad=6)
    _save(fig, outdir, "fig4_roc_curve")


# ---------------------------------------------------------------- Figure 5
def fig5_temporal(df: pd.DataFrame, outdir: Path, scenario: str):
    """
    Mean smoothed score over time within a suspicious scenario, with the
    decision thresholds marked. Axis units are explicit (Reviewer #1 #8).
    """
    sub = df[df["scenario"] == scenario].copy()
    if sub.empty:
        sub = df[df["label"] == 1].copy()
    if sub.empty:
        print("  skipping Fig. 5: no cheating-scenario rows found")
        return

    # smoothed_score is produced by eval.add_rule_scores, not written by
    # extract_features, so a freshly extracted dataset does not carry it.
    # Recompute rather than assume it is present.
    if "smoothed_score" not in sub.columns:
        import eval as ev
        sub = ev.add_rule_scores(sub)

    if "t_seconds" in sub.columns:
        sub["t"] = sub["t_seconds"].astype(float)
    else:
        sub["t"] = sub["frame_idx"].astype(float)

    piv = sub.pivot_table(index="t", columns="clip_id", values="smoothed_score")
    mean = piv.mean(axis=1)
    lo, hi = piv.min(axis=1), piv.max(axis=1)

    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.87, bottom=0.20)

    ax.fill_between(mean.index, lo, hi, color=ACCENT, alpha=0.13, linewidth=0,
                    label="range across repeats")
    ax.plot(mean.index, mean.values, color=ACCENT, linewidth=1.5,
            label="mean smoothed score")

    for y, lab, col in THRESHOLDS:
        ax.axhline(y, color=col, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
        ax.text(mean.index.max(), y + 1.6, lab, fontsize=5.8, color=col,
                ha="right", va="bottom")

    ax.set_xlabel("Time (s)", fontsize=7.5)
    ax.set_ylabel("Smoothed cheating score (0\u2013100)", fontsize=7.5)
    ax.set_ylim(0, 105)
    ax.set_xlim(mean.index.min(), mean.index.max())
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.22, linewidth=0.5)
    ax.legend(fontsize=6.2, loc="upper left", frameon=False)
    ax.set_title(f"Temporal progression during sustained suspicious activity "
                 f"({scenario}, {piv.shape[1]} repeats)", fontsize=7.6, color=INK, pad=6)
    _save(fig, outdir, "fig5_temporal_score")


# ---------------------------------------------------------------- selftest
def _selftest():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval as ev

    df = ev.make_synthetic()
    df = ev.add_rule_scores(df)
    res = ev.evaluate(df)
    with tempfile.TemporaryDirectory() as td:
        figs = Path(td) / "figures"
        fig3_confusion(res["models"]["hybrid"]["confusion_matrix"], figs)
        aucs = {k: res["models"][k].get("roc_auc") for k in
                ("rule_only", "ml_only", "ml_without_phone", "hybrid")}
        fig4_roc(res["figures"]["roc_curves"], figs, aucs)
        fig5_temporal(df, figs, "S6")
        from PIL import Image
        widths = {
            "fig3_confusion_matrix": HALF_W,
            "fig4_roc_curve": HALF_W,
            "fig5_temporal_score": FULL_W,
        }
        for p in sorted(figs.glob("*.png")):
            with Image.open(p) as im:
                stem = p.stem
                w_in = widths.get(stem, FULL_W)
                ok = "PASS" if im.width / w_in >= 500 else "FAIL"
                print(f"  {p.name:34} {im.width}x{im.height}px  display {w_in}in "
                      f"-> {im.width/w_in:.0f} dpi  {ok}")
    print("  OK (written to a temp dir, nothing left behind)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="data/results.json")
    ap.add_argument("--data", default="data/dataset_labelled.csv")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--scenario", default="S6",
                    help="scenario to plot for Fig. 5 (default S6, phone visible)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("SELFTEST (temp dir only)")
        _selftest()
        return 0

    rp, dp = Path(args.results), Path(args.data)
    if not rp.exists():
        print(f"error: {rp} not found - run scripts/eval.py first", file=sys.stderr)
        return 1
    res = json.loads(rp.read_text())
    outdir = Path(args.out)

    cm = res["figures"]["confusion_matrix"]
    aucs = {k: res["models"][k].get("roc_auc") for k in
            ("rule_only", "ml_only", "ml_without_phone", "hybrid")}
    fig3_confusion(cm, outdir)
    fig4_roc(res["figures"]["roc_curves"], outdir, aucs)

    if dp.exists():
        fig5_temporal(pd.read_csv(dp), outdir, args.scenario)
    else:
        print(f"  skipping Fig. 5: {dp} not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

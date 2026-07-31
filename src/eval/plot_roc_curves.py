"""One-vs-rest ROC curves per scenario for all classifiers.

For each scenario, produces a grid figure with rows = models and columns = target
languages. Each cell shows three ROC curves (Not Hate, Neutral, Hate — one-vs-rest)
with per-class AUC values in the legend, at seed 42.

Outputs:
    outputs/figures/roc_curves_<scenario>_s<seed>.png
        For scenario in {monolingual, multilingual_joint, zero_shot}.

Run: python -m src.eval.plot_roc_curves
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import FIGURES_DIR, LABELS, LANGUAGES, PREDICTIONS_DIR  # noqa: E402


ALL_MODELS = ["rf", "mbert", "xlmr", "afroxlmr", "naijaxlmt", "bertweet"]
PHASE = {"rf": "phase3", "svm": "phase3", "nb": "phase3",
         "mbert": "phase4", "xlmr": "phase4", "afroxlmr": "phase4",
         "naijaxlmt": "phase4", "bertweet": "phase4"}
CLASS_COLORS = ["#4C72B0", "#DD8452", "#C44E52"]


def pred_path(scenario, model, lang, seed):
    lang_l = lang.lower()
    phase = PHASE[model]
    if scenario == "monolingual":
        return PREDICTIONS_DIR / f"{phase}_monolingual_{lang_l}_{model}_s{seed}.csv"
    if scenario == "multilingual_joint":
        return PREDICTIONS_DIR / f"{phase}_joint_{model}_s{seed}.csv"
    if scenario == "zero_shot":
        return PREDICTIONS_DIR / f"{phase}_zeroshot_{lang_l}_{model}_s{seed}.csv"
    return None


def _extract_proba(df, lang):
    """Return (y_true, proba matrix) for the given language slice."""
    if "language" in df.columns and lang != "":
        sub = df[df["language"] == lang]
    else:
        sub = df
    proba_cols = [f"proba_{l}" for l in LABELS]
    if not all(c in sub.columns for c in proba_cols):
        return None, None
    y_true = sub["class"].values
    proba = sub[proba_cols].values.astype(float)
    return y_true, proba


def plot_grid(scenario, seed):
    fig, axes = plt.subplots(len(ALL_MODELS), len(LANGUAGES),
                             figsize=(3.4 * len(LANGUAGES), 2.8 * len(ALL_MODELS)),
                             squeeze=False)
    any_ok = False
    for i, model in enumerate(ALL_MODELS):
        for j, lang in enumerate(LANGUAGES):
            ax = axes[i, j]
            path = pred_path(scenario, model, lang, seed)
            df = None
            if path and path.exists():
                df = pd.read_csv(path)
            if df is None:
                ax.axis("off")
                continue
            y_true, proba = _extract_proba(df, lang if scenario == "multilingual_joint" else "")
            if y_true is None or len(y_true) == 0:
                ax.axis("off")
                continue
            any_ok = True
            for cls_id, lbl in enumerate(LABELS):
                y_binary = (y_true == cls_id).astype(int)
                if y_binary.sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_binary, proba[:, cls_id])
                a = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=CLASS_COLORS[cls_id],
                        label=f"{lbl} AUC={a:.2f}", linewidth=1.3)
            ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.set_title(f"{model} — {lang}", fontsize=9)
            ax.set_xlabel("FPR" if i == len(ALL_MODELS) - 1 else "")
            ax.set_ylabel("TPR" if j == 0 else "")
            ax.legend(loc="lower right", fontsize=6.5)
            ax.grid(alpha=0.3)
    if not any_ok:
        print(f"  [{scenario}] no prediction files with probabilities; skipping")
        plt.close(fig)
        return None
    plt.suptitle(f"ROC curves (one-vs-rest) — {scenario} (seed {seed})", y=1.00)
    plt.tight_layout()
    out = FIGURES_DIR / f"roc_curves_{scenario}_s{seed}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", nargs="*",
                    default=["monolingual", "multilingual_joint", "zero_shot"])
    args = ap.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for sc in args.scenarios:
        plot_grid(sc, args.seed)


if __name__ == "__main__":
    main()

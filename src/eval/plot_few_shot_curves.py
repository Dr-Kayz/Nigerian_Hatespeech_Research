"""Few-shot data-efficiency line plots per target language.

Produces one figure with three panels (Yoruba, Igbo, Hausa). Each panel has one
line per model showing macro-F1 as k (number of native target-language examples
added to English training) grows from 0 (zero-shot) through 50, 100, 500, up to
the full monolingual result.

Output:
    outputs/figures/few_shot_efficiency_curve.png

Run: python -m src.eval.plot_few_shot_curves
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import FIGURES_DIR, OUTPUTS_DIR  # noqa: E402


RESULTS_PATH = OUTPUTS_DIR / "results.jsonl"
MODELS_ORDER = ["rf", "mbert", "xlmr", "afroxlmr", "naijaxlmt", "bertweet"]
MODEL_LABELS = {
    "rf": "Random Forest",
    "mbert": "mBERT", "xlmr": "XLM-R", "afroxlmr": "Afro-XLMR",
    "naijaxlmt": "NaijaXLM-T", "bertweet": "BERTweet",
}
MODEL_COLORS = {
    "rf": "#8c564b", "mbert": "#1f77b4", "xlmr": "#ff7f0e",
    "afroxlmr": "#2ca02c", "naijaxlmt": "#d62728", "bertweet": "#9467bd",
}
MODEL_MARKERS = {
    "rf": "s", "mbert": "o", "xlmr": "^",
    "afroxlmr": "D", "naijaxlmt": "v", "bertweet": "P",
}
TARGET_LANGS = ["Yoruba", "Igbo", "Hausa"]
K_VALUES = [0, 50, 100, 500]  # 0 = zero-shot; the mono anchor is added separately


def _phase(m): return "phase3" if m in ("rf", "svm", "nb") else "phase4"


def _load_grouped():
    rows = [json.loads(l) for l in open(RESULTS_PATH) if l.strip()]
    grp = defaultdict(list)
    for r in rows:
        key = (r.get("phase"), r.get("scenario"), r.get("model"),
               r.get("train_lang"), r.get("test_lang"), r.get("split"), r.get("shots"))
        grp[key].append(r)
    return grp


def _agg(grp, key, metric="f1_macro"):
    vals = [g[metric] for g in grp.get(key, []) if g.get(metric) is not None]
    if not vals: return None, None
    return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)


def _series(grp, model, lang):
    """Return (xs, means, stds) with anchor points at k=0 (zero-shot), 50, 100, 500, and mono."""
    xs, means, stds = [], [], []
    for k in K_VALUES:
        if k == 0:
            key = (_phase(model), "zero_shot", model, "English", lang, "test", None)
        else:
            key = (_phase(model), "few_shot", model, f"English+{k}{lang}", lang, "test", k)
        mv, sv = _agg(grp, key)
        if mv is not None:
            xs.append(k)
            means.append(mv)
            stds.append(sv)
    mono_key = (_phase(model), "monolingual", model, lang, lang, "test", None)
    mv, sv = _agg(grp, mono_key)
    if mv is not None:
        xs.append(4648)  # full English train size; plotted at right
        means.append(mv)
        stds.append(sv)
    return xs, means, stds


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grp = _load_grouped()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, lang in zip(axes, TARGET_LANGS):
        for m in MODELS_ORDER:
            xs, means, stds = _series(grp, m, lang)
            if not xs: continue
            ax.errorbar(xs, means, yerr=stds, marker=MODEL_MARKERS[m],
                        color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                        linewidth=1.5, markersize=6, capsize=2)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 50, 100, 500, 4648])
        ax.set_xticklabels(["0\n(zero-shot)", "50", "100", "500", "ALL\n(mono)"])
        ax.set_xlabel("Native target-language examples added to English training (k)")
        ax.set_title(lang)
        ax.set_ylim(0.15, 0.75)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Macro-F1 (mean ± std over 3 seeds)")
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Few-shot data-efficiency curves — English + k native examples", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "few_shot_efficiency_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

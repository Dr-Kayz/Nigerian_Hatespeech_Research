"""Bar-chart figures for model comparison and per-class F1 breakdown.

Produces:
    outputs/figures/bar_comparison_<scenario>_<metric>.png
        Grouped bar chart: for each language (x-axis), one bar per model
        (colour), height = metric mean, error bar = std across seeds.
        Generated for scenarios in {monolingual, multilingual_joint, zero_shot}
        and metrics in {f1_macro, accuracy, pr_auc_macro}.

    outputs/figures/bar_per_class_f1_<scenario>.png
        Grouped bar chart: per (model, language) three bars for F1 on
        Not Hate / Neutral / Hate. Shows the class-specific balance.

Run:  python -m src.eval.plot_bar_charts
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
from src.utils.io import FIGURES_DIR, LABELS, LANGUAGES, OUTPUTS_DIR  # noqa: E402


RESULTS_PATH = OUTPUTS_DIR / "results.jsonl"
MODELS_ORDER = ["rf", "mbert", "xlmr", "afroxlmr", "naijaxlmt", "bertweet"]
MODEL_LABELS = {
    "rf": "Random Forest",
    "svm": "SVM",
    "nb": "Naive Bayes",
    "mbert": "mBERT",
    "xlmr": "XLM-R",
    "afroxlmr": "Afro-XLMR",
    "naijaxlmt": "NaijaXLM-T",
    "bertweet": "BERTweet",
}
MODEL_COLORS = {
    "rf": "#8c564b",
    "svm": "#e377c2",
    "nb": "#7f7f7f",
    "mbert": "#1f77b4",
    "xlmr": "#ff7f0e",
    "afroxlmr": "#2ca02c",
    "naijaxlmt": "#d62728",
    "bertweet": "#9467bd",
}


def _phase(m): return "phase3" if m in ("rf", "svm", "nb") else "phase4"


def _load_grouped():
    rows = [json.loads(l) for l in open(RESULTS_PATH) if l.strip()]
    grp = defaultdict(list)
    for r in rows:
        key = (r.get("phase"), r.get("scenario"), r.get("model"),
               r.get("train_lang"), r.get("test_lang"), r.get("split"), r.get("shots"))
        grp[key].append(r)
    return grp


def _agg(grp, key, metric):
    vals = [g[metric] for g in grp.get(key, []) if g.get(metric) is not None]
    if not vals: return None, None
    return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)


def plot_model_comparison(scenario, metric, grp, out_path):
    """Grouped bar chart, x = languages, groups = models, height = metric."""
    n_langs = len(LANGUAGES)
    n_models = len(MODELS_ORDER)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_langs)
    for i, m in enumerate(MODELS_ORDER):
        means, stds = [], []
        for lang in LANGUAGES:
            if scenario == "multilingual_joint":
                key = (_phase(m), scenario, m, "all", lang, "test", None)
            else:
                key = (_phase(m), scenario, m, lang, lang, "test", None)
            mv, sv = _agg(grp, key, metric)
            means.append(mv if mv is not None else 0.0)
            stds.append(sv if sv is not None else 0.0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2,
               color=MODEL_COLORS[m], edgecolor="black", linewidth=0.4,
               label=MODEL_LABELS[m])
    metric_label = {"f1_macro": "Macro-F1", "accuracy": "Accuracy",
                    "pr_auc_macro": "Macro PR-AUC",
                    "precision_macro": "Macro-Precision",
                    "recall_macro": "Macro-Recall"}.get(metric, metric)
    scenario_label = {"monolingual": "Monolingual",
                      "multilingual_joint": "Multilingual joint",
                      "zero_shot": "Zero-shot (English-trained)"}.get(scenario, scenario)
    ax.set_xticks(x)
    ax.set_xticklabels(LANGUAGES)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{scenario_label} — {metric_label} per model")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.333, color="grey", linestyle=":", linewidth=0.8,
               label="random baseline (3-class)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(scenario, grp, out_path):
    """Grouped bar chart: per (model, language), 3 bars for per-class F1."""
    class_metrics = [f"f1_{l}" for l in LABELS]
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(4.5 * len(LANGUAGES), 4.5), sharey=True)
    width = 0.24
    x = np.arange(len(MODELS_ORDER))
    for lang_idx, lang in enumerate(LANGUAGES):
        ax = axes[lang_idx]
        for cls_idx, (cls_lbl, metric_key) in enumerate(zip(LABELS, class_metrics)):
            vals = []
            for m in MODELS_ORDER:
                if scenario == "multilingual_joint":
                    key = (_phase(m), scenario, m, "all", lang, "test", None)
                else:
                    key = (_phase(m), scenario, m, lang, lang, "test", None)
                mv, _ = _agg(grp, key, metric_key)
                vals.append(mv if mv is not None else 0.0)
            offset = (cls_idx - 1) * width
            ax.bar(x + offset, vals, width,
                   color=["#4C72B0", "#DD8452", "#C44E52"][cls_idx],
                   edgecolor="black", linewidth=0.4,
                   label=cls_lbl if lang_idx == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS_ORDER],
                           rotation=45, ha="right", fontsize=9)
        ax.set_title(lang)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1 (per class)")
    scenario_label = {"monolingual": "Monolingual",
                      "multilingual_joint": "Multilingual joint",
                      "zero_shot": "Zero-shot"}.get(scenario, scenario)
    fig.suptitle(f"Per-class F1 — {scenario_label}", y=1.02)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grp = _load_grouped()

    for scenario in ["monolingual", "multilingual_joint", "zero_shot"]:
        for metric in ["f1_macro", "accuracy", "pr_auc_macro"]:
            out = FIGURES_DIR / f"bar_comparison_{scenario}_{metric}.png"
            plot_model_comparison(scenario, metric, grp, out)
            print(f"Wrote {out}")

    for scenario in ["monolingual", "multilingual_joint", "zero_shot"]:
        out = FIGURES_DIR / f"bar_per_class_f1_{scenario}.png"
        plot_per_class_f1(scenario, grp, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

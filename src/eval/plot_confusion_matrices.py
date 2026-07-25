"""Generate per-scenario confusion-matrix grid figures for all models.

For each requested scenario, produces one PNG showing a grid of confusion
matrices with rows = models and columns = target languages, using predictions
from a chosen seed (default 42).

Output filenames:
    outputs/figures/confusion_matrices_<scenario>_s<seed>.png

Run:
    python -m src.eval.plot_confusion_matrices
    python -m src.eval.plot_confusion_matrices --seed 42 --scenarios monolingual multilingual_joint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import FIGURES_DIR, LABELS, LANGUAGES, PREDICTIONS_DIR  # noqa: E402


DEFAULT_MODELS_PHASE4 = ["mbert", "xlmr", "afroxlmr", "naijaxlmt", "bertweet"]
DEFAULT_MODELS_PHASE3 = ["svm", "nb", "rf"]
ALL_MODELS = DEFAULT_MODELS_PHASE3 + DEFAULT_MODELS_PHASE4


def pred_path(scenario: str, model: str, lang: str, seed: int, shots: int | None = None) -> Path | None:
    """Return the prediction-CSV path for a given (scenario, model, lang, seed)."""
    lang_l = lang.lower()
    is_p4 = model in DEFAULT_MODELS_PHASE4
    phase = "phase4" if is_p4 else "phase3"

    if scenario == "monolingual":
        p = PREDICTIONS_DIR / f"{phase}_monolingual_{lang_l}_{model}_s{seed}.csv"
    elif scenario == "multilingual_joint":
        p = PREDICTIONS_DIR / f"{phase}_joint_{model}_s{seed}.csv"
    elif scenario == "zero_shot":
        p = PREDICTIONS_DIR / f"{phase}_zeroshot_{lang_l}_{model}_s{seed}.csv"
    elif scenario == "few_shot":
        assert shots is not None
        p = PREDICTIONS_DIR / f"{phase}_fewshot_{lang_l}_{shots}_{model}_s{seed}.csv"
    else:
        return None
    return p if p.exists() else None


def _cm_for_language(df: pd.DataFrame, lang: str) -> np.ndarray | None:
    if "language" in df.columns:
        sub = df[df["language"] == lang]
    else:
        sub = df
    if not len(sub):
        return None
    return confusion_matrix(
        sub["class"].values,
        sub["predicted_class"].values,
        labels=list(range(len(LABELS))),
    )


def plot_grid(scenario: str, models: list[str], languages: list[str], seed: int, shots: int | None = None) -> Path | None:
    grid: dict[tuple[str, str], np.ndarray] = {}
    for model in models:
        if scenario == "multilingual_joint":
            path = pred_path(scenario, model, "", seed)
            if path is None:
                continue
            df = pd.read_csv(path)
            for lang in languages:
                cm = _cm_for_language(df, lang)
                if cm is not None:
                    grid[(model, lang)] = cm
        else:
            for lang in languages:
                path = pred_path(scenario, model, lang, seed, shots=shots)
                if path is None:
                    continue
                df = pd.read_csv(path)
                cm = _cm_for_language(df, lang)
                if cm is not None:
                    grid[(model, lang)] = cm

    if not grid:
        print(f"  [{scenario}] no prediction files found for seed={seed}; skipping")
        return None

    present_models = [m for m in models if any((m, l) in grid for l in languages)]
    present_langs = [l for l in languages if any((m, l) in grid for m in present_models)]

    fig, axes = plt.subplots(
        len(present_models),
        len(present_langs),
        figsize=(4 * len(present_langs), 3.2 * len(present_models)),
        squeeze=False,
    )
    for i, model in enumerate(present_models):
        for j, lang in enumerate(present_langs):
            ax = axes[i, j]
            cm = grid.get((model, lang))
            if cm is None:
                ax.axis("off")
                continue
            row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
            cm_norm = cm.astype(float) / row_sums
            sns.heatmap(
                cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS,
                cbar=False, ax=ax, vmin=0.0, vmax=1.0,
            )
            title = f"{model} — {lang}"
            if scenario == "zero_shot":
                title = f"{model} — Eng→{lang}"
            if scenario == "few_shot" and shots is not None:
                title = f"{model} — {lang} k={shots}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Predicted" if i == len(present_models) - 1 else "")
            ax.set_ylabel("True" if j == 0 else "")

    plt.suptitle(f"Confusion matrices — scenario={scenario}, seed={seed}"
                 + (f", shots={shots}" if shots is not None else ""),
                 y=1.00)
    plt.tight_layout()
    suffix = f"_k{shots}" if shots is not None else ""
    out = FIGURES_DIR / f"confusion_matrices_{scenario}{suffix}_s{seed}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [{scenario}] -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", nargs="*",
                    default=["monolingual", "multilingual_joint", "zero_shot", "few_shot"])
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    ap.add_argument("--few-shot-k", nargs="*", type=int, default=[50, 100, 500])
    args = ap.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Plotting for seed={args.seed}, models={args.models}\n")

    for sc in args.scenarios:
        if sc == "zero_shot":
            plot_grid(sc, args.models, ["Yoruba", "Igbo", "Hausa"], args.seed)
        elif sc == "few_shot":
            for k in args.few_shot_k:
                plot_grid(sc, args.models, ["Yoruba", "Igbo", "Hausa"], args.seed, shots=k)
        elif sc == "multilingual_joint":
            plot_grid(sc, args.models, LANGUAGES, args.seed)
        elif sc == "monolingual":
            plot_grid(sc, args.models, LANGUAGES, args.seed)


if __name__ == "__main__":
    main()

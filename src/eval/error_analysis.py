"""Error analysis for the strongest joint transformer models.

Produces:
    outputs/figures/confusion_matrices_joint.png
        Grid of per-language confusion matrices for joint mBERT and joint
        Afro-XLMR (the two transformers that converged in joint training).
    outputs/tables/neutral_hate_confusion.csv
        Per (model, language) Neutral->Hate and Hate->Neutral confusion rates.
    outputs/tables/error_breakdown_by_flag.csv
        Per (model, language) test-set error rate, broken down by whether the
        row was flagged as is_artifact (high NLLB repetition) or
        is_code_switched (detected language differs from labelled language).
    outputs/predictions/error_samples_joint.csv
        A balanced sample of misclassified rows per (model, language, true_class)
        with all preprocessing flags retained for qualitative inspection.

Run:  python -m src.eval.error_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import FIGURES_DIR, LABELS, LANGUAGES, PREDICTIONS_DIR, TABLES_DIR  # noqa: E402


MODELS = ["mbert", "afroxlmr"]
ERROR_SAMPLE_PER_CLASS = 5  # misclassified examples to keep per (model, lang, true_class)


def load_joint(model: str) -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_DIR / f"phase4_joint_{model}.csv")
    df["correct"] = df["class"] == df["predicted_class"]
    return df


def per_lang_confusion(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for lang in LANGUAGES:
        sub = df[df["language"] == lang]
        out[lang] = confusion_matrix(
            sub["class"].values,
            sub["predicted_class"].values,
            labels=list(range(len(LABELS))),
        )
    return out


def plot_confusion_grid(per_model_per_lang: dict[str, dict[str, np.ndarray]], out_path: Path) -> None:
    n_rows, n_cols = len(MODELS), len(LANGUAGES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    for i, model in enumerate(MODELS):
        for j, lang in enumerate(LANGUAGES):
            cm = per_model_per_lang[model][lang]
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            ax = axes[i, j]
            sns.heatmap(
                cm_norm,
                annot=cm,
                fmt="d",
                cmap="Blues",
                xticklabels=LABELS,
                yticklabels=LABELS,
                cbar=False,
                ax=ax,
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_title(f"joint {model} — {lang}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True" if j == 0 else "")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def neutral_hate_table(per_model_per_lang: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    neutral_idx = LABELS.index("Neutral")
    hate_idx = LABELS.index("Hate")
    for model in MODELS:
        for lang in LANGUAGES:
            cm = per_model_per_lang[model][lang]
            true_neutral = cm[neutral_idx].sum()
            true_hate = cm[hate_idx].sum()
            n_to_h = cm[neutral_idx, hate_idx] / max(true_neutral, 1)
            h_to_n = cm[hate_idx, neutral_idx] / max(true_hate, 1)
            rows.append({
                "model": f"joint_{model}",
                "language": lang,
                "neutral_to_hate_rate": round(float(n_to_h), 4),
                "hate_to_neutral_rate": round(float(h_to_n), 4),
                "n_true_neutral": int(true_neutral),
                "n_true_hate": int(true_hate),
            })
    return pd.DataFrame(rows)


def error_breakdown_by_flag(df_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in df_by_model.items():
        for lang in LANGUAGES:
            sub = df[df["language"] == lang]
            n = len(sub)
            n_err = int((~sub["correct"]).sum())
            n_cs = int(sub["is_code_switched"].sum())
            n_art = int(sub["is_artifact"].sum())
            err_cs = float((~sub.loc[sub["is_code_switched"], "correct"]).mean()) if n_cs else float("nan")
            err_not_cs = float((~sub.loc[~sub["is_code_switched"], "correct"]).mean()) if (n - n_cs) else float("nan")
            err_art = float((~sub.loc[sub["is_artifact"], "correct"]).mean()) if n_art else float("nan")
            err_not_art = float((~sub.loc[~sub["is_artifact"], "correct"]).mean()) if (n - n_art) else float("nan")
            rows.append({
                "model": f"joint_{model}",
                "language": lang,
                "overall_err": round(n_err / n, 4) if n else float("nan"),
                "n_code_switched": n_cs,
                "err_code_switched": round(err_cs, 4) if err_cs == err_cs else None,
                "err_not_code_switched": round(err_not_cs, 4) if err_not_cs == err_not_cs else None,
                "n_artifact": n_art,
                "err_artifact": round(err_art, 4) if err_art == err_art else None,
                "err_not_artifact": round(err_not_art, 4) if err_not_art == err_not_art else None,
            })
    return pd.DataFrame(rows)


def collect_error_samples(df_by_model: dict[str, pd.DataFrame], per_class: int) -> pd.DataFrame:
    samples = []
    keep_cols = [
        "model", "language", "label", "predicted_label",
        "is_code_switched", "is_artifact", "detected_lang", "detected_lang_conf",
        "n_words", "source_id", "text",
    ]
    for model, df in df_by_model.items():
        df = df.copy()
        df["model"] = f"joint_{model}"
        df["predicted_label"] = df["predicted_class"].map(dict(enumerate(LABELS)))
        for lang in LANGUAGES:
            for cls_id, cls_name in enumerate(LABELS):
                wrong = df[
                    (df["language"] == lang)
                    & (df["class"] == cls_id)
                    & (df["predicted_class"] != cls_id)
                ]
                if not len(wrong):
                    continue
                # Deterministic sampling with seed for reproducibility.
                pick = wrong.sample(n=min(per_class, len(wrong)), random_state=42)
                samples.append(pick[keep_cols])
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=keep_cols)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df_by_model = {m: load_joint(m) for m in MODELS}

    per_model_per_lang = {m: per_lang_confusion(df) for m, df in df_by_model.items()}

    cm_path = FIGURES_DIR / "confusion_matrices_joint.png"
    plot_confusion_grid(per_model_per_lang, cm_path)
    print(f"Wrote {cm_path}")

    nh_table = neutral_hate_table(per_model_per_lang)
    nh_path = TABLES_DIR / "neutral_hate_confusion.csv"
    nh_table.to_csv(nh_path, index=False)
    print(f"Wrote {nh_path}")
    print(nh_table.to_string(index=False))

    eb_table = error_breakdown_by_flag(df_by_model)
    eb_path = TABLES_DIR / "error_breakdown_by_flag.csv"
    eb_table.to_csv(eb_path, index=False)
    print(f"\nWrote {eb_path}")
    print(eb_table.to_string(index=False))

    samples = collect_error_samples(df_by_model, ERROR_SAMPLE_PER_CLASS)
    sm_path = PREDICTIONS_DIR / "error_samples_joint.csv"
    samples.to_csv(sm_path, index=False)
    print(f"\nWrote {sm_path}: {len(samples)} sampled misclassifications")


if __name__ == "__main__":
    main()

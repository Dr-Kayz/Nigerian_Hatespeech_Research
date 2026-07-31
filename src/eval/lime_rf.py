"""LIME explanations for Random Forest joint predictions.

Trains the joint Random Forest on the combined training set (same as
run_phase3.py joint) at seed 42, then runs LIME on a hand-picked set of test
examples: two per class (Not Hate, Neutral, Hate), split roughly half correct
and half misclassified, sampled from the joint prediction file.

Produces:
    outputs/figures/lime_rf_<idx>_<label>.png
        One PNG per selected test tweet showing the LIME token contributions.
    outputs/tables/lime_rf_summary.csv
        A summary table listing selected tweets, their true/predicted labels,
        top positive tokens for the predicted class, and correctness flag.

Run: python -m src.eval.lime_rf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.tfidf import make_tfidf_features  # noqa: E402
from src.models.ml_baselines import build_pipeline  # noqa: E402
from src.utils.io import FIGURES_DIR, LABELS, LANGUAGES, PREDICTIONS_DIR, SPLITS_DIR, TABLES_DIR  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


SEED = 42
TEXT_COL = "text_proc"
N_SAMPLES_LIME = 500  # perturbations per LIME call


def load_split(lang, split):
    df = pd.read_csv(SPLITS_DIR / f"{lang.lower()}_{split}.csv")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def train_joint_rf(features_cfg):
    set_seed(SEED)
    train = pd.concat([load_split(l, "train") for l in LANGUAGES], ignore_index=True)
    pipe = build_pipeline("rf", make_tfidf_features(**features_cfg), seed=SEED)
    t0 = time.time()
    pipe.fit(train[TEXT_COL].values, train["class"].values)
    print(f"  trained joint RF in {time.time()-t0:.1f}s on {len(train)} rows")
    return pipe


def pick_examples(pred_df, per_class=2):
    """Return a small mix of correctly and mis-classified rows per true class."""
    picks = []
    for cls_id, lbl in enumerate(LABELS):
        correct = pred_df[(pred_df["class"] == cls_id) & (pred_df["predicted_class"] == cls_id)]
        wrong = pred_df[(pred_df["class"] == cls_id) & (pred_df["predicted_class"] != cls_id)]
        if len(correct):
            picks.append(correct.sample(n=min(per_class // 2 or 1, len(correct)), random_state=SEED))
        if len(wrong):
            picks.append(wrong.sample(n=min(per_class // 2 or 1, len(wrong)), random_state=SEED))
    return pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features_cfg = {"word_ngrams": (1, 2), "char_ngrams": (3, 5),
                    "word_max_features": 50000, "char_max_features": 50000, "min_df": 5}
    pipe = train_joint_rf(features_cfg)

    joint_pred_path = PREDICTIONS_DIR / f"phase3_joint_rf_s{SEED}.csv"
    if not joint_pred_path.exists():
        raise SystemExit(f"Missing prediction file {joint_pred_path}; run phase3 first.")
    pred = pd.read_csv(joint_pred_path)
    picks = pick_examples(pred, per_class=2)
    print(f"Selected {len(picks)} test tweets for LIME")

    explainer = LimeTextExplainer(class_names=LABELS)

    summary_rows = []
    label_map = dict(enumerate(LABELS))
    for i, row in picks.iterrows():
        text = str(row[TEXT_COL]) or str(row["text"])
        if not text.strip():
            continue
        true_lbl = label_map[row["class"]]
        pred_lbl = label_map[row["predicted_class"]]
        correct = "correct" if row["class"] == row["predicted_class"] else "wrong"
        try:
            exp = explainer.explain_instance(text, pipe.predict_proba,
                                             num_features=8, num_samples=N_SAMPLES_LIME,
                                             top_labels=1)
        except Exception as e:
            print(f"  LIME failed on idx={i}: {e}")
            continue

        fig = exp.as_pyplot_figure(label=int(row["predicted_class"]))
        fig.set_size_inches(8, 4)
        fig.suptitle(
            f"[RF joint · {row['language']}] true={true_lbl} · pred={pred_lbl} ({correct})",
            fontsize=10,
        )
        fig.tight_layout()
        out = FIGURES_DIR / f"lime_rf_{i:03d}_{true_lbl.replace(' ', '')}_{correct}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        top_tokens = ", ".join(f"{tok}({weight:+.2f})"
                               for tok, weight in exp.as_list(int(row["predicted_class"]))[:5])
        summary_rows.append({
            "figure": out.name,
            "language": row.get("language", ""),
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "correct": correct,
            "text_snippet": text[:160],
            "top_tokens_pred_class": top_tokens,
        })
        print(f"  wrote {out.name}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "lime_rf_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path} with {len(summary_df)} entries")


if __name__ == "__main__":
    main()

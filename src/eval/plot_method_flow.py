"""Method flow diagram: raw data → preprocessing → features → models → evaluation.

Produces:
    outputs/figures/method_flow.png

Run: python -m src.eval.plot_method_flow
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import FIGURES_DIR  # noqa: E402


STAGE_COLOR = "#DDEBF7"
FEATURE_COLOR = "#FFF2CC"
MODEL_COLOR = "#E2F0D9"
EVAL_COLOR = "#FBE5D6"
ARROW_COLOR = "#404040"


def _box(ax, x, y, w, h, text, color, fontsize=10):
    r = mpatches.FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.1, edgecolor="black",
                                facecolor=color)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=ARROW_COLOR))


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7)
    ax.axis("off")

    # Stage 1: data
    _box(ax, 0.1, 3.0, 1.9, 1.2,
         "Raw dataset\n26,479 rows\n(English + Yoruba +\nIgbo + Hausa)",
         STAGE_COLOR, fontsize=9)
    _arrow(ax, 2.0, 3.6, 2.6, 3.6)

    # Stage 2: cleaning
    _box(ax, 2.6, 3.0, 1.8, 1.2,
         "Clean\nUnicode NFC,\ndedup, label\nmapping",
         STAGE_COLOR, fontsize=9)
    _arrow(ax, 4.4, 3.6, 5.0, 3.6)

    # Stage 3: alignment
    _box(ax, 5.0, 3.0, 1.9, 1.2,
         "LaBSE\ntranslation\nalignment\n(source_id)",
         STAGE_COLOR, fontsize=9)
    _arrow(ax, 6.9, 3.6, 7.5, 3.6)

    # Stage 4: preprocessing
    _box(ax, 7.5, 3.0, 2.1, 1.2,
         "Preprocess\nURLs, artifacts,\ncode-switch flags",
         STAGE_COLOR, fontsize=9)
    _arrow(ax, 9.6, 3.6, 10.2, 3.6)

    # Stage 5: splits
    _box(ax, 10.2, 3.0, 1.9, 1.2,
         "Split\n70/10/20\nsource-id-\ndisjoint",
         STAGE_COLOR, fontsize=9)
    _arrow(ax, 12.1, 3.6, 12.7, 4.7)
    _arrow(ax, 12.1, 3.6, 12.7, 2.5)

    # Feature branches
    _box(ax, 12.7, 4.5, 1.3, 0.9,
         "TF-IDF\nword + char\nn-grams",
         FEATURE_COLOR, fontsize=8)
    _box(ax, 12.7, 1.9, 1.3, 0.9,
         "Tokeniser\nsub-word",
         FEATURE_COLOR, fontsize=8)

    # Model row
    _box(ax, 0.4, 5.4, 3.0, 1.0,
         "Classical models\nSVM · NB · RF",
         MODEL_COLOR, fontsize=10)
    _box(ax, 3.8, 5.4, 6.0, 1.0,
         "Transformer models\nmBERT · XLM-R · Afro-XLMR · NaijaXLM-T · BERTweet",
         MODEL_COLOR, fontsize=10)

    _arrow(ax, 12.7, 4.9, 3.5, 5.9)
    _arrow(ax, 12.7, 2.3, 6.8, 5.5)

    # Scenarios label
    _box(ax, 0.4, 0.7, 9.2, 1.0,
         "Scenarios: monolingual · multilingual joint · zero-shot · few-shot (k = 50 / 100 / 500) · translation impact\nEach scenario × model × seed (42, 123, 2024) fine-tuned separately",
         MODEL_COLOR, fontsize=9)

    _arrow(ax, 5.0, 5.4, 5.0, 1.7)

    # Evaluation
    _box(ax, 10.2, 0.7, 3.6, 1.0,
         "Evaluation\nAccuracy · Precision · Recall\nMacro-F1 · Per-class F1 · PR-AUC\nConfusion matrix",
         EVAL_COLOR, fontsize=9)
    _arrow(ax, 9.6, 1.2, 10.2, 1.2)

    plt.title("Method flow — cross-lingual hate speech detection pipeline",
              fontsize=12, pad=20)
    plt.tight_layout()
    out = FIGURES_DIR / "method_flow.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

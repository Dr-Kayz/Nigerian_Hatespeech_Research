"""Parse HuggingFace Trainer logs and plot per-epoch training + validation curves.

Reads outputs/logs/*.log and extracts the per-epoch training loss, eval loss,
and eval macro-F1 that the HuggingFace Trainer prints. Averages across seeds
and plots one curve per (model, scenario).

Outputs:
    outputs/figures/training_curves_joint.png
        Training/validation loss and eval F1 for each model under joint
        training, averaged across seeds.

Notes:
    Because logging_strategy="epoch" was used, curves have only 3 points
    (one per epoch). Enough for the qualitative shape but not smooth.

Run: python -m src.eval.plot_training_curves
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import FIGURES_DIR, LOGS_DIR  # noqa: E402


MODELS_ORDER = ["mbert", "xlmr", "afroxlmr", "naijaxlmt", "bertweet"]
MODEL_COLORS = {
    "mbert": "#1f77b4", "xlmr": "#ff7f0e", "afroxlmr": "#2ca02c",
    "naijaxlmt": "#d62728", "bertweet": "#9467bd",
}
MODEL_LABELS = {
    "mbert": "mBERT", "xlmr": "XLM-R", "afroxlmr": "Afro-XLMR",
    "naijaxlmt": "NaijaXLM-T", "bertweet": "BERTweet",
}


TRAIN_LINE = re.compile(r"\{'loss': '([\d.]+)'.*'epoch': '([\d.]+)'\}")
EVAL_LINE = re.compile(r"\{'eval_loss': '([\d.]+)'.*'eval_f1_macro': '([\d.]+)'.*'epoch': '([\d.]+)'\}")
SUMMARY_LINE = re.compile(r"\[s=(\d+)\s+(\w+)\s+\|\s+([\w\- ]+?)\s*\]\s*f1_macro")


def _parse_log(path: Path) -> list[dict]:
    """Parse a log file into a list of run records.

    Each record: {seed, model, scenario, train_losses, eval_losses, eval_f1s}
    A new run starts at every summary line (matches SUMMARY_LINE).
    """
    runs = []
    current = None
    train_buf, eval_buf = [], []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            m = TRAIN_LINE.search(line)
            if m and "eval_" not in line:
                train_buf.append((float(m.group(2)), float(m.group(1))))
                continue
            m = EVAL_LINE.search(line)
            if m:
                eval_buf.append((float(m.group(3)), float(m.group(1)), float(m.group(2))))
                continue
            m = SUMMARY_LINE.search(line)
            if m:
                seed = int(m.group(1))
                model = m.group(2)
                scenario = m.group(3).strip()
                current = {"seed": seed, "model": model, "scenario": scenario,
                           "train": list(train_buf), "eval": list(eval_buf)}
                runs.append(current)
                train_buf, eval_buf = [], []
    return runs


def _mean_curves(runs: list[dict], scenario_key: str, model: str):
    """Average per-epoch training loss / eval loss / eval f1 across seeds."""
    matched = [r for r in runs
               if r["model"] == model
               and scenario_key in r["scenario"].lower()
               and r["train"] and r["eval"]]
    if not matched:
        return None
    max_epochs = max(len(r["train"]) for r in matched)
    train_by_ep = [[] for _ in range(max_epochs)]
    evloss_by_ep = [[] for _ in range(max_epochs)]
    evf1_by_ep = [[] for _ in range(max_epochs)]
    for r in matched:
        for i, (_, tl) in enumerate(r["train"]):
            train_by_ep[i].append(tl)
        for i, (_, el, ef) in enumerate(r["eval"]):
            if i < max_epochs:
                evloss_by_ep[i].append(el)
                evf1_by_ep[i].append(ef)
    epochs = list(range(1, max_epochs + 1))
    return {
        "epochs": epochs,
        "train_loss": [mean(vs) if vs else float("nan") for vs in train_by_ep],
        "eval_loss": [mean(vs) if vs else float("nan") for vs in evloss_by_ep],
        "eval_f1": [mean(vs) if vs else float("nan") for vs in evf1_by_ep],
        "n_runs": len(matched),
    }


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    log_files = list(LOGS_DIR.glob("*.log"))
    print(f"Parsing {len(log_files)} log files ...")
    for lf in log_files:
        try:
            all_runs.extend(_parse_log(lf))
        except Exception as e:
            print(f"  skipped {lf.name}: {e}")
    print(f"Parsed {len(all_runs)} training runs.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for m in MODELS_ORDER:
        curves = _mean_curves(all_runs, "joint", m)
        if not curves:
            print(f"  no joint runs matched for {m}")
            continue
        axes[0].plot(curves["epochs"], curves["train_loss"],
                     marker="o", color=MODEL_COLORS[m],
                     label=f"{MODEL_LABELS[m]} (n={curves['n_runs']})")
        axes[1].plot(curves["epochs"], curves["eval_loss"],
                     marker="o", color=MODEL_COLORS[m],
                     label=MODEL_LABELS[m])
        axes[2].plot(curves["epochs"], curves["eval_f1"],
                     marker="o", color=MODEL_COLORS[m],
                     label=MODEL_LABELS[m])

    for ax, title, ylabel in zip(
        axes,
        ["Training loss (per epoch)", "Validation loss", "Validation macro-F1"],
        ["train loss", "eval loss", "F1_macro"],
    ):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Training & validation curves — joint multilingual training (mean across seeds)",
                 y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "training_curves_joint.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

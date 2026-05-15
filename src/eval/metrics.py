"""Evaluation metrics for the hate-speech classifier (§7 of the template).

Computes:
    * Accuracy
    * Precision, Recall, F1 (macro-averaged across the 3 classes)
    * Per-class F1 (Not Hate, Neutral, Hate)
    * Confusion matrix (3x3)

`append_result` writes one row to outputs/results.jsonl per (scenario, model,
test_lang, split). JSONL is preferred over a single CSV because each row may
add new optional fields (e.g., train_time_sec, num_train_samples) without
breaking previously-written rows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Return a flat dict of metrics; labels indexes are 0..len(labels)-1."""
    n = len(labels)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    per_class = f1_score(
        y_true, y_pred, average=None, labels=list(range(n)), zero_division=0
    )
    for lbl, f1 in zip(labels, per_class):
        out[f"f1_{lbl}"] = float(f1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    out["confusion_matrix"] = cm.tolist()
    return out


def append_result(
    results_path: Path,
    *,
    scenario: str,
    model: str,
    train_lang: str,
    test_lang: str,
    split: str,
    metrics: dict,
    extra: dict | None = None,
) -> None:
    """Append one JSON line to results.jsonl."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": scenario,
        "model": model,
        "train_lang": train_lang,
        "test_lang": test_lang,
        "split": split,
        **metrics,
    }
    if extra:
        row.update(extra)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

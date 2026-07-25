"""Evaluation metrics for the hate-speech classifier.

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

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, labels: list[str], y_proba=None) -> dict:
    """Return a flat dict of metrics; label indexes are 0..len(labels)-1.

    If y_proba is provided (shape [n_samples, n_classes]) PR-AUC values are
    added: pr_auc_<label> for each class and pr_auc_macro for the mean.
    """
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

    per_prec = precision_score(
        y_true, y_pred, average=None, labels=list(range(n)), zero_division=0
    )
    per_rec = recall_score(
        y_true, y_pred, average=None, labels=list(range(n)), zero_division=0
    )
    for lbl, p, r in zip(labels, per_prec, per_rec):
        out[f"precision_{lbl}"] = float(p)
        out[f"recall_{lbl}"] = float(r)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    out["confusion_matrix"] = cm.tolist()

    if y_proba is not None:
        y_true_arr = np.asarray(y_true)
        y_proba_arr = np.asarray(y_proba)
        pr_aucs = []
        for cls_id, lbl in enumerate(labels):
            y_binary = (y_true_arr == cls_id).astype(int)
            if y_binary.sum() == 0:
                pr_auc = float("nan")
            else:
                pr_auc = float(
                    average_precision_score(y_binary, y_proba_arr[:, cls_id])
                )
            out[f"pr_auc_{lbl}"] = pr_auc
            pr_aucs.append(pr_auc)
        finite = [v for v in pr_aucs if not np.isnan(v)]
        out["pr_auc_macro"] = float(np.mean(finite)) if finite else float("nan")

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

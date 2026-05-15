"""Classical ML baselines on TF-IDF features.

Three baselines from the template §5.1: Linear SVM, Multinomial Naive Bayes,
Random Forest. Each is wrapped in a sklearn Pipeline so that fit/predict on raw
text strings handles vectorisation + classification end-to-end.

Hyperparameter notes:
    * LinearSVC C=1.0, class_weight='balanced' — class_weight compensates for
      class imbalance (Not Hate slightly outnumbers Hate); max_iter=5000 avoids
      convergence warnings on high-dim sparse inputs.
    * MultinomialNB alpha=0.1 — light Laplace smoothing; with TF-IDF inputs,
      smaller alpha works better than the default 1.0.
    * RandomForest n_estimators=300, n_jobs=-1 — 300 trees is a reasonable
      balance of accuracy and training time; -1 uses all CPU cores.

All models receive seed=42 for reproducibility.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(model_name: str, features, seed: int = 42) -> Pipeline:
    """Compose (TF-IDF features) -> (classifier) into one sklearn Pipeline."""
    model_name = model_name.lower()
    if model_name == "svm":
        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=seed,
            max_iter=5000,
            dual="auto",
        )
    elif model_name == "nb":
        clf = MultinomialNB(alpha=0.1)
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose svm | nb | rf.")
    return Pipeline([("features", features), ("clf", clf)])

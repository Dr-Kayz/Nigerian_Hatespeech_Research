"""Driver for Phase 3: run all classical ML baseline experiments.

Reads configs/phase3_baselines.yaml and runs every (model x scenario) pair.

Outputs:
    outputs/predictions/phase3_<scenario>_<lang>_<model>.csv
        Per-row predictions on the test set for error analysis.
    outputs/results.jsonl
        One JSON line per (scenario, model, test_lang, split).

Run:
    python -m src.models.run_phase3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import append_result, compute_metrics  # noqa: E402
from src.features.tfidf import make_tfidf_features  # noqa: E402
from src.models.ml_baselines import build_pipeline  # noqa: E402
from src.utils.io import (  # noqa: E402
    CONFIGS_DIR,
    LABELS,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    SPLITS_DIR,
)
from src.utils.seed import set_seed  # noqa: E402


CONFIG_PATH = CONFIGS_DIR / "phase3_baselines.yaml"
RESULTS_PATH = OUTPUTS_DIR / "results.jsonl"
TEXT_COL = "text_proc"  # use the preprocessed text (URLs/mentions stripped)


def load_split(language: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{language.lower()}_{split}.csv")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def run_monolingual(model_name: str, language: str, features_cfg: dict, seed: int) -> None:
    train = load_split(language, "train")
    val = load_split(language, "val")
    test = load_split(language, "test")

    pipe = build_pipeline(model_name, make_tfidf_features(**features_cfg), seed=seed)
    t0 = time.time()
    pipe.fit(train[TEXT_COL].values, train["class"].values)
    train_time = time.time() - t0

    for split_name, df_eval in (("val", val), ("test", test)):
        y_pred = pipe.predict(df_eval[TEXT_COL].values)
        m = compute_metrics(df_eval["class"].values, y_pred, LABELS)
        append_result(
            RESULTS_PATH,
            scenario="monolingual",
            model=model_name,
            train_lang=language,
            test_lang=language,
            split=split_name,
            metrics=m,
            extra={"train_time_sec": train_time, "n_train": len(train)},
        )
        if split_name == "test":
            out_path = PREDICTIONS_DIR / f"phase3_monolingual_{language.lower()}_{model_name}.csv"
            df_eval.assign(predicted_class=y_pred).to_csv(out_path, index=False)
            f1 = m["f1_macro"]
            print(f"  [{model_name:>3s} | mono {language:<7s}] f1_macro(test)={f1:.4f}  train={train_time:5.1f}s")


def run_joint(model_name: str, languages: list[str], features_cfg: dict, seed: int) -> None:
    train = pd.concat([load_split(l, "train") for l in languages], ignore_index=True)
    test = pd.concat([load_split(l, "test") for l in languages], ignore_index=True)
    val = pd.concat([load_split(l, "val") for l in languages], ignore_index=True)

    pipe = build_pipeline(model_name, make_tfidf_features(**features_cfg), seed=seed)
    t0 = time.time()
    pipe.fit(train[TEXT_COL].values, train["class"].values)
    train_time = time.time() - t0

    # Val once (for tuning record), then test overall + per-language.
    y_pred_val = pipe.predict(val[TEXT_COL].values)
    m_val = compute_metrics(val["class"].values, y_pred_val, LABELS)
    append_result(
        RESULTS_PATH,
        scenario="multilingual_joint",
        model=model_name,
        train_lang="all",
        test_lang="all",
        split="val",
        metrics=m_val,
        extra={"train_time_sec": train_time, "n_train": len(train)},
    )

    y_pred = pipe.predict(test[TEXT_COL].values)
    test = test.assign(predicted_class=y_pred)
    m_overall = compute_metrics(test["class"].values, y_pred, LABELS)
    append_result(
        RESULTS_PATH,
        scenario="multilingual_joint",
        model=model_name,
        train_lang="all",
        test_lang="all",
        split="test",
        metrics=m_overall,
        extra={"train_time_sec": train_time, "n_train": len(train)},
    )
    for lang in languages:
        sub = test[test["language"] == lang]
        m_lang = compute_metrics(sub["class"].values, sub["predicted_class"].values, LABELS)
        append_result(
            RESULTS_PATH,
            scenario="multilingual_joint",
            model=model_name,
            train_lang="all",
            test_lang=lang,
            split="test",
            metrics=m_lang,
        )

    out_path = PREDICTIONS_DIR / f"phase3_joint_{model_name}.csv"
    test.to_csv(out_path, index=False)
    print(f"  [{model_name:>3s} | joint        ] f1_macro(test)={m_overall['f1_macro']:.4f}  train={train_time:5.1f}s")


def main() -> None:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    # YAML loads sequences as lists; TfidfVectorizer requires tuples for ngram_range.
    cfg["features"]["word_ngrams"] = tuple(cfg["features"]["word_ngrams"])
    cfg["features"]["char_ngrams"] = tuple(cfg["features"]["char_ngrams"])
    set_seed(cfg["seed"])

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Phase 3 driver appends to results.jsonl; if you re-run, you'll get duplicate
    # rows. Truncate any prior Phase 3 entries by filtering on scenario when reading.
    # We do NOT auto-truncate here so multiple model versions can coexist.

    for model_name in cfg["models"]:
        print(f"\n=== {model_name.upper()} ===")
        for scenario in cfg["scenarios"]:
            if scenario["name"] == "monolingual":
                for lang in scenario["languages"]:
                    run_monolingual(model_name, lang, cfg["features"], cfg["seed"])
            elif scenario["name"] == "multilingual_joint":
                run_joint(model_name, scenario["train_languages"], cfg["features"], cfg["seed"])

    print(f"\nDone. Results appended to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

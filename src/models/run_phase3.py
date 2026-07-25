"""Driver for Phase 3: classical ML baseline experiments.

Reads configs/phase3_baselines.yaml and runs every combination of
(model x scenario x seed). All models produce probability outputs so PR-AUC
is reported uniformly. Predictions and class probabilities are saved to CSV.

    python -m src.models.run_phase3                              # everything
    python -m src.models.run_phase3 --only monolingual multilingual_joint
    python -m src.models.run_phase3 --models rf --seeds 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import append_result, compute_metrics  # noqa: E402
from src.features.tfidf import make_tfidf_features  # noqa: E402
from src.models.ml_baselines import build_pipeline  # noqa: E402
from src.utils.io import CONFIGS_DIR, LABELS, OUTPUTS_DIR, PREDICTIONS_DIR, SPLITS_DIR  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


CONFIG_PATH = CONFIGS_DIR / "phase3_baselines.yaml"
RESULTS_PATH = OUTPUTS_DIR / "results.jsonl"
TEXT_COL = "text_proc"


def load_split(language: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{language.lower()}_{split}.csv")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def save_predictions(df: pd.DataFrame, y_pred: np.ndarray, y_proba: np.ndarray, out_path: Path) -> None:
    out = df.assign(predicted_class=y_pred)
    for i, lbl in enumerate(LABELS):
        out[f"proba_{lbl}"] = y_proba[:, i]
    out.to_csv(out_path, index=False)


def _record(scenario, model, train_lang, test_lang, split, y_true, y_pred, y_proba, seed, extra):
    m = compute_metrics(y_true, y_pred, LABELS, y_proba=y_proba)
    append_result(
        RESULTS_PATH,
        scenario=scenario,
        model=model,
        train_lang=train_lang,
        test_lang=test_lang,
        split=split,
        metrics=m,
        extra={"phase": "phase3", "seed": seed, **extra},
    )
    return m["f1_macro"]


def _make_pipe(model_name, features_cfg, seed):
    return build_pipeline(model_name, make_tfidf_features(**features_cfg), seed=seed)


def run_monolingual(model_name, language, features_cfg, seed):
    tr, va, te = (load_split(language, s) for s in ("train", "val", "test"))
    pipe = _make_pipe(model_name, features_cfg, seed)
    t0 = time.time()
    pipe.fit(tr[TEXT_COL].values, tr["class"].values)
    dt = time.time() - t0
    y = pipe.predict(te[TEXT_COL].values)
    p = pipe.predict_proba(te[TEXT_COL].values)
    f1 = _record(
        "monolingual", model_name, language, language, "test",
        te["class"].values, y, p, seed,
        {"train_time_sec": round(dt, 1), "n_train": len(tr)},
    )
    save_predictions(
        te, y, p,
        PREDICTIONS_DIR / f"phase3_monolingual_{language.lower()}_{model_name}_s{seed}.csv",
    )
    print(f"  [s={seed} {model_name:>3s} | mono {language:<7s}] f1_macro={f1:.4f}  train={dt:5.1f}s")


def run_joint(model_name, languages, features_cfg, seed):
    tr = pd.concat([load_split(l, "train") for l in languages], ignore_index=True)
    te = pd.concat([load_split(l, "test") for l in languages], ignore_index=True)
    pipe = _make_pipe(model_name, features_cfg, seed)
    t0 = time.time()
    pipe.fit(tr[TEXT_COL].values, tr["class"].values)
    dt = time.time() - t0
    y = pipe.predict(te[TEXT_COL].values)
    p = pipe.predict_proba(te[TEXT_COL].values)
    te = te.assign(predicted_class=y)
    for i, lbl in enumerate(LABELS):
        te[f"proba_{lbl}"] = p[:, i]

    _record(
        "multilingual_joint", model_name, "all", "all", "test",
        te["class"].values, y, p, seed,
        {"train_time_sec": round(dt, 1), "n_train": len(tr)},
    )
    for lang in languages:
        sub = te[te["language"] == lang]
        sub_proba = sub[[f"proba_{lbl}" for lbl in LABELS]].values
        _record(
            "multilingual_joint", model_name, "all", lang, "test",
            sub["class"].values, sub["predicted_class"].values, sub_proba, seed,
            {},
        )
    te.to_csv(PREDICTIONS_DIR / f"phase3_joint_{model_name}_s{seed}.csv", index=False)
    overall_f1 = compute_metrics(te["class"].values, y, LABELS, y_proba=p)["f1_macro"]
    print(f"  [s={seed} {model_name:>3s} | joint       ] f1_macro={overall_f1:.4f}  train={dt:5.1f}s")


def run_zero_shot(model_name, train_lang, test_langs, features_cfg, seed):
    tr = load_split(train_lang, "train")
    pipe = _make_pipe(model_name, features_cfg, seed)
    t0 = time.time()
    pipe.fit(tr[TEXT_COL].values, tr["class"].values)
    dt = time.time() - t0
    for tl in test_langs:
        te = load_split(tl, "test")
        y = pipe.predict(te[TEXT_COL].values)
        p = pipe.predict_proba(te[TEXT_COL].values)
        f1 = _record(
            "zero_shot", model_name, train_lang, tl, "test",
            te["class"].values, y, p, seed,
            {"train_time_sec": round(dt, 1), "n_train": len(tr)},
        )
        save_predictions(
            te, y, p,
            PREDICTIONS_DIR / f"phase3_zeroshot_{tl.lower()}_{model_name}_s{seed}.csv",
        )
        print(f"  [s={seed} {model_name:>3s} | zero-shot {train_lang}->{tl:<7s}] f1_macro={f1:.4f}")


def run_few_shot(model_name, base_lang, target_langs, shots, features_cfg, seed):
    base_tr = load_split(base_lang, "train")
    for tl in target_langs:
        tgt_tr = load_split(tl, "train")
        te = load_split(tl, "test")
        for k in shots:
            shot = tgt_tr.sample(n=min(k, len(tgt_tr)), random_state=seed)
            combined = pd.concat([base_tr, shot], ignore_index=True)
            pipe = _make_pipe(model_name, features_cfg, seed)
            t0 = time.time()
            pipe.fit(combined[TEXT_COL].values, combined["class"].values)
            dt = time.time() - t0
            y = pipe.predict(te[TEXT_COL].values)
            p = pipe.predict_proba(te[TEXT_COL].values)
            f1 = _record(
                "few_shot", model_name, f"{base_lang}+{k}{tl}", tl, "test",
                te["class"].values, y, p, seed,
                {"train_time_sec": round(dt, 1), "n_train": len(combined), "shots": k},
            )
            save_predictions(
                te, y, p,
                PREDICTIONS_DIR / f"phase3_fewshot_{tl.lower()}_{k}_{model_name}_s{seed}.csv",
            )
            print(f"  [s={seed} {model_name:>3s} | few-shot {tl:<7s} k={k:<3d}] f1_macro={f1:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario names to run (default: all in config)")
    ap.add_argument("--models", nargs="*", help="model keys to run (default: all in config)")
    ap.add_argument("--seeds", nargs="*", type=int, help="seeds to run (default: all in config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.open())
    cfg["features"]["word_ngrams"] = tuple(cfg["features"]["word_ngrams"])
    cfg["features"]["char_ngrams"] = tuple(cfg["features"]["char_ngrams"])

    models = args.models or cfg["models"]
    want = set(args.only) if args.only else None
    seeds = args.seeds or cfg["seeds"]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"models={models}  scenarios={want or 'all'}  seeds={seeds}\n")

    for seed in seeds:
        set_seed(seed)
        for model_name in models:
            print(f"\n=== seed={seed}  {model_name.upper()} ===")
            for scenario in cfg["scenarios"]:
                if want and scenario["name"] not in want:
                    continue
                if scenario["name"] == "monolingual":
                    for lang in scenario["languages"]:
                        run_monolingual(model_name, lang, cfg["features"], seed)
                elif scenario["name"] == "multilingual_joint":
                    run_joint(model_name, scenario["train_languages"], cfg["features"], seed)
                elif scenario["name"] == "zero_shot":
                    run_zero_shot(model_name, scenario["train_language"],
                                  scenario["test_languages"], cfg["features"], seed)
                elif scenario["name"] == "few_shot":
                    run_few_shot(model_name, scenario["base_train_language"],
                                 scenario["target_languages"], scenario["shots"],
                                 cfg["features"], seed)

    print(f"\nDone. Results appended to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

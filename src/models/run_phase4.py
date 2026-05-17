"""Driver for Phase 4: transformer fine-tuning experiments.

Reads configs/phase4_transformers.yaml and runs the requested (model x scenario)
combinations. Because each fine-tune takes minutes, you can run subsets:

    python -m src.models.run_phase4                         # everything
    python -m src.models.run_phase4 --only monolingual      # one scenario
    python -m src.models.run_phase4 --only zero_shot --models afroxlmr
    python -m src.models.run_phase4 --only monolingual --models mbert xlmr

Outputs:
    outputs/predictions/phase4_<scenario>_<tag>_<model>.csv
    outputs/results.jsonl   (rows tagged phase="phase4")
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import append_result, compute_metrics  # noqa: E402
from src.models.transformer_finetune import finetune, predict, pick_device  # noqa: E402
from src.utils.io import CONFIGS_DIR, LABELS, OUTPUTS_DIR, PREDICTIONS_DIR, SPLITS_DIR  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

CONFIG_PATH = CONFIGS_DIR / "phase4_transformers.yaml"
RESULTS_PATH = OUTPUTS_DIR / "results.jsonl"
CKPT_DIR = OUTPUTS_DIR / "checkpoints"
TEXT_COL = "text_proc"


def load_split(language: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{language.lower()}_{split}.csv")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def _record(scenario, model, train_lang, test_lang, split, y_true, y_pred, extra):
    m = compute_metrics(y_true, y_pred, LABELS)
    append_result(
        RESULTS_PATH,
        scenario=scenario,
        model=model,
        train_lang=train_lang,
        test_lang=test_lang,
        split=split,
        metrics=m,
        extra={"phase": "phase4", **extra},
    )
    return m["f1_macro"]


def run_monolingual(model, lang, tcfg, seed):
    tr, va, te = (load_split(lang, s) for s in ("train", "val", "test"))
    t0 = time.time()
    trainer, tok = finetune(model, tr, va, CKPT_DIR / f"_p4_mono_{lang}_{model}", **tcfg, seed=seed)
    dt = time.time() - t0
    y = predict(trainer, tok, te, max_length=tcfg["max_length"])
    f1 = _record("monolingual", model, lang, lang, "test", te["class"].values, y,
                 {"train_time_sec": round(dt, 1), "n_train": len(tr)})
    te.assign(predicted_class=y).to_csv(
        PREDICTIONS_DIR / f"phase4_monolingual_{lang.lower()}_{model}.csv", index=False)
    print(f"  [{model} | mono {lang:<7s}] f1_macro={f1:.4f}  train={dt/60:.1f}min")


def run_joint(model, langs, tcfg, seed):
    tr = pd.concat([load_split(l, "train") for l in langs], ignore_index=True)
    va = pd.concat([load_split(l, "val") for l in langs], ignore_index=True)
    te = pd.concat([load_split(l, "test") for l in langs], ignore_index=True)
    t0 = time.time()
    trainer, tok = finetune(model, tr, va, CKPT_DIR / f"_p4_joint_{model}", **tcfg, seed=seed)
    dt = time.time() - t0
    y = predict(trainer, tok, te, max_length=tcfg["max_length"])
    te = te.assign(predicted_class=y)
    f1 = _record("multilingual_joint", model, "all", "all", "test",
                 te["class"].values, y, {"train_time_sec": round(dt, 1), "n_train": len(tr)})
    for lang in langs:
        sub = te[te["language"] == lang]
        _record("multilingual_joint", model, "all", lang, "test",
                sub["class"].values, sub["predicted_class"].values, {})
    te.to_csv(PREDICTIONS_DIR / f"phase4_joint_{model}.csv", index=False)
    print(f"  [{model} | joint] f1_macro(overall)={f1:.4f}  train={dt/60:.1f}min")


def run_zero_shot(model, train_lang, test_langs, tcfg, seed):
    tr, va = load_split(train_lang, "train"), load_split(train_lang, "val")
    t0 = time.time()
    trainer, tok = finetune(model, tr, va, CKPT_DIR / f"_p4_zs_{model}", **tcfg, seed=seed)
    dt = time.time() - t0
    for tl in test_langs:
        te = load_split(tl, "test")
        y = predict(trainer, tok, te, max_length=tcfg["max_length"])
        f1 = _record("zero_shot", model, train_lang, tl, "test",
                     te["class"].values, y, {"train_time_sec": round(dt, 1), "n_train": len(tr)})
        te.assign(predicted_class=y).to_csv(
            PREDICTIONS_DIR / f"phase4_zeroshot_{tl.lower()}_{model}.csv", index=False)
        print(f"  [{model} | zero-shot {train_lang}->{tl:<7s}] f1_macro={f1:.4f}")


def run_few_shot(model, base_lang, target_langs, shots, tcfg, seed):
    base_tr = load_split(base_lang, "train")
    base_va = load_split(base_lang, "val")
    for tl in target_langs:
        tgt_tr_full = load_split(tl, "train")
        te = load_split(tl, "test")
        for k in shots:
            shot = tgt_tr_full.sample(n=min(k, len(tgt_tr_full)), random_state=seed)
            tr = pd.concat([base_tr, shot], ignore_index=True)
            t0 = time.time()
            trainer, tok = finetune(model, tr, base_va,
                                    CKPT_DIR / f"_p4_fs_{tl}_{k}_{model}", **tcfg, seed=seed)
            dt = time.time() - t0
            y = predict(trainer, tok, te, max_length=tcfg["max_length"])
            f1 = _record("few_shot", model, f"{base_lang}+{k}{tl}", tl, "test",
                         te["class"].values, y,
                         {"train_time_sec": round(dt, 1), "n_train": len(tr), "shots": k})
            te.assign(predicted_class=y).to_csv(
                PREDICTIONS_DIR / f"phase4_fewshot_{tl.lower()}_{k}_{model}.csv", index=False)
            print(f"  [{model} | few-shot {tl} k={k:<3d}] f1_macro={f1:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="scenario names to run (default: all in config)")
    ap.add_argument("--models", nargs="*", help="model keys to run (default: all in config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.open())
    set_seed(cfg["seed"])
    tcfg = cfg["train"]
    models = args.models or cfg["models"]
    want = set(args.only) if args.only else None

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={pick_device()}  models={models}  scenarios={want or 'all'}\n")

    for model in models:
        print(f"=== {model} ===")
        for sc in cfg["scenarios"]:
            if want and sc["name"] not in want:
                continue
            if sc["name"] == "monolingual":
                for lang in sc["languages"]:
                    run_monolingual(model, lang, tcfg, cfg["seed"])
            elif sc["name"] == "multilingual_joint":
                run_joint(model, sc["train_languages"], tcfg, cfg["seed"])
            elif sc["name"] == "zero_shot":
                run_zero_shot(model, sc["train_language"], sc["test_languages"], tcfg, cfg["seed"])
            elif sc["name"] == "few_shot":
                run_few_shot(model, sc["base_train_language"], sc["target_languages"],
                             sc["shots"], tcfg, cfg["seed"])

    print(f"\nDone. Results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()

"""Re-train XLM-R on the African languages with extended schedule and warmup.

The default monolingual configuration (3 epochs, no warmup) caused XLM-R to
underfit on Yoruba/Igbo/Hausa: training loss decreased by only ~0.12 from the
random-guessing baseline of ln(3), versus ~0.55 on English. This script
re-trains XLM-R on those three languages with 5 epochs and a 10% warmup ratio
and appends result rows tagged variant='warmup_5ep'. The original result rows
are preserved alongside the new ones for transparent reporting.

Run:  python -m src.models.run_xlmr_warmup_diagnostic
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import append_result, compute_metrics  # noqa: E402
from src.models.transformer_finetune import finetune, predict  # noqa: E402
from src.utils.io import LABELS, OUTPUTS_DIR, PREDICTIONS_DIR, SPLITS_DIR  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

RESULTS_PATH = OUTPUTS_DIR / "results.jsonl"
CKPT_DIR = OUTPUTS_DIR / "checkpoints"

LANGUAGES = ["Yoruba", "Igbo", "Hausa"]
EPOCHS = 5
WARMUP_RATIO = 0.10
BATCH_SIZE = 16
MAX_LENGTH = 128
LR = 2e-5
SEED = 42
TEXT_COL = "text_proc"


def load_split(language: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{language.lower()}_{split}.csv")
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def main() -> None:
    set_seed(SEED)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"XLM-R diagnostic re-run | epochs={EPOCHS} warmup_ratio={WARMUP_RATIO} lr={LR}\n"
    )
    for lang in LANGUAGES:
        train = load_split(lang, "train")
        val = load_split(lang, "val")
        test = load_split(lang, "test")

        t0 = time.time()
        trainer, tokenizer = finetune(
            "xlmr",
            train,
            val,
            CKPT_DIR / f"_xlmr_warmup_{lang}",
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            max_length=MAX_LENGTH,
            warmup_ratio=WARMUP_RATIO,
            seed=SEED,
        )
        dt = time.time() - t0

        y_pred = predict(trainer, tokenizer, test, max_length=MAX_LENGTH)
        m = compute_metrics(test["class"].values, y_pred, LABELS)
        append_result(
            RESULTS_PATH,
            scenario="monolingual",
            model="xlmr",
            train_lang=lang,
            test_lang=lang,
            split="test",
            metrics=m,
            extra={
                "phase": "phase4",
                "variant": "warmup_5ep",
                "epochs": EPOCHS,
                "warmup_ratio": WARMUP_RATIO,
                "train_time_sec": round(dt, 1),
                "n_train": len(train),
            },
        )
        out_path = (
            PREDICTIONS_DIR
            / f"phase4_monolingual_{lang.lower()}_xlmr_warmup.csv"
        )
        test.assign(predicted_class=y_pred).to_csv(out_path, index=False)
        print(
            f"  [xlmr | mono {lang:<7s}] f1_macro={m['f1_macro']:.4f}  train={dt/60:.1f}min"
        )

    print(f"\nDone. Results appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

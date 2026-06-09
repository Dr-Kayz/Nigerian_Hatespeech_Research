"""Fine-tune a pretrained multilingual transformer for 3-class hate speech.

Wraps the Hugging Face Trainer so a model can be fine-tuned with one function
call. Model keys map to Hub identifiers in MODEL_REGISTRY:

    mbert    -> bert-base-multilingual-cased
    xlmr     -> xlm-roberta-base
    afroxlmr -> Davlan/afro-xlmr-base

Runs on Apple MPS / CUDA / CPU (auto-detected). Uses fp32 (MPS fp16 is
unreliable). Input column is `text_proc`; label column is `class` (0/1/2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import compute_metrics as _compute_metrics  # noqa: E402
from src.utils.io import LABELS  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


MODEL_REGISTRY = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "afroxlmr": "Davlan/afro-xlmr-base",
}

TEXT_COL = "text_proc"
LABEL_COL = "class"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_hf_dataset(df, tokenizer, max_length: int) -> Dataset:
    sub = (
        df[[TEXT_COL, LABEL_COL]]
        .rename(columns={LABEL_COL: "labels"})
        .copy()
    )
    sub[TEXT_COL] = sub[TEXT_COL].fillna("").astype(str)
    ds = Dataset.from_pandas(sub.reset_index(drop=True))

    def tok(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)

    return ds.map(tok, batched=True, remove_columns=[TEXT_COL])


def _hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = _compute_metrics(labels, preds, LABELS)
    return {k: v for k, v in m.items() if k != "confusion_matrix"}


def finetune(
    model_key: str,
    train_df,
    val_df,
    output_dir: Path,
    *,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    warmup_ratio: float = 0.0,
    seed: int = 42,
):
    """Fine-tune `model_key` on train_df, evaluating on val_df each epoch."""
    set_seed(seed)
    model_name = MODEL_REGISTRY[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)
    )

    train_ds = _to_hf_dataset(train_df, tokenizer, max_length)
    val_ds = _to_hf_dataset(val_df, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        seed=seed,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=_hf_metrics,
    )
    trainer.train()
    return trainer, tokenizer


def predict(trainer, tokenizer, df, *, max_length: int = 128):
    """Return integer class predictions for df[TEXT_COL]."""
    ds = _to_hf_dataset(df, tokenizer, max_length)
    out = trainer.predict(ds)
    return np.argmax(out.predictions, axis=-1)

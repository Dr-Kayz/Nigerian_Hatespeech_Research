"""Recover translation alignment in the multilingual hate-speech dataset.

The Kaggle data card states that every tweet in the dataset appears four times
(once per language: English, Yoruba, Igbo, Hausa) with the same label. The raw
CSV rows are shuffled, so this script recovers the alignment by content-matching
LaBSE sentence embeddings.

Approach:
    1. Encode all rows with LaBSE (multilingual sentence embeddings on MPS).
    2. For each non-English language and each class, solve a 1-1 assignment
       problem (Hungarian) between English rows and target-language rows of the
       same class. Class constraint is justified because the data card
       guarantees matched rows share a class.
    3. Each English row's position becomes the canonical `source_id`. Target-
       language rows inherit the `source_id` of their matched English row.

Output: data/aligned_multilingual_hate_speech_dataset.csv with an added
`source_id` column linking the four versions of each tweet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import CLEAN_CSV, DATA_DIR  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


MODEL_NAME = "sentence-transformers/LaBSE"
OUTPUT_CSV = DATA_DIR / "aligned_multilingual_hate_speech_dataset.csv"
LANGUAGES = ["English", "Yoruba", "Igbo", "Hausa"]
NON_ENGLISH = ["Yoruba", "Igbo", "Hausa"]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def encode_all(df: pd.DataFrame, model: SentenceTransformer) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for lang in LANGUAGES:
        texts = df.loc[df["language"] == lang, "text"].tolist()
        print(f"  Encoding {lang}: {len(texts)} rows ...", flush=True)
        t0 = time()
        emb = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"    done in {time()-t0:.1f}s; shape={emb.shape}", flush=True)
        out[lang] = emb.astype(np.float32)
    return out


def align_one_target(
    eng_sub: pd.DataFrame,
    tgt_sub: pd.DataFrame,
    eng_emb: np.ndarray,
    tgt_emb: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Return per-target-row source_id (-1 if unmatched) and similarity per match."""
    source_ids = np.full(len(tgt_sub), -1, dtype=int)
    per_class_mean_sim: list[float] = []

    for cls in [0, 1, 2]:
        eng_idx = np.where(eng_sub["class"].values == cls)[0]
        tgt_idx = np.where(tgt_sub["class"].values == cls)[0]
        if len(eng_idx) == 0 or len(tgt_idx) == 0:
            continue

        sim = eng_emb[eng_idx] @ tgt_emb[tgt_idx].T  # cosine (L2-normed inputs)
        cost = -sim
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_sims = []
        for r, c in zip(row_ind, col_ind):
            source_ids[tgt_idx[c]] = eng_idx[r]
            matched_sims.append(float(sim[r, c]))
        per_class_mean_sim.append(float(np.mean(matched_sims)))
        print(
            f"    class {cls}: matched {len(matched_sims)} pairs, mean cosine={np.mean(matched_sims):.3f}",
            flush=True,
        )

    return source_ids, per_class_mean_sim


def main() -> None:
    set_seed(42)

    print(f"Loading cleaned dataset from {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV)
    print(f"  shape={df.shape}")
    print(f"  per-language counts: {df.groupby('language').size().to_dict()}")

    df = df.reset_index(drop=True)
    df["_lang_pos"] = df.groupby("language").cumcount()

    device = pick_device()
    print(f"\nLoading {MODEL_NAME} on device={device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("\nEncoding all texts ...")
    emb_by_lang = encode_all(df, model)

    eng_sub = df[df["language"] == "English"].reset_index(drop=True)
    eng_emb = emb_by_lang["English"]

    source_ids_full = pd.Series(-1, index=df.index, dtype=int)
    source_ids_full.loc[df["language"] == "English"] = np.arange(len(eng_sub))

    for tgt_lang in NON_ENGLISH:
        print(f"\nAligning {tgt_lang} -> English (class-constrained Hungarian)")
        tgt_sub = df[df["language"] == tgt_lang].reset_index(drop=True)
        tgt_emb = emb_by_lang[tgt_lang]
        ids, _ = align_one_target(eng_sub, tgt_sub, eng_emb, tgt_emb)
        n_matched = int((ids >= 0).sum())
        print(f"  total matched: {n_matched}/{len(tgt_sub)}")
        source_ids_full.loc[df["language"] == tgt_lang] = ids

    df["source_id"] = source_ids_full.values
    df = df.drop(columns=["_lang_pos"])

    quad_counts = df.groupby("source_id")["language"].nunique()
    full_quads = int((quad_counts == 4).sum())
    print(
        f"\nSource-id groups with all 4 languages present: {full_quads} / {(df['source_id'] >= 0).sum() // 4}"
    )

    print(f"\nSaving aligned dataset to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  shape={df.shape}, columns={list(df.columns)}")


if __name__ == "__main__":
    main()

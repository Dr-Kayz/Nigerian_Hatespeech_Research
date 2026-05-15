"""Generate train/val/test splits — source_id-disjoint, stratified by class.

Design:
- We split at the source_id level (1 source_id = 1 tweet + its 3 translations).
- A source_id assigned to train means all 4 language rows for that tweet are in train.
- This guarantees: (a) no leakage between train/val/test, (b) the same test source_ids
  appear across all 4 languages — required for the §6.5 same-tweet translation impact
  study, (c) zero-shot transfer evaluation is fair (test tweets never seen in any lang).

Split sizes: 70% train, 10% val, 20% test, stratified by class.

Orphans: rows whose source_id is -1 (no alignment recovered) are dropped from splits.
This affects ~0-50 rows total based on Phase-1 alignment results.

Outputs:
    data/splits/{language}_{split}.csv     for language in {english,yoruba,igbo,hausa}
                                           and split in {train,val,test}
    data/splits/manifest.json              full manifest with seed + sizes + label distros
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import DATA_DIR, LANGUAGES, SPLITS_DIR  # noqa: E402
from src.utils.seed import SEED  # noqa: E402


PROCESSED_CSV = DATA_DIR / "processed_aligned_dataset.csv"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    print(f"Loaded {len(df)} rows from {PROCESSED_CSV}")

    df = df[df["source_id"] >= 0].copy()
    print(f"After dropping orphans: {len(df)} rows")

    english = df[df["language"] == "English"].drop_duplicates("source_id")
    sids = english[["source_id", "class"]].reset_index(drop=True)
    print(f"Unique source_ids: {len(sids)}")
    print(f"Class distribution at source_id level:")
    print(sids["class"].value_counts().sort_index())

    train_ids, hold_ids = train_test_split(
        sids,
        test_size=VAL_FRAC + TEST_FRAC,
        stratify=sids["class"],
        random_state=SEED,
    )
    val_ids, test_ids = train_test_split(
        hold_ids,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=hold_ids["class"],
        random_state=SEED,
    )

    split_map: dict[int, str] = {}
    for sid in train_ids["source_id"]:
        split_map[int(sid)] = "train"
    for sid in val_ids["source_id"]:
        split_map[int(sid)] = "val"
    for sid in test_ids["source_id"]:
        split_map[int(sid)] = "test"

    df["split"] = df["source_id"].map(split_map)
    df = df.dropna(subset=["split"]).copy()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "seed": SEED,
        "method": "source_id_disjoint_stratified",
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "n_source_ids": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "sizes": {},
        "label_counts": {},
    }

    print("\nWriting per-language splits:")
    for lang in LANGUAGES:
        for split in ("train", "val", "test"):
            sub = df[(df["language"] == lang) & (df["split"] == split)]
            out = SPLITS_DIR / f"{lang.lower()}_{split}.csv"
            sub.to_csv(out, index=False)
            key = f"{lang}/{split}"
            manifest["sizes"][key] = len(sub)
            manifest["label_counts"][key] = sub["label"].value_counts().to_dict()
            dist = sub["label"].value_counts().to_dict()
            print(f"  {key:18s} {len(sub):5d} rows   {dist}")

    with (SPLITS_DIR / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest at {SPLITS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()

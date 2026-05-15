"""Stage 1 of the data pipeline: clean the raw multilingual hate-speech CSV.

What this does (in order):
    1. Normalise text Unicode (NFC), strip hidden/control characters.
    2. Drop rows with missing or empty text.
    3. Standardise the `language` column to Title-case.
    4. Map the numeric `class` (0/1/2) to readable `label` ("Not Hate"/"Neutral"/"Hate")
       per the Kaggle "Nigerian Multilingual Hate Speech" data card.
    5. Drop any rows whose language or class didn't map.
    6. Drop exact-duplicate rows.
    7. Save to data/cleaned_multilingual_hate_speech_dataset.csv.

Run:  python -m src.data.clean
Reads:  data/multilingual_hate_speech_dataset.csv
Writes: data/cleaned_multilingual_hate_speech_dataset.csv
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import CLEAN_CSV, ID_TO_LABEL, RAW_CSV  # noqa: E402


LANGUAGE_NORMALISE = {
    "english": "English",
    "yoruba": "Yoruba",
    "igbo": "Igbo",
    "hausa": "Hausa",
}

# Hidden/control chars: zero-width spaces, direction marks, BOM, etc.
HIDDEN_CHARS_RE = re.compile(r"[​-‏‪-‮⁠-⁯﻿]")
MULTI_WHITESPACE_RE = re.compile(r"\s+")


def clean_hidden_unicode(text: str) -> str:
    """Normalise Unicode and strip hidden/control chars; preserve diacritics + emojis."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = HIDDEN_CHARS_RE.sub("", text)
    text = text.replace(" ", " ")  # non-breaking space → normal space
    text = MULTI_WHITESPACE_RE.sub(" ", text).strip()
    return text


def main() -> None:
    print(f"Reading: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"  raw shape: {df.shape}")
    print(f"  raw missing text: {df['text'].isna().sum()}")
    print(f"  raw exact duplicates: {df.duplicated().sum()}")

    # 1. Drop missing text.
    df = df.dropna(subset=["text"]).copy()

    # 2. Clean Unicode and drop empty rows after cleaning.
    df["text"] = df["text"].apply(clean_hidden_unicode)
    df = df[df["text"].astype(str).str.strip() != ""].copy()

    # 3. Normalise language.
    df["language"] = df["language"].astype(str).str.strip().str.lower().map(LANGUAGE_NORMALISE)

    # 4. Map class → label (matches src/utils/io.py constants).
    df["label"] = df["class"].map(ID_TO_LABEL)

    # 5. Drop rows where language or label didn't map.
    before = len(df)
    df = df.dropna(subset=["language", "label"]).copy()
    print(f"  dropped {before - len(df)} rows with unmapped language/label")

    # 6. Deduplicate.
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"  dropped {before - len(df)} exact-duplicate rows")

    # 7. Final column order.
    df = df[["text", "language", "class", "label"]]

    print("\nFinal language × label distribution:")
    print(pd.crosstab(df["language"], df["label"]))
    print(f"\nFinal shape: {df.shape}")

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"\nSaved: {CLEAN_CSV}")


if __name__ == "__main__":
    main()

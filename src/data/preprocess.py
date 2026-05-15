"""Phase 2 preprocessing — extends the aligned dataset.

Adds these columns:
    text_proc        : URLs/mentions/RT/LINK removed, whitespace collapsed
    n_chars, n_words : length features for EDA
    rep_unigram      : longest run of identical consecutive tokens / total tokens
    rep_trigram      : fraction of trigrams that repeat at least once
    is_artifact      : True if rep_unigram > 0.25 OR rep_trigram > 0.40
    detected_lang_raw  : raw GlotLID prediction (ISO-639-3_Script form)
    detected_lang_conf : top-1 GlotLID confidence
    detected_lang      : mapped to {English, Yoruba, Igbo, Hausa, Other}
    is_code_switched   : True when detected_lang != labelled language

Source : data/aligned_multilingual_hate_speech_dataset.csv
Output : data/processed_aligned_dataset.csv
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import fasttext
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import DATA_DIR  # noqa: E402


URL_RE = re.compile(r"https?://\S+|www\.\S+")
LINK_TOKEN_RE = re.compile(r"\bLINK\b", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
RT_PREFIX_RE = re.compile(r"^[\s:]*RT\s+", flags=re.IGNORECASE)
LEAD_PUNCT_RE = re.compile(r'^[:;,.\-"\'\s]+')
MULTI_SPACE_RE = re.compile(r"\s+")

GLOTLID_TO_SHORT = {
    "eng_Latn": "English",
    "yor_Latn": "Yoruba",
    "ibo_Latn": "Igbo",
    "hau_Latn": "Hausa",
}

REP_UNIGRAM_THRESHOLD = 0.25
REP_TRIGRAM_THRESHOLD = 0.40


def clean_text(s: str) -> str:
    s = URL_RE.sub("", s)
    s = LINK_TOKEN_RE.sub("", s)
    s = MENTION_RE.sub("", s)
    s = RT_PREFIX_RE.sub("", s)
    s = LEAD_PUNCT_RE.sub("", s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def repetition_scores(s: str) -> tuple[float, float]:
    toks = s.split()
    n = len(toks)
    if n < 4:
        return 0.0, 0.0

    longest = run = 1
    for i in range(1, n):
        if toks[i] == toks[i - 1]:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    uni = max(0.0, (longest - 1) / n)

    if n < 6:
        return uni, 0.0
    trigrams = [tuple(toks[i : i + 3]) for i in range(n - 2)]
    counts = Counter(trigrams)
    repeated = sum(c for c in counts.values() if c > 1)
    tri = repeated / len(trigrams)
    return uni, tri


def load_glotlid() -> fasttext.FastText._FastText:
    model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model_v3.bin")
    return fasttext.load_model(model_path)


def detect_lang(model: fasttext.FastText._FastText, text: str) -> tuple[str, float]:
    text = text.replace("\n", " ").strip()
    if not text:
        return "unknown", 0.0
    labels, probs = model.predict(text, k=1)
    return labels[0].replace("__label__", ""), float(probs[0])


def main() -> None:
    src = DATA_DIR / "aligned_multilingual_hate_speech_dataset.csv"
    out = DATA_DIR / "processed_aligned_dataset.csv"

    df = pd.read_csv(src)
    print(f"Loaded {len(df)} rows from {src}")

    print("Cleaning text ...")
    df["text_proc"] = df["text"].astype(str).map(clean_text)
    df["n_chars"] = df["text_proc"].str.len()
    df["n_words"] = df["text_proc"].str.split().map(len)

    print("Computing repetition scores ...")
    scores = df["text_proc"].map(repetition_scores)
    df["rep_unigram"] = scores.map(lambda x: x[0])
    df["rep_trigram"] = scores.map(lambda x: x[1])
    df["is_artifact"] = (df["rep_unigram"] > REP_UNIGRAM_THRESHOLD) | (
        df["rep_trigram"] > REP_TRIGRAM_THRESHOLD
    )

    print("Loading GlotLID v3 ...")
    model = load_glotlid()

    print("Running language detection ...")
    raws, confs = [], []
    for text in tqdm(df["text_proc"].tolist(), unit="row"):
        lang, conf = detect_lang(model, text)
        raws.append(lang)
        confs.append(conf)
    df["detected_lang_raw"] = raws
    df["detected_lang_conf"] = confs
    df["detected_lang"] = df["detected_lang_raw"].map(GLOTLID_TO_SHORT).fillna("Other")
    df["is_code_switched"] = df["detected_lang"] != df["language"]

    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")

    print(f"\nArtifact-flagged   : {df['is_artifact'].sum():5d}  ({df['is_artifact'].mean():.2%})")
    print(f"Code-switch-flagged: {df['is_code_switched'].sum():5d}  ({df['is_code_switched'].mean():.2%})")
    print("\nCode-switching rate per labelled language:")
    print(df.groupby("language")["is_code_switched"].agg(["sum", "mean", "count"]).round(3))
    print("\nDetected vs labelled language (crosstab):")
    print(pd.crosstab(df["language"], df["detected_lang"]))


if __name__ == "__main__":
    main()

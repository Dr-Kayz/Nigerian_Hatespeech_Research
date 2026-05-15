"""Exploratory data analysis — produces tables and figures for §2 of the report.

Reads : data/processed_aligned_dataset.csv
Writes:
    outputs/tables/  composition.csv, class_per_language.csv, lengths_per_language.csv,
                     code_switching_per_language.csv, artifact_per_language.csv,
                     top_tokens_per_language.csv
    outputs/figures/ class_balance.png, length_distributions.png, code_switching.png,
                     artifacts.png, confidence_distribution.png
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.io import DATA_DIR, FIGURES_DIR, LANGUAGES, TABLES_DIR  # noqa: E402


PROCESSED_CSV = DATA_DIR / "processed_aligned_dataset.csv"

sns.set_theme(style="whitegrid", context="notebook")


def composition_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pd.crosstab(df["language"], df["label"], margins=True, margins_name="Total")
    return pivot[["Hate", "Neutral", "Not Hate", "Total"]]


def length_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("language")[["n_chars", "n_words"]]
        .agg(["mean", "median", "max"])
        .round(1)
    )


def code_switch_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("language")["is_code_switched"]
    return pd.DataFrame(
        {"n_flagged": g.sum(), "total": g.count(), "rate": g.mean().round(4)}
    )


def artifact_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("language")["is_artifact"]
    return pd.DataFrame(
        {"n_flagged": g.sum(), "total": g.count(), "rate": g.mean().round(4)}
    )


def top_tokens(df: pd.DataFrame, lang: str, top_k: int = 20) -> pd.DataFrame:
    rows = []
    for label in ("Hate", "Neutral", "Not Hate"):
        toks: list[str] = []
        for t in df[(df["language"] == lang) & (df["label"] == label)]["text_proc"]:
            toks.extend(str(t).lower().split())
        toks = [t for t in toks if len(t) > 2]
        common = Counter(toks).most_common(top_k)
        for tok, ct in common:
            rows.append({"language": lang, "label": label, "token": tok, "count": ct})
    return pd.DataFrame(rows)


def fig_class_balance(df: pd.DataFrame, path: Path) -> None:
    ct = pd.crosstab(df["language"], df["label"])[["Not Hate", "Neutral", "Hate"]]
    ax = ct.plot(kind="bar", stacked=False, figsize=(8, 5), edgecolor="black")
    ax.set_title("Class distribution per language")
    ax.set_ylabel("Number of tweets")
    ax.set_xlabel("Language")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_length_dist(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=df, x="language", y="n_chars", ax=axes[0], order=LANGUAGES)
    axes[0].set_title("Character length per language")
    axes[0].set_ylim(0, df["n_chars"].quantile(0.99))
    sns.boxplot(data=df, x="language", y="n_words", ax=axes[1], order=LANGUAGES)
    axes[1].set_title("Word count per language")
    axes[1].set_ylim(0, df["n_words"].quantile(0.99))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_code_switch(df: pd.DataFrame, path: Path) -> None:
    rates = df.groupby("language")["is_code_switched"].mean().reindex(LANGUAGES)
    ax = rates.plot(kind="bar", figsize=(7, 4), color="#4477AA", edgecolor="black")
    ax.set_title("Code-switching / LID-mismatch rate per language")
    ax.set_ylabel("Fraction flagged")
    plt.xticks(rotation=0)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_artifact(df: pd.DataFrame, path: Path) -> None:
    rates = df.groupby("language")["is_artifact"].mean().reindex(LANGUAGES)
    ax = rates.plot(kind="bar", figsize=(7, 4), color="#CC6677", edgecolor="black")
    ax.set_title("NLLB translation-artifact rate per language")
    ax.set_ylabel("Fraction flagged")
    plt.xticks(rotation=0)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.001, f"{v:.2%}", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def fig_confidence(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for lang in LANGUAGES:
        sns.kdeplot(
            df[df["language"] == lang]["detected_lang_conf"],
            label=lang,
            ax=ax,
            clip=(0, 1),
        )
    ax.set_title("GlotLID top-1 confidence by labelled language")
    ax.set_xlabel("Detected-language confidence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    df = pd.read_csv(PROCESSED_CSV)
    print(f"Loaded {len(df)} rows")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Composition ===")
    comp = composition_table(df)
    print(comp)
    comp.to_csv(TABLES_DIR / "composition.csv")

    print("\n=== Length stats ===")
    lengths = length_table(df)
    print(lengths)
    lengths.to_csv(TABLES_DIR / "lengths_per_language.csv")

    print("\n=== Code-switching ===")
    cs = code_switch_table(df)
    print(cs)
    cs.to_csv(TABLES_DIR / "code_switching_per_language.csv")

    print("\n=== Translation artifacts ===")
    art = artifact_table(df)
    print(art)
    art.to_csv(TABLES_DIR / "artifact_per_language.csv")

    print("\n=== Top tokens per (language, label) — first 5 rows of each language ===")
    all_top = pd.concat([top_tokens(df, lang) for lang in LANGUAGES], ignore_index=True)
    all_top.to_csv(TABLES_DIR / "top_tokens_per_language.csv", index=False)
    for lang in LANGUAGES:
        print(f"\n  {lang}:")
        print(all_top[all_top["language"] == lang].head(5).to_string(index=False))

    print("\nWriting figures ...")
    fig_class_balance(df, FIGURES_DIR / "class_balance.png")
    fig_length_dist(df, FIGURES_DIR / "length_distributions.png")
    fig_code_switch(df, FIGURES_DIR / "code_switching.png")
    fig_artifact(df, FIGURES_DIR / "artifacts.png")
    fig_confidence(df, FIGURES_DIR / "confidence_distribution.png")
    print(f"Figures in {FIGURES_DIR}")
    print(f"Tables  in {TABLES_DIR}")


if __name__ == "__main__":
    main()

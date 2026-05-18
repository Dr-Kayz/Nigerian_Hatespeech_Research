"""TF-IDF feature extractor with a shared multilingual vocabulary.

Builds a sklearn FeatureUnion concatenating:
    * word n-grams (default 1-2): captures words and short phrases
    * character n-grams (default 3-5, word-boundary aware): captures subword
      morphology — important for the African languages, which inflect heavily

When fit on the combined training data from all languages, the resulting
vectoriser produces a single feature space in which any language's text can be
represented with the same vocabulary. This lets one model train on combined
multilingual data, and lets a model trained on one language score text in
another, without refitting the vectoriser per language.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def make_tfidf_features(
    word_ngrams: tuple[int, int] = (1, 2),
    char_ngrams: tuple[int, int] = (3, 5),
    word_max_features: int = 50_000,
    char_max_features: int = 50_000,
    min_df: int = 5,
) -> FeatureUnion:
    """Return a FeatureUnion of word + char TF-IDF vectorisers."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngrams,
        max_features=word_max_features,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngrams,
        max_features=char_max_features,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=True,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])

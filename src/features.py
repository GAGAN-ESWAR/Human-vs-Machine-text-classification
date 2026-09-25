"""
features.py — Feature engineering for token-ID sequences.

Feature groups:
  1. TfidfFeatures          — TF-IDF over token-ID strings (unigrams to trigrams)
  2. StyleFeatures          — 9 hand-crafted stylometric / statistical features
  3. ExtendedStyleFeatures  — 9 base + bigram transitions, positional stats,
                               top-N token frequency fingerprints (bonus features)
  4. CombinedFeatures       — TF-IDF + StyleFeatures (BL3 baseline)
  5. EnhancedCombinedFeatures — TF-IDF + ExtendedStyleFeatures (tuned model)

Follows strict featurization ordering:
  - Fit ONLY on training fold; transform both train & val/test.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
import zlib

from src.features_generative import GenerativeStyleFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _texts_to_strings(texts: List[List[int]]) -> List[str]:
    """Convert list-of-ints to whitespace-joined strings for TF-IDF."""
    return [" ".join(map(str, t)) for t in texts]


def _entropy(arr: np.ndarray) -> float:
    """Compute Shannon entropy of a token-frequency distribution."""
    if len(arr) == 0:
        return 0.0
    counts = np.bincount(arr)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _repetition_rate(tokens: List[int]) -> float:
    """Fraction of adjacent identical token pairs."""
    if len(tokens) < 2:
        return 0.0
    pairs = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return pairs / (len(tokens) - 1)


def _max_run_length(tokens: List[int]) -> int:
    """Length of the longest run of the same token."""
    if not tokens:
        return 0
    max_run = cur = 1
    for a, b in zip(tokens, tokens[1:]):
        if a == b:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max_run


# ---------------------------------------------------------------------------
# 1. TF-IDF n-gram features
# ---------------------------------------------------------------------------

class TfidfFeatures(BaseEstimator, TransformerMixin):
    """
    Wraps TfidfVectorizer to accept List[List[int]] directly.

    Parameters
    ----------
    ngram_range : tuple, default (1, 3)
    max_features : int, default 150_000
    sublinear_tf : bool, default True
    """

    def __init__(self, ngram_range=(1, 3), max_features=150_000, sublinear_tf=True):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self._vec = None

    def fit(self, texts: List[List[int]], y=None):
        self._vec = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            dtype=np.float32,
        )
        self._vec.fit(_texts_to_strings(texts))
        return self

    def transform(self, texts: List[List[int]]):
        assert self._vec is not None, "Call fit() before transform()."
        return self._vec.transform(_texts_to_strings(texts))

    def fit_transform(self, texts: List[List[int]], y=None):
        return self.fit(texts, y).transform(texts)


# ---------------------------------------------------------------------------
# 2. Stylometric / statistical features
# ---------------------------------------------------------------------------

STYLE_FEATURE_NAMES = [
    "doc_length",          # total tokens
    "unique_tokens",       # vocabulary size
    "type_token_ratio",    # unique / total
    "oov_rate",            # fraction of token-0 (rare/unknown)
    "token_entropy",       # Shannon entropy of token distribution
    "compression_ratio",   # zlib compression ratio of text
]


class StyleFeatures(BaseEstimator, TransformerMixin):
    """
    Computes 6 hand-crafted stylometric features per document.
    Fit is a no-op (no data-dependent state needed).
    """

    def fit(self, texts: List[List[int]], y=None):
        return self

    def transform(self, texts: List[List[int]]) -> np.ndarray:
        X = np.zeros((len(texts), len(STYLE_FEATURE_NAMES)), dtype=np.float32)
        for i, tokens in enumerate(texts):
            arr = np.array(tokens, dtype=np.int32)
            n = len(arr)
            X[i, 0] = n
            X[i, 1] = len(set(tokens))
            X[i, 2] = X[i, 1] / (n + 1e-9)
            X[i, 3] = float(np.sum(arr == 0)) / (n + 1e-9)
            X[i, 4] = _entropy(arr)
            
            # Compression ratio
            s = " ".join(map(str, tokens)).encode('utf-8')
            if len(s) > 0:
                compressed = zlib.compress(s)
                X[i, 5] = len(compressed) / len(s)
            else:
                X[i, 5] = 1.0
        return X

    def fit_transform(self, texts: List[List[int]], y=None) -> np.ndarray:
        return self.fit(texts, y).transform(texts)

    @property
    def feature_names(self):
        return STYLE_FEATURE_NAMES


# ---------------------------------------------------------------------------
# 3. Combined features (TF-IDF sparse + StyleFeatures dense)
# ---------------------------------------------------------------------------

class CombinedFeatures(BaseEstimator, TransformerMixin):
    """
    Horizontally stacks sparse TF-IDF matrix with dense stylometric features.

    Returns a scipy.sparse.csr_matrix of shape
    (n_docs, tfidf_features + 9).
    """

    def __init__(self, ngram_range=(1, 3), max_features=150_000, sublinear_tf=True):
        self.tfidf = TfidfFeatures(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.style = StyleFeatures()

    def fit(self, texts: List[List[int]], y=None):
        self.tfidf.fit(texts)
        # StyleFeatures has no state, but call fit for API consistency
        self.style.fit(texts)
        return self

    def transform(self, texts: List[List[int]]):
        tfidf_X = self.tfidf.transform(texts)        # sparse
        style_X = self.style.transform(texts)         # dense
        style_sp = sp.csr_matrix(style_X)
        return sp.hstack([tfidf_X, style_sp], format="csr")

    def fit_transform(self, texts: List[List[int]], y=None):
        return self.fit(texts, y).transform(texts)


# ---------------------------------------------------------------------------
# 4. Extended stylometric features  (BONUS — new features for controlled study)
# ---------------------------------------------------------------------------

EXTENDED_FEATURE_NAMES = STYLE_FEATURE_NAMES + [
    # Bigram / transition features
    "bigram_entropy",       # Shannon entropy of adjacent token-pair distribution
    "unique_bigrams",       # number of distinct (a,b) pairs
    "bigram_ttr",           # unique bigrams / (n-1)  — bigram type-token ratio
    # Positional features
    "head_mean_token",      # mean token ID in first 20 tokens
    "tail_mean_token",      # mean token ID in last 20 tokens
    "head_tail_diff",       # abs difference between head and tail mean token IDs
    # Frequency concentration
    "top1_freq",            # relative frequency of most common token
    "top5_freq_sum",        # relative frequency of top-5 tokens combined
    "hapax_rate",           # fraction of tokens appearing exactly once
    # Vocabulary range
    "token_range",          # max_token_id - min_token_id
    "q75_token_id",         # 75th percentile of token ID distribution
    "q25_token_id",         # 25th percentile of token ID distribution
]

N_EXTENDED = len(EXTENDED_FEATURE_NAMES)


def _bigram_entropy(tokens: List[int]) -> tuple:
    """Return (entropy, n_unique, ttr) of adjacent token-pair distribution."""
    if len(tokens) < 2:
        return 0.0, 0, 0.0
    from collections import Counter
    bigrams = Counter(zip(tokens, tokens[1:]))
    total = sum(bigrams.values())
    p = np.array(list(bigrams.values()), dtype=np.float32) / total
    ent = float(-np.sum(p * np.log2(p + 1e-12)))
    n_unique = len(bigrams)
    ttr = n_unique / (len(tokens) - 1)
    return ent, n_unique, ttr


class ExtendedStyleFeatures(BaseEstimator, TransformerMixin):
    """
    18 stylometric features = 6 base StyleFeatures +
    12 extended features (bigram entropy, positional, freq-concentration).

    Fit is a no-op (no data-dependent state).
    For the bonus mark controlled study: compare CombinedFeatures (BL3 baseline)
    vs EnhancedCombinedFeatures (this class) under identical CV protocol.
    """

    def fit(self, texts: List[List[int]], y=None):
        return self

    def transform(self, texts: List[List[int]]) -> np.ndarray:
        base = StyleFeatures().transform(texts)
        ext = np.zeros((len(texts), N_EXTENDED - len(STYLE_FEATURE_NAMES)), dtype=np.float32)

        for i, tokens in enumerate(texts):
            arr = np.array(tokens, dtype=np.int32)
            n = len(arr)
            from collections import Counter
            freq = Counter(tokens)

            # Bigram features
            bg_ent, bg_uniq, bg_ttr = _bigram_entropy(tokens)
            ext[i, 0] = bg_ent
            ext[i, 1] = bg_uniq
            ext[i, 2] = bg_ttr

            # Positional features
            head = arr[:20] if n >= 20 else arr
            tail = arr[-20:] if n >= 20 else arr
            head_mean = float(np.mean(head)) if len(head) > 0 else 0.0
            tail_mean = float(np.mean(tail)) if len(tail) > 0 else 0.0
            ext[i, 3] = head_mean
            ext[i, 4] = tail_mean
            ext[i, 5] = abs(head_mean - tail_mean)

            # Frequency concentration
            most_common = freq.most_common(5)
            ext[i, 6] = most_common[0][1] / (n + 1e-9)  # top-1
            ext[i, 7] = sum(c for _, c in most_common) / (n + 1e-9)  # top-5
            hapax = sum(1 for c in freq.values() if c == 1)
            ext[i, 8] = hapax / (n + 1e-9)

            # Vocabulary range
            ext[i, 9] = float(arr.max() - arr.min()) if n > 0 else 0.0
            ext[i, 10] = float(np.percentile(arr, 75)) if n > 0 else 0.0
            ext[i, 11] = float(np.percentile(arr, 25)) if n > 0 else 0.0

        return np.hstack([base, ext])   # (N, 22)

    def fit_transform(self, texts: List[List[int]], y=None) -> np.ndarray:
        return self.fit(texts, y).transform(texts)

    @property
    def feature_names(self):
        return EXTENDED_FEATURE_NAMES


# ---------------------------------------------------------------------------
# 5. Enhanced combined features  (TF-IDF + ExtendedStyleFeatures)
# ---------------------------------------------------------------------------

class EnhancedCombinedFeatures(BaseEstimator, TransformerMixin):
    """
    TF-IDF (1–3 gram) + 22 extended stylometric features.
    Used for the tuned / bonus-mark model.
    """

    def __init__(self, ngram_range=(1, 3), max_features=150_000, sublinear_tf=True):
        self.tfidf = TfidfFeatures(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.style = ExtendedStyleFeatures()

    def fit(self, texts: List[List[int]], y=None):
        self.tfidf.fit(texts)
        self.style.fit(texts)
        return self

    def transform(self, texts: List[List[int]]):
        tfidf_X = self.tfidf.transform(texts)
        style_X = self.style.transform(texts)
        style_sp = sp.csr_matrix(style_X)
        return sp.hstack([tfidf_X, style_sp], format="csr")

    def fit_transform(self, texts: List[List[int]], y=None):
        return self.fit(texts, y).transform(texts)


# ---------------------------------------------------------------------------
# 6. Generative combined features (TF-IDF + ExtendedStyle + GenerativeStyle)
# ---------------------------------------------------------------------------

class GenerativeCombinedFeatures(BaseEstimator, TransformerMixin):
    """
    Horizontally stacks sparse TF-IDF, dense ExtendedStyle, and GenerativeStyle features.
    """

    def __init__(self, ngram_range=(1, 3), max_features=150_000, sublinear_tf=True):
        self.tfidf = TfidfFeatures(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.style = ExtendedStyleFeatures()
        self.gen = GenerativeStyleFeatures(alpha=0.1, use_trigram=True, vocab_size=18438)

    def fit(self, texts: List[List[int]], y=None):
        self.tfidf.fit(texts)
        self.style.fit(texts)
        self.gen.fit(texts, y)
        return self

    def transform(self, texts: List[List[int]]):
        tfidf_X = self.tfidf.transform(texts)
        style_X = self.style.transform(texts)
        gen_X = self.gen.transform(texts)
        return sp.hstack([tfidf_X, sp.csr_matrix(style_X), sp.csr_matrix(gen_X)], format="csr")

    def fit_transform(self, texts: List[List[int]], y=None):
        tfidf_X = self.tfidf.fit_transform(texts, y)
        style_X = self.style.fit_transform(texts, y)
        gen_X = self.gen.fit_transform(texts, y)
        return sp.hstack([tfidf_X, sp.csr_matrix(style_X), sp.csr_matrix(gen_X)], format="csr")

# ---------------------------------------------------------------------------
# 7. Heaps' Law Feature
# ---------------------------------------------------------------------------

def heaps_exponent(tokens: List[int]) -> float:
    """Vocabulary-growth exponent (Heaps' Law)."""
    n = len(tokens)
    if n < 10:
        return 0.0
    seen = set()
    growth = []
    for i, t in enumerate(tokens, 1):
        seen.add(t)
        if i % max(1, n // 20) == 0:  # sample ~20 points along the sequence
            growth.append((i, len(seen)))
    if len(growth) < 3:
        return 0.0
    log_pos = np.log([g[0] for g in growth])
    log_vocab = np.log([g[1] for g in growth])
    beta, _ = np.polyfit(log_pos, log_vocab, 1)
    return beta


class HeapsLawFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts the Heaps' law exponent (beta).
    Fit is a no-op (no data-dependent state needed).
    """

    def fit(self, texts: List[List[int]], y=None):
        return self

    def transform(self, texts: List[List[int]]) -> np.ndarray:
        heaps = np.zeros((len(texts), 1), dtype=np.float32)
        for i, tokens in enumerate(texts):
            heaps[i, 0] = heaps_exponent(tokens)
        return heaps

    def fit_transform(self, texts: List[List[int]], y=None) -> np.ndarray:
        return self.transform(texts)


# ---------------------------------------------------------------------------
# 8. Champion combined features
# ---------------------------------------------------------------------------

class ChampionFeatures(BaseEstimator, TransformerMixin):
    """
    Horizontally stacks sparse TF-IDF (150k), dense ExtendedStyle (18),
    GenerativeStyle (8), and HeapsLawFeatures (1). Total = 150,027 features.
    """

    def __init__(self, ngram_range=(1, 3), max_features=150_000, sublinear_tf=True):
        self.tfidf = TfidfFeatures(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.style = ExtendedStyleFeatures()
        self.gen = GenerativeStyleFeatures(alpha=0.1, use_trigram=True, vocab_size=18438)
        self.heaps = HeapsLawFeatures()

    def fit(self, texts: List[List[int]], y=None):
        self.tfidf.fit(texts)
        self.style.fit(texts)
        self.gen.fit(texts, y)
        self.heaps.fit(texts, y)
        return self

    def transform(self, texts: List[List[int]]):
        tfidf_X = self.tfidf.transform(texts)
        style_X = self.style.transform(texts)
        gen_X = self.gen.transform(texts)
        heaps_X = self.heaps.transform(texts)
        return sp.hstack([
            tfidf_X, 
            sp.csr_matrix(style_X), 
            sp.csr_matrix(gen_X),
            sp.csr_matrix(heaps_X)
        ], format="csr")

    def fit_transform(self, texts: List[List[int]], y=None):
        tfidf_X = self.tfidf.fit_transform(texts, y)
        style_X = self.style.fit_transform(texts, y)
        gen_X = self.gen.fit_transform(texts, y)
        heaps_X = self.heaps.fit_transform(texts, y)
        return sp.hstack([
            tfidf_X, 
            sp.csr_matrix(style_X), 
            sp.csr_matrix(gen_X),
            sp.csr_matrix(heaps_X)
        ], format="csr")


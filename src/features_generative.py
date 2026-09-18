"""
Generative / stylometric features for token-sequence classification.

Key design choice: ALL transition counts use Python dicts keyed by
observed (t1, t2) or (t1, t2, t3) tuples -- never a dense V x V or
V x V x V array. With V ~= 18,438, a dense bigram matrix alone is
~1.3 GB of float64 (18438**2 * 8 bytes), and a trigram matrix is
infeasible (~24 TB). A dict only stores pairs/triples that actually
occur in the corpus, which for ~10k documents is a tiny fraction of
the full space (typically a few hundred thousand entries, a few MB).
This is almost certainly why the earlier generative-feature attempt
crashed on memory -- not the underlying idea.

Leave-One-Out (LOO) correction: when scoring a training document, we
subtract that document's own contribution to the counts before
computing its likelihood, so the model can't "cheat" by having
memorized itself. This is done in O(len(doc)) per document, not by
rebuilding global counts per row.
"""

import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GenerativeStyleFeatures(BaseEstimator, TransformerMixin):
    """
    Produces dense features per document:
      - ll_human_bi, ll_ai_bi     : class-conditional bigram Markov log-likelihood (LOO on train)
      - ll_human_tri, ll_ai_tri   : class-conditional trigram Markov log-likelihood (LOO on train), 0 if too short
      - ll_diff_bi, ll_diff_tri   : ll_human - ll_ai (the single most informative derived signal)
      - zipf_slope                : slope of log(freq) ~ log(rank) within the document
      - burstiness                : mean std-dev of inter-arrival gaps for repeated tokens
    """

    def __init__(self, alpha=0.1, use_trigram=True, vocab_size=18438):
        self.alpha = alpha
        self.use_trigram = use_trigram
        self.vocab_size = vocab_size

    # ---------- fitting class-conditional transition counts ----------

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("GenerativeStyleFeatures requires y (labels) at fit time.")
        self._fit_counts(X, y)
        return self

    def _fit_counts(self, X, y):
        # bigram counts: dict[(t1,t2)] -> count, per class
        self.bi_counts_ = {0: Counter(), 1: Counter()}
        self.uni_counts_ = {0: Counter(), 1: Counter()}
        self.bi_totals_ = {0: 0, 1: 0}

        self.tri_counts_ = {0: Counter(), 1: Counter()} if self.use_trigram else None
        self.bi_context_counts_ = {0: Counter(), 1: Counter()} if self.use_trigram else None

        for tokens, label in zip(X, y):
            cls = int(label)
            n = len(tokens)
            for i in range(1, n):
                t1, t2 = tokens[i - 1], tokens[i]
                self.bi_counts_[cls][(t1, t2)] += 1
                self.uni_counts_[cls][t1] += 1
                self.bi_totals_[cls] += 1
            if self.use_trigram:
                for i in range(2, n):
                    t1, t2, t3 = tokens[i - 2], tokens[i - 1], tokens[i]
                    self.tri_counts_[cls][(t1, t2, t3)] += 1
                    self.bi_context_counts_[cls][(t1, t2)] += 1
        return self

    # ---------- scoring ----------

    def _bigram_loglik(self, tokens, cls, loo_bi=None, loo_uni=None):
        """LOO-corrected bigram log-likelihood under class `cls`."""
        V = self.vocab_size
        alpha = self.alpha
        bi_c = self.bi_counts_[cls]
        uni_c = self.uni_counts_[cls]
        ll = 0.0
        n_pairs = 0
        for i in range(1, len(tokens)):
            t1, t2 = tokens[i - 1], tokens[i]
            c_bi = bi_c.get((t1, t2), 0)
            c_uni = uni_c.get(t1, 0)
            if loo_bi is not None:
                c_bi = max(0, c_bi - loo_bi.get((t1, t2), 0))
                c_uni = max(0, c_uni - loo_uni.get(t1, 0))
            p = (c_bi + alpha) / (c_uni + alpha * V)
            ll += math.log(p)
            n_pairs += 1
        return ll / n_pairs if n_pairs else 0.0

    def _trigram_loglik(self, tokens, cls, loo_tri=None, loo_bictx=None):
        if not self.use_trigram or len(tokens) < 3:
            return 0.0
        V = self.vocab_size
        alpha = self.alpha
        tri_c = self.tri_counts_[cls]
        bictx_c = self.bi_context_counts_[cls]
        ll = 0.0
        n = 0
        for i in range(2, len(tokens)):
            t1, t2, t3 = tokens[i - 2], tokens[i - 1], tokens[i]
            c_tri = tri_c.get((t1, t2, t3), 0)
            c_ctx = bictx_c.get((t1, t2), 0)
            if loo_tri is not None:
                c_tri = max(0, c_tri - loo_tri.get((t1, t2, t3), 0))
                c_ctx = max(0, c_ctx - loo_bictx.get((t1, t2), 0))
            p = (c_tri + alpha) / (c_ctx + alpha * V)
            ll += math.log(p)
            n += 1
        return ll / n if n else 0.0

    @staticmethod
    def _zipf_slope(tokens):
        counts = Counter(tokens)
        freqs = sorted(counts.values(), reverse=True)
        if len(freqs) < 3:
            return 0.0
        ranks = np.log(np.arange(1, len(freqs) + 1))
        logf = np.log(freqs)
        # simple linear regression slope
        A = np.vstack([ranks, np.ones_like(ranks)]).T
        slope, _ = np.linalg.lstsq(A, logf, rcond=None)[0]
        return float(slope)

    @staticmethod
    def _burstiness(tokens):
        positions = defaultdict(list)
        for i, t in enumerate(tokens):
            positions[t].append(i)
        gaps_std = []
        for t, pos in positions.items():
            if len(pos) >= 3:
                gaps = np.diff(pos)
                if len(gaps) > 1:
                    gaps_std.append(float(np.std(gaps)))
        return float(np.mean(gaps_std)) if gaps_std else 0.0

    # ---------- transform ----------

    def transform(self, X, y=None, loo=False):
        rows = []
        for idx, tokens in enumerate(X):
            loo_bi = loo_uni = loo_tri = loo_bictx = None
            if loo:
                cls_self = int(y[idx])
                loo_bi, loo_uni = Counter(), Counter()
                for i in range(1, len(tokens)):
                    t1, t2 = tokens[i - 1], tokens[i]
                    loo_bi[(t1, t2)] += 1
                    loo_uni[t1] += 1
                if self.use_trigram:
                    loo_tri, loo_bictx = Counter(), Counter()
                    for i in range(2, len(tokens)):
                        t1, t2, t3 = tokens[i - 2], tokens[i - 1], tokens[i]
                        loo_tri[(t1, t2, t3)] += 1
                        loo_bictx[(t1, t2)] += 1

            if loo:
                ll_h_bi = self._bigram_loglik(tokens, cls_self, loo_bi, loo_uni) \
                    if cls_self == 0 else self._bigram_loglik(tokens, 0)
                ll_a_bi = self._bigram_loglik(tokens, cls_self, loo_bi, loo_uni) \
                    if cls_self == 1 else self._bigram_loglik(tokens, 1)
                
                if self.use_trigram:
                    ll_h_tri = self._trigram_loglik(tokens, cls_self, loo_tri, loo_bictx) \
                        if cls_self == 0 else self._trigram_loglik(tokens, 0)
                    ll_a_tri = self._trigram_loglik(tokens, cls_self, loo_tri, loo_bictx) \
                        if cls_self == 1 else self._trigram_loglik(tokens, 1)
                else:
                    ll_h_tri = ll_a_tri = 0.0
            else:
                ll_h_bi = self._bigram_loglik(tokens, 0)
                ll_a_bi = self._bigram_loglik(tokens, 1)
                if self.use_trigram:
                    ll_h_tri = self._trigram_loglik(tokens, 0)
                    ll_a_tri = self._trigram_loglik(tokens, 1)
                else:
                    ll_h_tri = ll_a_tri = 0.0

            zipf = self._zipf_slope(tokens)
            burst = self._burstiness(tokens)

            rows.append([
                ll_h_bi, ll_a_bi, ll_h_bi - ll_a_bi,
                ll_h_tri, ll_a_tri, ll_h_tri - ll_a_tri,
                zipf, burst,
            ])
        return np.array(rows, dtype=np.float64)

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        return self.transform(X, y=y, loo=True)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "gen_ll_human_bi", "gen_ll_ai_bi", "gen_ll_diff_bi",
            "gen_ll_human_tri", "gen_ll_ai_tri", "gen_ll_diff_tri",
            "gen_zipf_slope", "gen_burstiness",
        ])

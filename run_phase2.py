import numpy as np
import scipy.sparse as sp
from scipy.stats import skew, kurtosis
from collections import Counter, defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import math
import sys

from src.data_loader import load_train
from src.features import ChampionFeatures, TfidfFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate

warnings.filterwarnings("ignore")

# 1. 4th-order Markov
class QuadgramOnlyFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.1, vocab_size=18438):
        self.alpha = alpha
        self.vocab_size = vocab_size

    def fit(self, X, y=None):
        self.quad_counts_ = {0: Counter(), 1: Counter()}
        self.tri_context_counts_ = {0: Counter(), 1: Counter()}
        for tokens, label in zip(X, y):
            cls = int(label)
            n = len(tokens)
            for i in range(3, n):
                t1, t2, t3, t4 = tokens[i - 3], tokens[i - 2], tokens[i - 1], tokens[i]
                self.quad_counts_[cls][(t1, t2, t3, t4)] += 1
                self.tri_context_counts_[cls][(t1, t2, t3)] += 1
        return self

    def _quadgram_loglik(self, tokens, cls, loo_quad=None, loo_trictx=None):
        if len(tokens) < 4: return 0.0
        V, alpha = self.vocab_size, self.alpha
        q_c = self.quad_counts_[cls]
        tc_c = self.tri_context_counts_[cls]
        ll, n = 0.0, 0
        for i in range(3, len(tokens)):
            t1, t2, t3, t4 = tokens[i - 3], tokens[i - 2], tokens[i - 1], tokens[i]
            cq = q_c.get((t1, t2, t3, t4), 0)
            ct = tc_c.get((t1, t2, t3), 0)
            if loo_quad is not None:
                cq = max(0, cq - loo_quad.get((t1, t2, t3, t4), 0))
                ct = max(0, ct - loo_trictx.get((t1, t2, t3), 0))
            p = (cq + alpha) / (ct + alpha * V)
            ll += math.log(p)
            n += 1
        return ll / n if n else 0.0

    def transform(self, X, y=None, loo=False):
        rows = []
        for idx, tokens in enumerate(X):
            loo_q = loo_tc = None
            if loo:
                cls_self = int(y[idx])
                loo_q, loo_tc = Counter(), Counter()
                for i in range(3, len(tokens)):
                    t1, t2, t3, t4 = tokens[i - 3], tokens[i - 2], tokens[i - 1], tokens[i]
                    loo_q[(t1, t2, t3, t4)] += 1
                    loo_tc[(t1, t2, t3)] += 1
            if loo:
                ll_h = self._quadgram_loglik(tokens, cls_self, loo_q, loo_tc) if cls_self == 0 else self._quadgram_loglik(tokens, 0)
                ll_a = self._quadgram_loglik(tokens, cls_self, loo_q, loo_tc) if cls_self == 1 else self._quadgram_loglik(tokens, 1)
            else:
                ll_h = self._quadgram_loglik(tokens, 0)
                ll_a = self._quadgram_loglik(tokens, 1)
            rows.append([ll_h, ll_a, ll_h - ll_a])
        return np.array(rows, dtype=np.float64)

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        return self.transform(X, y=y, loo=True)

# 2. ShapeStatsFeatures
class ShapeStatsFeatures(BaseEstimator, TransformerMixin):
    def fit(self, texts, y=None): return self
    def transform(self, texts):
        X = np.zeros((len(texts), 2), dtype=np.float32)
        for i, tokens in enumerate(texts):
            if len(tokens) > 2:
                X[i, 0] = skew(tokens)
                X[i, 1] = kurtosis(tokens)
        return X
    def fit_transform(self, texts, y=None): return self.transform(texts)

# 3. SlidingEntropyFeatures
class SlidingEntropyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, texts, y=None): return self
    def transform(self, texts):
        X = np.zeros((len(texts), 2), dtype=np.float32)
        for i, tokens in enumerate(texts):
            n = len(tokens)
            if n < 3:
                X[i, 0] = 0.0
            else:
                trigrams = [tuple(tokens[j:j+3]) for j in range(n-2)]
                counts = Counter(trigrams)
                repeats = sum(1 for c in counts.values() if c > 1)
                X[i, 0] = repeats / len(trigrams)
            if n == 0:
                X[i, 1] = 0.0
            else:
                w = min(100, n)
                entropies = []
                for j in range(n - w + 1):
                    window = tokens[j:j+w]
                    counts = Counter(window)
                    p = np.array(list(counts.values())) / w
                    ent = -np.sum(p * np.log2(p + 1e-12))
                    entropies.append(ent)
                X[i, 1] = np.mean(entropies) if entropies else 0.0
        return X
    def fit_transform(self, texts, y=None): return self.transform(texts)

# 4. GraphTopologyFeatures
class GraphTopologyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, texts, y=None): return self
    def transform(self, texts):
        X = np.zeros((len(texts), 2), dtype=np.float32)
        for i, tokens in enumerate(texts):
            n = len(tokens)
            if n < 2:
                continue
            edges = set()
            out_degrees = defaultdict(set)
            for j in range(n - 1):
                u, v = tokens[j], tokens[j+1]
                edges.add((u, v))
                out_degrees[u].add(v)
            num_nodes = len(set(tokens))
            X[i, 0] = len(edges) / (num_nodes * num_nodes) if num_nodes > 0 else 0
            if out_degrees:
                X[i, 1] = sum(len(v) for v in out_degrees.values()) / len(out_degrees)
        return X
    def fit_transform(self, texts, y=None): return self.transform(texts)

# 5. DuplicateFeatures
class DuplicateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.tfidf = TfidfFeatures(ngram_range=(1,2), max_features=10000)
        self.X_train_tfidf = self.tfidf.fit_transform(X)
        return self
    def transform(self, X):
        X_tfidf = self.tfidf.transform(X)
        sim = X_tfidf.dot(self.X_train_tfidf.T).max(axis=1).toarray()
        return sim
    def fit_transform(self, X, y=None):
        self.tfidf = TfidfFeatures(ngram_range=(1,2), max_features=10000)
        self.X_train_tfidf = self.tfidf.fit_transform(X)
        sim = self.X_train_tfidf.dot(self.X_train_tfidf.T).copy()
        sim.setdiag(0)
        return sim.max(axis=1).toarray()

# --- Wrapper ---
class TestFeatureWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, new_feat, tfidf_max=150000):
        self.champ = ChampionFeatures(max_features=tfidf_max)
        self.new_feat = new_feat
    def fit(self, X, y=None):
        self.champ.fit(X, y)
        if self.new_feat:
            self.new_feat.fit(X, y)
        return self
    def transform(self, X):
        xc = self.champ.transform(X)
        if self.new_feat:
            xn = self.new_feat.transform(X)
            return sp.hstack([xc, sp.csr_matrix(xn)], format="csr")
        return xc
    def fit_transform(self, X, y=None):
        xc = self.champ.fit_transform(X, y)
        if self.new_feat:
            xn = self.new_feat.fit_transform(X, y)
            return sp.hstack([xc, sp.csr_matrix(xn)], format="csr")
        return xc

def main():
    print("============================================================")
    print("  Phase 2: Individual Feature Experiments (CV Screening)")
    print("============================================================")

    _, train_texts, y_train = load_train("train.json")

    experiments = {
        "Exp1_Quadgram": TestFeatureWrapper(QuadgramOnlyFeatures()),
        "Exp2_ShapeStats": TestFeatureWrapper(ShapeStatsFeatures()),
        "Exp3_SlidingEntropy": TestFeatureWrapper(SlidingEntropyFeatures()),
        "Exp4_GraphTopology": TestFeatureWrapper(GraphTopologyFeatures()),
        "Exp5_Duplicate": TestFeatureWrapper(DuplicateFeatures()),
        "Exp6_Tfidf200k": TestFeatureWrapper(None, tfidf_max=200000),
    }
    
    experiment_to_run = sys.argv[1] if len(sys.argv) > 1 else None

    if experiment_to_run:
        print(f"Running single experiment: {experiment_to_run}")
        feat_factory = lambda: experiments[experiment_to_run]
        res, _ = cv_evaluate(
            model_factory=lambda: make_lgbm(scale_pos_weight=1.8483),
            feature_factory=feat_factory,
            texts=train_texts,
            y=y_train,
            verbose=False,
            model_name=experiment_to_run
        )
        print(f"[{experiment_to_run}] CV Acc: {res['mean_acc']:.4f}  CV AUC: {res['mean_auc']:.4f}")
    else:
        for name, wrapper in experiments.items():
            print(f"\n--- Running {name} ---")
            feat_factory = lambda wrapper=wrapper: wrapper
            res, _ = cv_evaluate(
                model_factory=lambda: make_lgbm(scale_pos_weight=1.8483),
                feature_factory=feat_factory,
                texts=train_texts,
                y=y_train,
                verbose=False,
                model_name=name
            )
            print(f"[{name}] CV Acc: {res['mean_acc']:.4f}  CV AUC: {res['mean_auc']:.4f}")

if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_loader import load_train, load_test
from src.features import ChampionFeatures, TfidfFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate, train_and_predict, generate_submission

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
    print("  Generating Final Submission: Champion + DuplicateFeatures")
    print("============================================================")

    print("Loading data...")
    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")

    print(f"Loaded {len(train_texts)} train samples, {len(test_texts)} test samples.")

    # 1. Feature Factory
    feat_factory = lambda: TestFeatureWrapper(DuplicateFeatures())

    # 2. Run leak-free CV just to confirm the metric once more (optional but good for log)
    print("\nRunning CV to confirm performance...")
    results, oof_probs = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=1.8483),
        feature_factory=feat_factory,
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="Champion+Duplicate"
    )

    print("\nCV Results Summary:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # 3. Train on full train data and predict on test
    print("\nTraining on full train data and generating test predictions...")
    test_probs = train_and_predict(
        model_factory=lambda: make_lgbm(scale_pos_weight=1.8483),
        feature_factory=feat_factory,
        train_texts=train_texts,
        y_train=y_train,
        test_texts=test_texts,
    )

    # 4. Generate Submission CSV
    out_path = generate_submission(
        test_ids=test_ids,
        test_probs=test_probs,
        sample_sub_path="sample_submission.csv",
        out_dir="submissions",
        model_name="Champion-Duplicate",
        threshold=0.5
    )
    print(f"\nSuccessfully generated submission file: {out_path}")

if __name__ == "__main__":
    main()

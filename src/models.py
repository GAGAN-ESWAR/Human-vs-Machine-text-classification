"""
models.py — Model definitions for all four baselines.

BL1 : TF-IDF + LinearSVC / LogisticRegression
BL2 : Stylometric features + LightGBM / XGBoost
BL3 : Combined (TF-IDF + stylo) + LightGBM
BL4 : OOF Stacking Ensemble → LogisticRegression meta-learner

All classifiers are wrapped to expose a unified interface:
    fit(X, y) / predict(X) / predict_proba(X)
"""

import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from typing import List, Optional


# ---------------------------------------------------------------------------
# Baseline 1 — Linear models on TF-IDF
# ---------------------------------------------------------------------------

def make_logreg(C: float = 1.0, max_iter: int = 2000) -> LogisticRegression:
    """Logistic Regression with balanced class weights."""
    return LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="saga",
        max_iter=max_iter,
        n_jobs=-1,
        random_state=42,
    )


def make_linear_svc(C: float = 0.1) -> CalibratedClassifierCV:
    """
    LinearSVC wrapped in CalibratedClassifierCV to expose predict_proba.
    Uses balanced class weights.
    """
    svc = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    return CalibratedClassifierCV(svc, cv=3, method="isotonic")


# ---------------------------------------------------------------------------
# Baseline 2 / 3 — Gradient Boosted Trees
# ---------------------------------------------------------------------------

def make_lgbm(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    scale_pos_weight: Optional[float] = None,
    n_jobs: int = -1,
) -> lgb.LGBMClassifier:
    """
    LightGBM binary classifier.

    scale_pos_weight: ratio of negative/positive samples to handle class imbalance.
    If None, 'is_unbalance=True' is used instead.
    """
    extra = {}
    if scale_pos_weight is not None:
        extra["scale_pos_weight"] = scale_pos_weight
    else:
        extra["is_unbalance"] = True

    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=n_jobs,
        random_state=42,
        verbose=-1,
        **extra,
    )


def make_xgb(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    scale_pos_weight: float = 1.848,  # 6837 / 3699 ≈ class ratio
    n_jobs: int = -1,
) -> xgb.XGBClassifier:
    """XGBoost binary classifier with class-imbalance correction."""
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=n_jobs,
        random_state=42,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Baseline 4 — OOF Stacking Ensemble
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """
    Trains a LogisticRegression meta-learner on OOF probability arrays
    produced by the base classifiers.

    Usage
    -----
    ensemble = StackingEnsemble()
    ensemble.fit(oof_probs_train, y_train)  # oof_probs: (N, n_base_models)
    probs = ensemble.predict_proba(oof_probs_test)
    preds = ensemble.predict(oof_probs_test)
    """

    def __init__(self, C: float = 1.0):
        self.meta = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

    def fit(self, oof_probs: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        oof_probs : (N, n_base_models)  — OOF probability of class 1 per model
        y         : (N,)                — true binary labels
        """
        assert oof_probs.ndim == 2, "oof_probs must be 2-D"
        self.meta.fit(oof_probs, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.meta.predict(X)


# ---------------------------------------------------------------------------
# Catalogue — easy access by name
# ---------------------------------------------------------------------------

def get_model(name: str):
    """
    Factory for models by name.

    Names: 'logreg', 'linearsvc', 'lgbm', 'xgb'
    """
    registry = {
        "logreg": make_logreg,
        "linearsvc": make_linear_svc,
        "lgbm": make_lgbm,
        "xgb": make_xgb,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name]()

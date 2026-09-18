"""
tune.py — Hyperparameter tuning + extended features + threshold optimisation.

This script produces the bonus-mark controlled comparison:
  CONTROL  : BL3-Combined  (TF-IDF + 9 stylo features, default LGBM params)
  TREATMENT: Tuned-Enhanced (TF-IDF + 22 extended stylo features, tuned LGBM)

Workflow
--------
1. Load data
2. Control CV   — BL3 as-is (results already logged from train.py; re-run here
                  for identical random seed reproducibility)
3. Treatment CV — EnhancedCombinedFeatures + RandomizedSearchCV for LGBM params
4. Threshold tuning on OOF probs of best model
5. Final retrain on full data → generate best submission
6. Print side-by-side comparison table (bonus mark evidence)

Run:
    venv\\Scripts\\python tune.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.data_loader import load_train, load_test, load_sample_submission
from src.features import CombinedFeatures, EnhancedCombinedFeatures
from src.evaluate import (
    cv_evaluate,
    train_and_predict,
    generate_submission,
    print_confusion_matrix,
)
from src.models import make_lgbm

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import scipy.sparse as sp

SCALE_POS_WEIGHT = 6837 / 3699   # class B / class A ratio

# ---------------------------------------------------------------------------
# Hyperparameter search space for LightGBM
# ---------------------------------------------------------------------------
LGBM_PARAM_DIST = {
    "n_estimators":    [300, 500, 700, 1000],
    "learning_rate":   [0.01, 0.03, 0.05, 0.08],
    "num_leaves":      [31, 63, 127, 255],
    "min_child_samples": [10, 20, 30, 50],
    "subsample":       [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":[0.5, 0.7, 0.8, 1.0],
    "reg_alpha":       [0.0, 0.1, 0.5, 1.0],
    "reg_lambda":      [0.0, 0.1, 0.5, 1.0],
}


def combined_factory():
    return CombinedFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)


def enhanced_factory():
    return EnhancedCombinedFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)


# ---------------------------------------------------------------------------
# Threshold optimisation on OOF probabilities
# ---------------------------------------------------------------------------

def tune_threshold(oof_probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Grid-search over [0.30 .. 0.70] to find threshold maximising accuracy.
    Returns the optimal threshold.
    """
    best_acc, best_t = 0.0, 0.5
    for t in np.arange(0.30, 0.71, 0.01):
        preds = (oof_probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return round(float(best_t), 2)


# ---------------------------------------------------------------------------
# RandomizedSearchCV wrapper that works with our sparse feature matrices
# ---------------------------------------------------------------------------

def lgbm_random_search(train_texts, y_train: np.ndarray, n_iter: int = 20) -> dict:
    """
    Run RandomizedSearchCV for LightGBM on the fast 22-dim stylometric features
    (not the 150K TF-IDF matrix — that would take hours).
    Best params are then transferred to the full enhanced model.
    Returns the best params dict.
    """
    from src.features import ExtendedStyleFeatures
    print(f"  Building 22-dim style features for fast hyperparameter search ...")
    style_feat = ExtendedStyleFeatures()
    X_style = style_feat.fit_transform(train_texts)
    print(f"  Style feature matrix: {X_style.shape}")

    print(f"  Running RandomizedSearchCV (n_iter={n_iter}, cv=3) ...")
    base_lgbm = lgb.LGBMClassifier(
        scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    rs = RandomizedSearchCV(
        base_lgbm,
        param_distributions=LGBM_PARAM_DIST,
        n_iter=n_iter,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=1,
        verbose=0,
        refit=True,
    )
    rs.fit(X_style, y_train)
    print(f"  Best CV accuracy (3-fold on style features): {rs.best_score_:.4f}")
    print(f"  Best params: {rs.best_params_}")
    return rs.best_params_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  CP219 Tuning Run — Extended Features + Hyperparameter Search")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    print(f"  Train: {len(train_texts)} docs | A={np.sum(y_train==0)} B={np.sum(y_train==1)}")

    all_results = {}

    # ================================================================
    # CONTROL  — BL3-Combined (baseline, identical to train.py)
    # ================================================================
    print("\n[2/5] CONTROL: BL3-Combined (TF-IDF + 9 stylo features, default LGBM) …")
    res_ctrl, oof_ctrl = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT),
        feature_factory=combined_factory,
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="CONTROL-BL3",
    )
    all_results["CONTROL (BL3-Combined)"] = res_ctrl
    ctrl_threshold = tune_threshold(oof_ctrl, y_train)
    print(f"  Optimal threshold (control): {ctrl_threshold}")

    # ================================================================
    # TREATMENT Step 1 — Feature-space search (full train for RS input)
    # ================================================================
    print("\n[3/5] TREATMENT Step 1: Preparing for hyperparameter search ...")

    # ================================================================
    # TREATMENT Step 2 — Hyperparameter search
    # ================================================================
    print("\n[4/5] TREATMENT Step 2: RandomizedSearchCV for LightGBM ...")
    best_params = lgbm_random_search(train_texts, y_train, n_iter=20)

    # ================================================================
    # TREATMENT Step 3 — 5-fold CV with tuned params + enhanced features
    # ================================================================
    print("\n  TREATMENT: 5-fold CV with tuned params + enhanced features …")

    def tuned_lgbm_factory():
        return lgb.LGBMClassifier(
            **best_params,
            scale_pos_weight=SCALE_POS_WEIGHT,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

    res_trt, oof_trt = cv_evaluate(
        model_factory=tuned_lgbm_factory,
        feature_factory=enhanced_factory,
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="TREATMENT-Tuned",
    )
    all_results["TREATMENT (Tuned+Enhanced)"] = res_trt
    trt_threshold = tune_threshold(oof_trt, y_train)
    print(f"  Optimal threshold (treatment): {trt_threshold}")

    # OOF confusion matrix for treatment
    oof_trt_preds = (oof_trt >= trt_threshold).astype(int)
    print_confusion_matrix(y_train, oof_trt_preds)

    # ================================================================
    # Bonus mark — controlled comparison table
    # ================================================================
    print("\n" + "=" * 65)
    print("  BONUS MARK EVIDENCE — Controlled Comparison")
    print("  Same CV protocol (5-fold StratifiedKFold, seed=42)")
    print("=" * 65)
    print(f"{'Model':<35} {'Acc':>8} {'+-':>6} {'F1':>8} {'AUC':>8}")
    print("-" * 65)
    for name, res in all_results.items():
        print(
            f"{name:<35} "
            f"{res['mean_acc']:>8.4f} "
            f"{res['std_acc']:>6.4f} "
            f"{res['mean_f1']:>8.4f} "
            f"{res['mean_auc']:>8.4f}"
        )
    delta_acc = res_trt['mean_acc'] - res_ctrl['mean_acc']
    delta_auc = res_trt['mean_auc'] - res_ctrl['mean_auc']
    print("-" * 65)
    print(f"  Delta (Treatment - Control): Acc={delta_acc:+.4f}  AUC={delta_auc:+.4f}")
    print("=" * 65)

    # ================================================================
    # Final retrain on full data → submission
    # ================================================================
    print("\n[5/5] Retraining on full data - generating best submission ...")
    # Pick the better model for final submission
    if res_ctrl['mean_acc'] >= res_trt['mean_acc']:
        print("  Control model won - using BL3-Combined params with optimal threshold.")
        feat_final = combined_factory()
        final_threshold = ctrl_threshold

        def final_clf_factory():
            return make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
    else:
        print("  Treatment model won - using Tuned+Enhanced params.")
        feat_final = enhanced_factory()
        final_threshold = trt_threshold

        def final_clf_factory():
            return lgb.LGBMClassifier(
                **best_params,
                scale_pos_weight=SCALE_POS_WEIGHT,
                random_state=42, verbose=-1, n_jobs=-1,
            )

    X_train_full = feat_final.fit_transform(train_texts, y_train)
    X_test_full = feat_final.transform(test_texts)
    final_clf = final_clf_factory()
    final_clf.fit(X_train_full, y_train)
    test_probs = final_clf.predict_proba(X_test_full)[:, 1]

    out_path = generate_submission(
        test_ids, test_probs,
        model_name="BL3-Combined" if res_ctrl['mean_acc'] >= res_trt['mean_acc'] else "Tuned-Enhanced",
        threshold=final_threshold,
    )

    print(f"\nBest submission: {out_path}")
    print(f"Using threshold: {final_threshold}")
    print("Done!\n")


if __name__ == "__main__":
    main()

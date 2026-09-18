"""
train.py — Full training pipeline for CP219 Project 1.

Pipeline stages:
  BL1  TF-IDF (1–3 gram) + LogisticRegression + LinearSVC
  BL2  Stylometric features + LightGBM + XGBoost
  BL3  Combined (TF-IDF + stylo) + LightGBM
  BL4  OOF Stacking Ensemble  →  final submission

Run:
    python train.py [--skip-bl2] [--skip-bl3] [--threshold 0.5]

Each model:
  - runs 5-fold Stratified K-Fold cross-validation
  - collects OOF probabilities for stacking (BL4)
  - retrains on the full training set → generates test predictions
  - writes a timestamped submission CSV to submissions/
"""

import argparse
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.data_loader import load_train, load_test, load_sample_submission
from src.features import TfidfFeatures, StyleFeatures, CombinedFeatures, GenerativeCombinedFeatures
from src.models import (
    make_logreg,
    make_linear_svc,
    make_lgbm,
    make_xgb,
    StackingEnsemble,
)
from src.evaluate import (
    cv_evaluate,
    train_and_predict,
    print_results_table,
    generate_submission,
    print_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Class-imbalance ratio (B:A = 6837:3699 ≈ 1.848)
# ---------------------------------------------------------------------------
SCALE_POS_WEIGHT = 6837 / 3699  # ~1.848


# ---------------------------------------------------------------------------
# Feature factory helpers (lambdas so CV can call fresh instance each fold)
# ---------------------------------------------------------------------------

def tfidf_factory():
    return TfidfFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)


def style_factory():
    return StyleFeatures()


def combined_factory():
    return CombinedFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)

def gen_combined_factory():
    return GenerativeCombinedFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    print("\n" + "=" * 60)
    print("  CP219 Project 1 — Human vs. Machine Classification")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    sample_ids = load_sample_submission("sample_submission.csv")

    print(f"  Train: {len(train_texts)} docs  |  A={np.sum(y_train==0)}  B={np.sum(y_train==1)}")
    print(f"  Test : {len(test_texts)} docs")
    print(f"  Scale-pos-weight (B/A): {SCALE_POS_WEIGHT:.4f}")

    all_results = {}
    oof_cols = []          # list of (N,) OOF prob arrays — one per base model
    test_preds_list = []   # list of (3000,) test prob arrays — one per base model

    # ================================================================
    # BASELINE 1 — TF-IDF + LogisticRegression
    # ================================================================
    print("\n[2/6] Baseline 1 — TF-IDF n-gram + Logistic Regression …")
    res_bl1_lr, oof_bl1_lr = cv_evaluate(
        model_factory=make_logreg,
        feature_factory=tfidf_factory,
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="BL1-LogReg",
    )
    all_results["BL1-LogReg"] = res_bl1_lr
    oof_cols.append(oof_bl1_lr)

    print("\n  Re-training on full data for test predictions …")
    test_bl1_lr = train_and_predict(make_logreg, tfidf_factory, train_texts, y_train, test_texts)
    test_preds_list.append(test_bl1_lr)
    generate_submission(test_ids, test_bl1_lr, model_name="BL1-LogReg")

    # --- BL1 LinearSVC ---
    print("\n  Baseline 1b — TF-IDF n-gram + LinearSVC …")
    res_bl1_svc, oof_bl1_svc = cv_evaluate(
        model_factory=make_linear_svc,
        feature_factory=tfidf_factory,
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="BL1-SVC",
    )
    all_results["BL1-SVC"] = res_bl1_svc
    oof_cols.append(oof_bl1_svc)

    print("\n  Re-training on full data for test predictions …")
    test_bl1_svc = train_and_predict(make_linear_svc, tfidf_factory, train_texts, y_train, test_texts)
    test_preds_list.append(test_bl1_svc)
    generate_submission(test_ids, test_bl1_svc, model_name="BL1-SVC")

    # ================================================================
    # BASELINE 2 — Stylometric + LightGBM / XGBoost
    # ================================================================
    if not args.skip_bl2:
        print("\n[3/6] Baseline 2 — Stylometric features + LightGBM …")
        res_bl2_lgbm, oof_bl2_lgbm = cv_evaluate(
            model_factory=lambda: make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT),
            feature_factory=style_factory,
            texts=train_texts,
            y=y_train,
            verbose=True,
            model_name="BL2-LGBM",
        )
        all_results["BL2-LGBM"] = res_bl2_lgbm
        oof_cols.append(oof_bl2_lgbm)

        print("\n  Re-training on full data for test predictions …")
        test_bl2_lgbm = train_and_predict(
            lambda: make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT),
            style_factory, train_texts, y_train, test_texts
        )
        test_preds_list.append(test_bl2_lgbm)
        generate_submission(test_ids, test_bl2_lgbm, model_name="BL2-LGBM")

        print("\n  Baseline 2b — Stylometric features + XGBoost …")
        res_bl2_xgb, oof_bl2_xgb = cv_evaluate(
            model_factory=lambda: make_xgb(scale_pos_weight=SCALE_POS_WEIGHT),
            feature_factory=style_factory,
            texts=train_texts,
            y=y_train,
            verbose=True,
            model_name="BL2-XGB",
        )
        all_results["BL2-XGB"] = res_bl2_xgb
        oof_cols.append(oof_bl2_xgb)

        print("\n  Re-training on full data for test predictions …")
        test_bl2_xgb = train_and_predict(
            lambda: make_xgb(scale_pos_weight=SCALE_POS_WEIGHT),
            style_factory, train_texts, y_train, test_texts
        )
        test_preds_list.append(test_bl2_xgb)
        generate_submission(test_ids, test_bl2_xgb, model_name="BL2-XGB")

    else:
        print("\n[3/6] Skipping BL2 (--skip-bl2 flag set)")

    # ================================================================
    # BASELINE 3 — Combined (TF-IDF + Stylo) + LightGBM
    # ================================================================
    if not args.skip_bl3:
        print("\n[4/6] Baseline 3 — Combined features + LightGBM …")
        res_bl3, oof_bl3 = cv_evaluate(
            model_factory=lambda: make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT),
            feature_factory=combined_factory,
            texts=train_texts,
            y=y_train,
            verbose=True,
            model_name="BL3-Combined",
        )
        all_results["BL3-Combined"] = res_bl3
        oof_cols.append(oof_bl3)

        print("\n  Re-training on full data for test predictions …")
        test_bl3 = train_and_predict(
            lambda: make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT),
            combined_factory, train_texts, y_train, test_texts
        )
        test_preds_list.append(test_bl3)
        generate_submission(test_ids, test_bl3, model_name="BL3-Combined")

    else:
        print("\n[4/6] Skipping BL3 (--skip-bl3 flag set)")

    # ================================================================
    # BASELINE 3b — Generative Combined Features + LightGBM
    # ================================================================
    print("\n[4b/6] Baseline 3b — Generative Combined Features + LightGBM …")
    res_bl3b, oof_bl3b = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT),
        feature_factory=gen_combined_factory,
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="BL3b-Generative",
    )
    all_results["BL3b-Generative"] = res_bl3b
    oof_cols.append(oof_bl3b)

    print("\n  Re-training on full data for test predictions …")
    test_bl3b = train_and_predict(
        lambda: make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT),
        gen_combined_factory, train_texts, y_train, test_texts
    )
    test_preds_list.append(test_bl3b)
    generate_submission(test_ids, test_bl3b, model_name="BL3b-Generative")

    # ================================================================
    # BASELINE 4 — OOF Stacking Ensemble
    # ================================================================
    if len(oof_cols) >= 2:
        print("\n[5/6] Baseline 4 — OOF Stacking Ensemble …")
        oof_meta = np.column_stack(oof_cols)           # (N, n_base_models)
        test_meta = np.column_stack(test_preds_list)   # (3000, n_base_models)

        print(f"  Meta-features shape: {oof_meta.shape}")

        # Simple CV of ensemble using OOF meta-features
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score

        meta_clf = LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs",
            max_iter=1000, random_state=42,
        )
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_ensemble_probs = cross_val_predict(
            meta_clf, oof_meta, y_train, cv=kf, method="predict_proba"
        )[:, 1]
        oof_ensemble_preds = (oof_ensemble_probs >= args.threshold).astype(int)

        from sklearn.metrics import f1_score, precision_score, recall_score
        ens_acc = accuracy_score(y_train, oof_ensemble_preds)
        ens_auc = roc_auc_score(y_train, oof_ensemble_probs)
        ens_f1 = f1_score(y_train, oof_ensemble_preds)
        ens_prec = precision_score(y_train, oof_ensemble_preds, zero_division=0)
        ens_rec = recall_score(y_train, oof_ensemble_preds, zero_division=0)

        print(
            f"  [BL4-Ensemble] OOF — "
            f"Acc={ens_acc:.4f}  Prec={ens_prec:.4f}  Rec={ens_rec:.4f}  "
            f"F1={ens_f1:.4f}  AUC={ens_auc:.4f}"
        )
        all_results["BL4-Ensemble"] = dict(
            mean_acc=ens_acc, std_acc=0.0,
            mean_prec=ens_prec, std_prec=0.0,
            mean_rec=ens_rec, std_rec=0.0,
            mean_f1=ens_f1, std_f1=0.0,
            mean_auc=ens_auc, std_auc=0.0,
        )

        print_confusion_matrix(y_train, oof_ensemble_preds)

        # Retrain meta-learner on all OOF data → predict test
        meta_clf.fit(oof_meta, y_train)
        test_ensemble_probs = meta_clf.predict_proba(test_meta)[:, 1]
        generate_submission(
            test_ids, test_ensemble_probs,
            model_name="BL4-Ensemble",
            threshold=args.threshold,
        )
    else:
        print("\n[5/6] Not enough base models for stacking — writing best single-model submission.")

    # ================================================================
    # Results Summary
    # ================================================================
    print("\n[6/6] Results Summary:")
    print_results_table(all_results)
    print("All submissions written to submissions/")
    print("Done!\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CP219 Project 1 Training Pipeline")
    parser.add_argument(
        "--skip-bl2", action="store_true",
        help="Skip Baseline 2 (Stylometric + GBDT) — faster runs for debugging"
    )
    parser.add_argument(
        "--skip-bl3", action="store_true",
        help="Skip Baseline 3 (Combined + LGBM)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification probability threshold for label B (default: 0.5)"
    )
    args = parser.parse_args()
    main(args)

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_loader import load_train, load_test
from src.features import ChampionFeatures
from src.models import make_lgbm, make_xgb, make_logreg, make_linear_svc
from src.evaluate import cv_evaluate, train_and_predict, generate_submission
from run_submission_duplicate import TestFeatureWrapper, DuplicateFeatures

def main():
    print("============================================================")
    print("  Phase 3: Stacking Ensemble (Champion + Duplicate)")
    print("============================================================")

    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    scale_pos = 1.8483

    # We use the winning feature factory
    feat_factory = lambda: TestFeatureWrapper(DuplicateFeatures())

    print("\n[1/4] Base Model 1: LightGBM")
    _, oof_lgbm = cv_evaluate(lambda: make_lgbm(scale_pos_weight=scale_pos), feat_factory, train_texts, y_train, verbose=False, model_name="LGBM")
    test_lgbm = train_and_predict(lambda: make_lgbm(scale_pos_weight=scale_pos), feat_factory, train_texts, y_train, test_texts)

    print("\n[2/4] Base Model 2: LightGBM (Shallow)")
    # Different hyperparameters for diversity
    lgbm2 = lambda: make_lgbm(n_estimators=400, learning_rate=0.08, num_leaves=31, scale_pos_weight=scale_pos)
    _, oof_lgbm2 = cv_evaluate(lgbm2, feat_factory, train_texts, y_train, verbose=False, model_name="LGBM2")
    test_lgbm2 = train_and_predict(lgbm2, feat_factory, train_texts, y_train, test_texts)

    print("\n[3/4] Stacking OOF Predictions...")
    # Stack the predictions horizontally
    oof_meta = np.column_stack([oof_lgbm, oof_lgbm2])
    test_meta = np.column_stack([test_lgbm, test_lgbm2])

    meta_clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate Meta-Model via CV
    oof_ensemble_probs = cross_val_predict(meta_clf, oof_meta, y_train, cv=kf, method="predict_proba")[:, 1]
    from sklearn.metrics import accuracy_score, roc_auc_score
    ens_acc = accuracy_score(y_train, (oof_ensemble_probs >= 0.5).astype(int))
    ens_auc = roc_auc_score(y_train, oof_ensemble_probs)
    print(f"\n=> Ensemble CV Acc: {ens_acc:.4f}  AUC: {ens_auc:.4f}")

    # Retrain Meta-Model on all OOF data -> Predict on Test
    meta_clf.fit(oof_meta, y_train)
    test_ensemble_probs = meta_clf.predict_proba(test_meta)[:, 1]

    out_path = generate_submission(
        test_ids=test_ids,
        test_probs=test_ensemble_probs,
        sample_sub_path="sample_submission.csv",
        out_dir="submissions",
        model_name="Phase3-Stacking",
        threshold=0.5
    )
    print(f"\nPhase 3 Stacking Submission Ready: {out_path}")

if __name__ == "__main__":
    main()

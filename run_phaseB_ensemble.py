import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_loader import load_train, load_test
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate, train_and_predict, generate_submission

def make_logreg(scale_pos_weight=1.8483):
    # Class weights for LR: {0: 1, 1: 1.8483}
    return LogisticRegression(
        C=1.0, 
        class_weight={0: 1.0, 1: scale_pos_weight},
        solver="liblinear",
        max_iter=500,
        random_state=42
    )

def main():
    print("============================================================")
    print("  Phase B: Stacking Ensemble (LGBM + LogReg)")
    print("============================================================")

    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    scale_pos = 1.8483

    # Use the Pure Champion features
    feat_factory = lambda: ChampionFeatures(max_features=150000)

    print("\n[1/2] Base Model 1: LightGBM (Champion Baseline)")
    _, oof_lgbm = cv_evaluate(lambda: make_lgbm(scale_pos_weight=scale_pos), feat_factory, train_texts, y_train, verbose=False, model_name="LGBM")
    test_lgbm = train_and_predict(lambda: make_lgbm(scale_pos_weight=scale_pos), feat_factory, train_texts, y_train, test_texts)

    print("\n[2/2] Base Model 2: Logistic Regression (Linear Model)")
    _, oof_lr = cv_evaluate(lambda: make_logreg(scale_pos_weight=scale_pos), feat_factory, train_texts, y_train, verbose=False, model_name="LogReg")
    test_lr = train_and_predict(lambda: make_logreg(scale_pos_weight=scale_pos), feat_factory, train_texts, y_train, test_texts)

    print("\n[3/3] Stacking OOF Predictions...")
    oof_meta = np.column_stack([oof_lgbm, oof_lr])
    test_meta = np.column_stack([test_lgbm, test_lr])

    meta_clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate Meta-Model via CV
    oof_ensemble_probs = cross_val_predict(meta_clf, oof_meta, y_train, cv=kf, method="predict_proba")[:, 1]
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Simple Threshold Optimization
    best_thresh, best_acc = 0.5, 0
    for t in np.linspace(0.3, 0.7, 41):
        acc = accuracy_score(y_train, (oof_ensemble_probs >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    ens_auc = roc_auc_score(y_train, oof_ensemble_probs)
    print(f"\n=> Ensemble CV Acc: {best_acc:.4f} (at threshold {best_thresh:.3f})  AUC: {ens_auc:.4f}")
    print(f"Target CV to clear noise floor: 0.9438")
    
    if best_acc > 0.9438:
        print(">>> SUCCESS! Cleared the noise floor! <<<")
    else:
        print(">>> WARNING: Did not clear the noise floor. <<<")

    # Retrain Meta-Model on all OOF data -> Predict on Test
    meta_clf.fit(oof_meta, y_train)
    test_ensemble_probs = meta_clf.predict_proba(test_meta)[:, 1]

    out_path = generate_submission(
        test_ids=test_ids,
        test_probs=test_ensemble_probs,
        sample_sub_path="sample_submission.csv",
        out_dir="submissions",
        model_name="PhaseB-Stacking-LGBM-LR",
        threshold=best_thresh
    )
    print(f"\nPhase B Stacking Submission Ready: {out_path}")

if __name__ == "__main__":
    main()

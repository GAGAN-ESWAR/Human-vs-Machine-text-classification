import numpy as np
import scipy.sparse as sp
import time
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data_loader import load_train, load_test
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import generate_submission

def main():
    print("============================================================")
    print("  Phase 4: Pseudo-Labeling (Confidence thresholds: 0.98, 0.02)")
    print("============================================================")

    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    scale_pos = 1.8483
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    texts_arr = np.array(train_texts, dtype=object)
    
    oof_probs = np.zeros(len(y_train), dtype=np.float32)
    
    print("\nRunning CV with in-fold Pseudo-Labeling...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(texts_arr, y_train), 1):
        X_tr_raw = texts_arr[train_idx].tolist()
        y_tr = y_train[train_idx]
        X_val_raw = texts_arr[val_idx].tolist()
        y_val = y_train[val_idx]
        
        # Step 1: Base Model Feature Extraction & Training
        feat = ChampionFeatures(max_features=150000)
        X_tr_sparse = feat.fit_transform(X_tr_raw, y_tr)
        
        clf_base = make_lgbm(scale_pos_weight=scale_pos)
        clf_base.fit(X_tr_sparse, y_tr)
        
        # Step 2: Predict on Unseen Test Data to get Pseudo-labels
        X_test_sparse = feat.transform(test_texts)
        test_base_probs = clf_base.predict_proba(X_test_sparse)[:, 1]
        
        # Step 3: Filter High-Confidence Predictions
        high_conf_idx = np.where((test_base_probs > 0.98) | (test_base_probs < 0.02))[0]
        pseudo_texts = [test_texts[i] for i in high_conf_idx]
        pseudo_labels = (test_base_probs[high_conf_idx] >= 0.5).astype(int)
        
        # Step 4: Augment Training Data and Retrain
        aug_texts = X_tr_raw + pseudo_texts
        aug_y = np.concatenate([y_tr, pseudo_labels])
        
        feat_aug = ChampionFeatures(max_features=150000)
        X_aug_sparse = feat_aug.fit_transform(aug_texts, aug_y)
        
        clf_aug = make_lgbm(scale_pos_weight=scale_pos)
        clf_aug.fit(X_aug_sparse, aug_y)
        
        # Step 5: Evaluate on Fold Validation Set
        X_val_sparse = feat_aug.transform(X_val_raw)
        probs = clf_aug.predict_proba(X_val_sparse)[:, 1]
        oof_probs[val_idx] = probs
        
        acc = accuracy_score(y_val, (probs >= 0.5).astype(int))
        auc = roc_auc_score(y_val, probs)
        print(f"  [Fold {fold}/5] Pseudo-labels added: {len(high_conf_idx)} | Acc={acc:.4f}  AUC={auc:.4f}")

    # Aggregate
    final_acc = accuracy_score(y_train, (oof_probs >= 0.5).astype(int))
    final_auc = roc_auc_score(y_train, oof_probs)
    print(f"\n=> Pseudo-Labeling CV Acc: {final_acc:.4f}  AUC: {final_auc:.4f}")

    print("\nTraining Final Pseudo-Label Model on 100% of data...")
    # Base training
    feat_final = ChampionFeatures(max_features=150000)
    X_train_sparse = feat_final.fit_transform(train_texts, y_train)
    clf_final = make_lgbm(scale_pos_weight=scale_pos)
    clf_final.fit(X_train_sparse, y_train)
    
    # Test predictions for pseudo labels
    X_test_sparse = feat_final.transform(test_texts)
    test_final_probs = clf_final.predict_proba(X_test_sparse)[:, 1]
    
    high_conf_idx = np.where((test_final_probs > 0.98) | (test_final_probs < 0.02))[0]
    pseudo_texts = [test_texts[i] for i in high_conf_idx]
    pseudo_labels = (test_final_probs[high_conf_idx] >= 0.5).astype(int)
    
    print(f"Final model adding {len(high_conf_idx)} high-confidence test samples.")
    
    # Augmented training
    aug_texts = train_texts + pseudo_texts
    aug_y = np.concatenate([y_train, pseudo_labels])
    
    feat_aug_final = ChampionFeatures(max_features=150000)
    X_aug_sparse_final = feat_aug_final.fit_transform(aug_texts, aug_y)
    clf_aug_final = make_lgbm(scale_pos_weight=scale_pos)
    clf_aug_final.fit(X_aug_sparse_final, aug_y)
    
    # Final Test predictions
    X_test_sparse_final = feat_aug_final.transform(test_texts)
    test_aug_probs = clf_aug_final.predict_proba(X_test_sparse_final)[:, 1]

    out_path = generate_submission(
        test_ids=test_ids,
        test_probs=test_aug_probs,
        sample_sub_path="sample_submission.csv",
        out_dir="submissions",
        model_name="Phase4-PseudoLabeling",
        threshold=0.5
    )
    print(f"\nPhase 4 Pseudo-Labeling Submission Ready: {out_path}")

if __name__ == "__main__":
    main()

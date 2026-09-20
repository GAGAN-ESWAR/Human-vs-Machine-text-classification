import numpy as np
from src.data_loader import load_train, load_test, load_sample_submission
from src.features import GenerativeCombinedFeatures
from src.models import make_lgbm
from src.evaluate import generate_submission
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import time
import os
import warnings

warnings.filterwarnings("ignore")

def gen_factory():
    return GenerativeCombinedFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)

def main():
    print("============================================================")
    print("  Pseudo-Labeling Experiment (Semi-Supervised Learning) ")
    print("============================================================")
    
    # 1. Load Data
    _, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    SCALE_POS_WEIGHT = 1.8483
    
    print("\n[1/4] Phase 1: Training initial model on original data...")
    feat_extractor_init = gen_factory()
    X_train_init = feat_extractor_init.fit_transform(train_texts, y_train)
    
    clf_init = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
    clf_init.fit(X_train_init, y_train)
    
    print("\n[2/4] Phase 2: Generating Pseudo-Labels on Test Set...")
    X_test_init = feat_extractor_init.transform(test_texts)
    test_probs = clf_init.predict_proba(X_test_init)[:, 1]
    
    # Define confidence thresholds
    CONFIDENT_MACHINE_THRESHOLD = 0.95
    CONFIDENT_HUMAN_THRESHOLD = 0.05
    
    confident_machine_idx = np.where(test_probs >= CONFIDENT_MACHINE_THRESHOLD)[0]
    confident_human_idx = np.where(test_probs <= CONFIDENT_HUMAN_THRESHOLD)[0]
    
    pseudo_test_texts = [test_texts[i] for i in confident_machine_idx] + \
                        [test_texts[i] for i in confident_human_idx]
    
    pseudo_test_y = np.array([1]*len(confident_machine_idx) + [0]*len(confident_human_idx))
    
    print(f"  Total Test Documents: {len(test_texts)}")
    print(f"  Highly Confident Machine (> {CONFIDENT_MACHINE_THRESHOLD}): {len(confident_machine_idx)}")
    print(f"  Highly Confident Human   (< {CONFIDENT_HUMAN_THRESHOLD}): {len(confident_human_idx)}")
    print(f"  Total Pseudo-Labeled Data: {len(pseudo_test_y)}")
    
    if len(pseudo_test_y) == 0:
        print("No confident predictions found. Exiting.")
        return
        
    print("\n[3/4] Phase 3: Fair Cross-Validation with Pseudo-Labels...")
    # Standard CV splits on the ORIGINAL train data
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_texts_arr = np.array(train_texts, dtype=object)
    
    fold_accs = []
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_texts_arr, y_train), 1):
        # 1. Base training fold (from ground truth)
        base_train_texts = train_texts_arr[train_idx].tolist()
        base_train_y = y_train[train_idx]
        
        # 2. Append Pseudo-Labeled data to training fold!
        combined_train_texts = base_train_texts + pseudo_test_texts
        combined_train_y = np.concatenate([base_train_y, pseudo_test_y])
        
        # 3. Validation fold remains STRICTLY ground truth
        val_texts = train_texts_arr[val_idx].tolist()
        val_y = y_train[val_idx]
        
        # Train & Evaluate
        feat_cv = gen_factory()
        X_tr_cv = feat_cv.fit_transform(combined_train_texts, combined_train_y)
        X_val_cv = feat_cv.transform(val_texts)
        
        clf_cv = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
        clf_cv.fit(X_tr_cv, combined_train_y)
        
        val_probs = clf_cv.predict_proba(X_val_cv)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        acc = accuracy_score(val_y, val_preds)
        auc = roc_auc_score(val_y, val_probs)
        fold_accs.append(acc)
        fold_aucs.append(auc)
        
        print(f"  Fold {fold}/5 — Acc: {acc:.4f}  AUC: {auc:.4f}")
        
    mean_acc = np.mean(fold_accs)
    print(f"  MEAN CV SCORE (Pseudo-Labeled): Acc={mean_acc:.4f}  AUC={np.mean(fold_aucs):.4f}")
    print(f"  (Original Baseline was ~0.9413 Acc)")
    
    print("\n[4/4] Phase 4: Final Training and Submission...")
    # Train on ALL training data + pseudo test data
    combined_full_texts = train_texts + pseudo_test_texts
    combined_full_y = np.concatenate([y_train, pseudo_test_y])
    
    final_feat = gen_factory()
    X_train_final = final_feat.fit_transform(combined_full_texts, combined_full_y)
    
    clf_final = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
    clf_final.fit(X_train_final, combined_full_y)
    
    X_test_final = final_feat.transform(test_texts)
    final_test_probs = clf_final.predict_proba(X_test_final)[:, 1]
    
    generate_submission(test_ids, final_test_probs, model_name="Pseudo-Generative")

if __name__ == "__main__":
    main()

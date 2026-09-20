import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

from src.data_loader import load_train, load_test, load_sample_submission
from src.features import GenerativeCombinedFeatures
from src.models import make_lgbm
from src.evaluate import generate_submission

warnings.filterwarnings("ignore")

def heaps_exponent(tokens):
    n = len(tokens)
    if n < 10:
        return 0.0
    seen = set()
    growth = []
    for i, t in enumerate(tokens, 1):
        seen.add(t)
        if i % max(1, n // 20) == 0:  # sample ~20 points along the sequence
            growth.append((i, len(seen)))
    if len(growth) < 3:
        return 0.0
    log_pos = np.log([g[0] for g in growth])
    log_vocab = np.log([g[1] for g in growth])
    beta, _ = np.polyfit(log_pos, log_vocab, 1)
    return beta

def main():
    print("============================================================")
    print("  Testing Heaps' Law Feature (Standalone Experiment) ")
    print("============================================================")
    
    # 1. Load Data
    _, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    SCALE_POS_WEIGHT = 1.8483
    
    print("\n[1/3] Extracting Baseline GenerativeCombinedFeatures...")
    feat_extractor = GenerativeCombinedFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)
    X_base = feat_extractor.fit_transform(train_texts, y_train)
    
    print("\n[2/3] Extracting Heaps' Law Feature...")
    heaps_feats = np.zeros((len(train_texts), 1), dtype=np.float32)
    for i, tokens in enumerate(train_texts):
        heaps_feats[i, 0] = heaps_exponent(tokens)
        
    # Combine features
    X_combined = sp.hstack([X_base, sp.csr_matrix(heaps_feats)], format="csr")
    
    print("\n[3/3] Running 5-Fold Cross Validation...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accs = []
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined, y_train), 1):
        X_tr = X_combined[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_combined[val_idx]
        y_val = y_train[val_idx]
        
        clf = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
        clf.fit(X_tr, y_tr)
        
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)
        fold_accs.append(acc)
        fold_aucs.append(auc)
        
        print(f"  Fold {fold}/5 — Acc: {acc:.4f}  AUC: {auc:.4f}")
        
    mean_acc = np.mean(fold_accs)
    print(f"\n  MEAN CV SCORE (With Heaps' Law): Acc={mean_acc:.4f}  AUC={np.mean(fold_aucs):.4f}")
    print(f"  (Original Baseline was ~0.9413 Acc)")
    
    print("\n[4/4] Phase 4: Final Training and Submission...")
    clf_final = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
    clf_final.fit(X_combined, y_train)
    
    print("  Extracting features for Test Set...")
    X_test_base = feat_extractor.transform(test_texts)
    
    heaps_test = np.zeros((len(test_texts), 1), dtype=np.float32)
    for i, tokens in enumerate(test_texts):
        heaps_test[i, 0] = heaps_exponent(tokens)
        
    X_test_combined = sp.hstack([X_test_base, sp.csr_matrix(heaps_test)], format="csr")
    
    final_test_probs = clf_final.predict_proba(X_test_combined)[:, 1]
    generate_submission(test_ids, final_test_probs, model_name="HeapsLaw-Generative")

if __name__ == "__main__":
    main()

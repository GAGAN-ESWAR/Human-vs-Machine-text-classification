import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD
import warnings

from src.data_loader import load_train, load_test
from src.features import TfidfFeatures, ExtendedStyleFeatures, GenerativeStyleFeatures
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
        if i % max(1, n // 20) == 0:
            growth.append((i, len(seen)))
    if len(growth) < 3:
        return 0.0
    log_pos = np.log([g[0] for g in growth])
    log_vocab = np.log([g[1] for g in growth])
    beta, _ = np.polyfit(log_pos, log_vocab, 1)
    return beta

def main():
    print("============================================================")
    print("  Testing Dimensionality Reduction (TruncatedSVD) ")
    print("============================================================")
    
    # 1. Load Data
    _, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    SCALE_POS_WEIGHT = 1.8483
    N_COMPONENTS = 300
    
    # Initialize individual extractors
    tfidf_extractor = TfidfFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)
    style_extractor = ExtendedStyleFeatures()
    gen_extractor = GenerativeStyleFeatures(alpha=0.1, use_trigram=True, vocab_size=18438)
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    
    print("\n[1/3] Extracting and Compressing Features...")
    print("  -> Fitting TF-IDF and SVD...")
    X_tfidf_sparse = tfidf_extractor.fit_transform(train_texts)
    X_tfidf_dense = svd.fit_transform(X_tfidf_sparse)
    
    print("  -> Extracting Style & Generative Features...")
    X_style = style_extractor.fit_transform(train_texts)
    X_gen = gen_extractor.fit_transform(train_texts, y_train)
    
    print("  -> Calculating Heaps' Law...")
    heaps_feats = np.zeros((len(train_texts), 1), dtype=np.float32)
    for i, tokens in enumerate(train_texts):
        heaps_feats[i, 0] = heaps_exponent(tokens)
        
    # Combine all into one DENSE matrix
    X_combined = np.hstack([X_tfidf_dense, X_style, X_gen, heaps_feats])
    
    print(f"  Final Training Matrix Shape: {X_combined.shape}")
    
    print("\n[2/3] Running 5-Fold Cross Validation...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accs = []
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined, y_train), 1):
        X_tr = X_combined[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_combined[val_idx]
        y_val = y_train[val_idx]
        
        # We can use the same LGBM configuration. Since it's all dense, it should be fast.
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
    print(f"\n  MEAN CV SCORE (SVD + Heaps): Acc={mean_acc:.4f}  AUC={np.mean(fold_aucs):.4f}")
    
    print("\n[3/3] Final Training and Submission Generation...")
    clf_final = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
    clf_final.fit(X_combined, y_train)
    
    # Process Test Set
    print("  Extracting features for Test Set...")
    X_test_tfidf_sparse = tfidf_extractor.transform(test_texts)
    X_test_tfidf_dense = svd.transform(X_test_tfidf_sparse)
    X_test_style = style_extractor.transform(test_texts)
    X_test_gen = gen_extractor.transform(test_texts)
    
    heaps_test = np.zeros((len(test_texts), 1), dtype=np.float32)
    for i, tokens in enumerate(test_texts):
        heaps_test[i, 0] = heaps_exponent(tokens)
        
    X_test_combined = np.hstack([X_test_tfidf_dense, X_test_style, X_test_gen, heaps_test])
    
    final_test_probs = clf_final.predict_proba(X_test_combined)[:, 1]
    generate_submission(test_ids, final_test_probs, model_name="SVD-HeapsLaw")

if __name__ == "__main__":
    main()

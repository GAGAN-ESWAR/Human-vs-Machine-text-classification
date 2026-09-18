import json
import time
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb

from src.features_generative import GenerativeStyleFeatures
from src.data_loader import load_train
from src.features import ExtendedStyleFeatures

RANDOM_STATE = 42
N_SPLITS = 5
TFIDF_MAX_FEATURES = 150_000

def tokens_to_text(tokens):
    return [" ".join(map(str, t)) for t in tokens]

def build_stylometric_features(tokens):
    style = ExtendedStyleFeatures()
    return style.fit_transform(tokens)

def main():
    t0 = time.time()

    print("Loading data...")
    train_ids, train_tokens, y = load_train("train.json")
    
    train_text = tokens_to_text(train_tokens)

    print(f"{len(train_tokens)} training docs, "
          f"class balance: A={sum(y == 0)}, B={sum(y == 1)}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))

    fold_times = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_text, y)):
        fold_t0 = time.time()
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

        y_tr, y_val = y[tr_idx], y[val_idx]
        tokens_tr = [train_tokens[i] for i in tr_idx]
        tokens_val = [train_tokens[i] for i in val_idx]
        text_tr = [train_text[i] for i in tr_idx]
        text_val = [train_text[i] for i in val_idx]

        # 1. TF-IDF
        tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
        )
        X_tfidf_tr = tfidf.fit_transform(text_tr)
        X_tfidf_val = tfidf.transform(text_val)

        # 2. Stylometric features
        X_style_tr = build_stylometric_features(tokens_tr)
        X_style_val = build_stylometric_features(tokens_val)

        # 3. Generative features
        gen = GenerativeStyleFeatures(alpha=0.1, use_trigram=True, vocab_size=18438)
        X_gen_tr = gen.fit_transform(tokens_tr, y=y_tr)
        X_gen_val = gen.transform(tokens_val)

        # Combine
        X_tr = sp.hstack([
            X_tfidf_tr,
            sp.csr_matrix(X_style_tr),
            sp.csr_matrix(X_gen_tr),
        ]).tocsr()
        X_val = sp.hstack([
            X_tfidf_val,
            sp.csr_matrix(X_style_val),
            sp.csr_matrix(X_gen_val),
        ]).tocsr()

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            colsample_bytree=0.7,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=1.848,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds

        fold_acc = accuracy_score(y_val, val_preds)
        fold_auc = roc_auc_score(y_val, val_probs)
        fold_time = time.time() - fold_t0
        fold_times.append(fold_time)
        print(f"Fold {fold + 1} acc={fold_acc:.4f} auc={fold_auc:.4f} "
              f"time={fold_time:.1f}s")

    overall_acc = accuracy_score(y, oof_preds)
    overall_auc = roc_auc_score(y, oof_probs)

    best_t, best_acc = 0.5, overall_acc
    for t in np.arange(0.30, 0.71, 0.01):
        acc = accuracy_score(y, (oof_probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t

    print("\n===== RESULTS =====")
    print(f"OOF Accuracy (t=0.5): {overall_acc:.4f}")
    print(f"OOF ROC-AUC:          {overall_auc:.4f}")
    print(f"Best threshold:       {best_t:.2f} -> acc={best_acc:.4f}")
    print(f"Total time:           {time.time() - t0:.1f}s "
          f"(avg {np.mean(fold_times):.1f}s/fold)")

if __name__ == "__main__":
    main()

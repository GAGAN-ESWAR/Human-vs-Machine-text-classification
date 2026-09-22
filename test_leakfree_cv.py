"""
test_leakfree_cv.py — Leak-Free Cross-Validation

Key differences from previous scripts:
1. TF-IDF is fit INSIDE each fold (on training fold only).
2. GenerativeStyleFeatures is fit INSIDE each fold (it uses y).
3. ExtendedStyleFeatures and Heaps' Law are per-document, so no leakage.
4. No pseudo-labeling — this gives us an honest, uncontaminated CV baseline.
5. Final submission is trained on all data (no CV leakage concern there).
"""

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
import time

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


def extract_features(texts, tfidf, style, gen, fit=False, y=None):
    """
    Extract and horizontally stack all features.
    If fit=True, fits the extractors on `texts` first.
    """
    if fit:
        X_tfidf = tfidf.fit_transform(texts)
        X_style = style.fit_transform(texts)
        X_gen = gen.fit_transform(texts, y)
    else:
        X_tfidf = tfidf.transform(texts)
        X_style = style.transform(texts)
        X_gen = gen.transform(texts)

    # Heaps' Law — pure per-document, no cross-doc leakage
    heaps = np.zeros((len(texts), 1), dtype=np.float32)
    for i, tokens in enumerate(texts):
        heaps[i, 0] = heaps_exponent(tokens)

    return sp.hstack([X_tfidf, sp.csr_matrix(X_style),
                       sp.csr_matrix(X_gen), sp.csr_matrix(heaps)],
                      format="csr")


def main():
    print("============================================================")
    print("  Leak-Free Cross-Validation (No Pseudo-Labels)")
    print("============================================================")

    _, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    SCALE_POS_WEIGHT = 1.8483

    train_texts_arr = np.array(train_texts, dtype=object)

    print(f"\n  Training samples: {len(train_texts)}")
    print(f"  Test samples:     {len(test_texts)}")

    # ---- Leak-Free 5-Fold CV ----
    print("\n[1/2] Running Leak-Free 5-Fold CV...")
    print("  (TF-IDF + Gen features are fit INSIDE each fold)\n")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accs = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_texts_arr, y_train), 1):
        t0 = time.time()

        # Split texts and labels — strictly by fold
        fold_train_texts = train_texts_arr[train_idx].tolist()
        fold_train_y = y_train[train_idx]
        fold_val_texts = train_texts_arr[val_idx].tolist()
        fold_val_y = y_train[val_idx]

        # Fresh extractors for each fold — this is the key fix
        tfidf = TfidfFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)
        style = ExtendedStyleFeatures()
        gen = GenerativeStyleFeatures(alpha=0.1, use_trigram=True, vocab_size=18438)

        # Fit on training fold ONLY, then transform both
        X_tr = extract_features(fold_train_texts, tfidf, style, gen,
                                fit=True, y=fold_train_y)
        X_val = extract_features(fold_val_texts, tfidf, style, gen,
                                 fit=False)

        # Train and evaluate
        clf = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
        clf.fit(X_tr, fold_train_y)

        val_probs = clf.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        acc = accuracy_score(fold_val_y, val_preds)
        auc = roc_auc_score(fold_val_y, val_probs)
        fold_accs.append(acc)
        fold_aucs.append(auc)

        elapsed = time.time() - t0
        print(f"  Fold {fold}/5 — Acc: {acc:.4f}  AUC: {auc:.4f}  ({elapsed:.1f}s)")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_auc = np.mean(fold_aucs)
    print(f"\n  ================================================")
    print(f"  LEAK-FREE CV: Acc={mean_acc:.4f} +/- {std_acc:.4f}  AUC={mean_auc:.4f}")
    print(f"  ================================================")

    # ---- Final Training & Submission (on ALL data) ----
    print("\n[2/2] Final Training on ALL data + Submission...")
    tfidf_final = TfidfFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)
    style_final = ExtendedStyleFeatures()
    gen_final = GenerativeStyleFeatures(alpha=0.1, use_trigram=True, vocab_size=18438)

    X_train_all = extract_features(train_texts, tfidf_final, style_final, gen_final,
                                   fit=True, y=y_train)

    clf_final = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
    clf_final.fit(X_train_all, y_train)

    X_test = extract_features(test_texts, tfidf_final, style_final, gen_final,
                              fit=False)

    test_probs = clf_final.predict_proba(X_test)[:, 1]
    generate_submission(test_ids, test_probs, model_name="LeakFree-HeapsLaw")


if __name__ == "__main__":
    main()

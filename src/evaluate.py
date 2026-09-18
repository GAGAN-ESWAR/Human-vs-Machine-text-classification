"""
evaluate.py — Cross-validation harness, metrics, confusion matrix, submission generation.

CV protocol: 5-fold StratifiedKFold, shuffle=True, random_state=42.
All metrics tracked: Accuracy, Precision, Recall, F1, ROC-AUC (per class + macro).
"""

import csv
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Callable

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CV configuration
# ---------------------------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42


def _make_kfold() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Core CV harness
# ---------------------------------------------------------------------------

def cv_evaluate(
    model_factory: Callable,
    feature_factory: Callable,
    texts: List[List[int]],
    y: np.ndarray,
    verbose: bool = True,
    model_name: str = "model",
) -> Tuple[dict, np.ndarray]:
    """
    Run 5-fold stratified CV.

    Parameters
    ----------
    model_factory   : callable with no args → sklearn-compatible classifier
    feature_factory : callable with no args → fitted on train fold, transforms both
    texts           : raw token sequences
    y               : integer labels (0/1)
    verbose         : print fold-by-fold results
    model_name      : string used in logging

    Returns
    -------
    results : dict with mean/std of all metrics
    oof_probs : np.ndarray of shape (N,) — OOF probability of class 1
    """
    kf = _make_kfold()
    texts_arr = np.array(texts, dtype=object)

    oof_probs = np.zeros(len(y), dtype=np.float32)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(texts_arr, y), 1):
        X_tr_raw = texts_arr[train_idx].tolist()
        X_val_raw = texts_arr[val_idx].tolist()
        y_tr, y_val = y[train_idx], y[val_idx]

        # --- Featurize (fit only on train fold) ---
        feat = feature_factory()
        X_tr = feat.fit_transform(X_tr_raw, y_tr)
        X_val = feat.transform(X_val_raw)

        # --- Train ---
        clf = model_factory()
        clf.fit(X_tr, y_tr)

        # --- Predict ---
        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        oof_probs[val_idx] = probs

        # --- Metrics ---
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        auc = roc_auc_score(y_val, probs)
        fold_results.append(dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc))

        if verbose:
            print(
                f"  [{model_name}] fold {fold}/{N_SPLITS} — "
                f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
                f"F1={f1:.4f}  AUC={auc:.4f}"
            )

    # --- Aggregate ---
    results = {}
    for metric in ["acc", "prec", "rec", "f1", "auc"]:
        vals = [fr[metric] for fr in fold_results]
        results[f"mean_{metric}"] = float(np.mean(vals))
        results[f"std_{metric}"] = float(np.std(vals))

    if verbose:
        print(
            f"  [{model_name}] MEAN — "
            f"Acc={results['mean_acc']:.4f}±{results['std_acc']:.4f}  "
            f"AUC={results['mean_auc']:.4f}±{results['std_auc']:.4f}"
        )

    return results, oof_probs


# ---------------------------------------------------------------------------
# Retrain on full training set + produce test predictions
# ---------------------------------------------------------------------------

def train_and_predict(
    model_factory: Callable,
    feature_factory: Callable,
    train_texts: List[List[int]],
    y_train: np.ndarray,
    test_texts: List[List[int]],
) -> np.ndarray:
    """
    Fit on all training data, return probability of class 1 for test set.
    """
    feat = feature_factory()
    X_train = feat.fit_transform(train_texts, y_train)
    X_test = feat.transform(test_texts)

    clf = model_factory()
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(all_results: dict):
    """Pretty-print a comparison table of all models."""
    print("\n" + "=" * 75)
    print(f"{'Model':<20} {'Acc':>8} {'±':>6} {'F1':>8} {'AUC':>8}")
    print("-" * 75)
    for name, res in all_results.items():
        print(
            f"{name:<20} "
            f"{res['mean_acc']:>8.4f} "
            f"{res['std_acc']:>6.4f} "
            f"{res['mean_f1']:>8.4f} "
            f"{res['mean_auc']:>8.4f}"
        )
    print("=" * 75 + "\n")


# ---------------------------------------------------------------------------
# Submission generation with full sanity checks
# ---------------------------------------------------------------------------

def generate_submission(
    test_ids: np.ndarray,
    test_probs: np.ndarray,
    sample_sub_path: str = "sample_submission.csv",
    out_dir: str = "submissions",
    model_name: str = "ensemble",
    threshold: float = 0.5,
) -> str:
    """
    Generate a valid Kaggle submission CSV.

    Parameters
    ----------
    test_ids       : ordered test IDs (must match sample_submission order)
    test_probs     : predicted probability of class B (machine) for each test doc
    sample_sub_path: path to sample_submission.csv for ID-order validation
    out_dir        : directory to write submission to
    model_name     : prefix used in filename
    threshold      : classification threshold (default 0.5)

    Returns
    -------
    out_path : str — path to written submission CSV
    """
    # Load sample submission for ID validation
    sample_sub = pd.read_csv(sample_sub_path)

    # Build submission dataframe
    preds = np.where(test_probs >= threshold, "B", "A")
    sub = pd.DataFrame({"id": test_ids, "label": preds})

    # ---- MANDATORY SANITY CHECKS ----
    assert sub.shape == (3000, 2), f"Expected shape (3000, 2), got {sub.shape}"
    assert set(sub.columns) == {"id", "label"}, f"Invalid columns: {sub.columns.tolist()}"
    assert set(sub["label"].unique()).issubset({"A", "B"}), \
        f"Labels contain values other than A/B: {sub['label'].unique()}"
    assert (sub["id"].values == sample_sub["id"].values).all(), \
        "Test IDs do not match sample submission order"

    # Write
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"submission_{model_name}_{timestamp}.csv")
    sub.to_csv(out_path, index=False)

    label_counts = sub["label"].value_counts().to_dict()
    print(f"\n[OK] Submission written to: {out_path}")
    print(f"     Label distribution: A={label_counts.get('A',0)}, B={label_counts.get('B',0)}")
    print(f"     All sanity checks passed.\n")

    return out_path


# ---------------------------------------------------------------------------
# Confusion matrix (text-based, no matplotlib dependency required)
# ---------------------------------------------------------------------------

def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=("A", "B")):
    """Print a text confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"         Pred {labels[0]}   Pred {labels[1]}")
    print(f"True {labels[0]}   {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"True {labels[1]}   {cm[1,0]:>6}   {cm[1,1]:>6}")
    print()

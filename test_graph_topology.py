import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

from src.data_loader import load_train, load_test
from src.features import GenerativeCombinedFeatures
from src.models import make_lgbm
from src.evaluate import generate_submission

warnings.filterwarnings("ignore")

def extract_graph_features(tokens):
    n = len(tokens)
    if n < 2:
        return 0.0, 0.0
        
    # Build graph edges
    edges = set()
    out_degrees = {}
    
    for i in range(n - 1):
        u = tokens[i]
        v = tokens[i+1]
        edges.add((u, v))
        if u not in out_degrees:
            out_degrees[u] = set()
        out_degrees[u].add(v)
        
    num_nodes = len(set(tokens))
    num_edges = len(edges)
    
    # Graph Density: actual edges / possible edges
    if num_nodes > 0:
        density = num_edges / (num_nodes * num_nodes)
    else:
        density = 0.0
        
    # Graph Out-Degree: average out-degree of nodes that have out-edges
    if out_degrees:
        avg_out_degree = sum(len(outs) for outs in out_degrees.values()) / len(out_degrees)
    else:
        avg_out_degree = 0.0
        
    return density, avg_out_degree

def main():
    print("============================================================")
    print("  Testing Graph Topology Features (Standalone Experiment) ")
    print("============================================================")
    
    # 1. Load Data
    _, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    SCALE_POS_WEIGHT = 1.8483
    
    print("\n[1/3] Extracting Baseline GenerativeCombinedFeatures...")
    feat_extractor = GenerativeCombinedFeatures(ngram_range=(1, 3), max_features=150_000, sublinear_tf=True)
    X_base = feat_extractor.fit_transform(train_texts, y_train)
    
    print("\n[2/3] Extracting Graph Topology Features (Density, Avg Out-Degree)...")
    graph_feats = np.zeros((len(train_texts), 2), dtype=np.float32)
    for i, tokens in enumerate(train_texts):
        dens, out_deg = extract_graph_features(tokens)
        graph_feats[i, 0] = dens
        graph_feats[i, 1] = out_deg
        
    # Combine features
    X_combined = sp.hstack([X_base, sp.csr_matrix(graph_feats)], format="csr")
    
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
    print(f"\n  MEAN CV SCORE (With Graph Topology): Acc={mean_acc:.4f}  AUC={np.mean(fold_aucs):.4f}")
    print(f"  (Original Baseline was ~0.9413 Acc)")
    
    print("\n[4/4] Phase 4: Final Training and Submission...")
    clf_final = make_lgbm(scale_pos_weight=SCALE_POS_WEIGHT)
    clf_final.fit(X_combined, y_train)
    
    print("  Extracting features for Test Set...")
    X_test_base = feat_extractor.transform(test_texts)
    
    graph_test = np.zeros((len(test_texts), 2), dtype=np.float32)
    for i, tokens in enumerate(test_texts):
        dens, out_deg = extract_graph_features(tokens)
        graph_test[i, 0] = dens
        graph_test[i, 1] = out_deg
        
    X_test_combined = sp.hstack([X_test_base, sp.csr_matrix(graph_test)], format="csr")
    
    final_test_probs = clf_final.predict_proba(X_test_combined)[:, 1]
    generate_submission(test_ids, final_test_probs, model_name="GraphTopology-Generative")

if __name__ == "__main__":
    main()

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

from src.data_loader import load_train, load_test
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate, train_and_predict, generate_submission

class GraphTopologyFeatures(BaseEstimator, TransformerMixin):
    """
    Computes Word-Transition Graph Topology features per document.
    Humans have different vocabulary switching dynamics than LLMs.
    """
    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), 4), dtype=np.float32)
        for i, tokens in enumerate(texts):
            if len(tokens) < 2:
                continue
            
            nodes = set(tokens)
            n_nodes = len(nodes)
            if n_nodes < 2:
                continue
                
            edges = set(zip(tokens, tokens[1:]))
            n_edges = len(edges)
            
            # 1. Graph Density
            max_possible_edges = n_nodes * (n_nodes - 1)
            X[i, 0] = n_edges / max_possible_edges
            
            # 2. Average Out-Degree
            X[i, 1] = n_edges / n_nodes
            
            # Out-degree distribution
            out_degrees = defaultdict(set)
            for t1, t2 in zip(tokens, tokens[1:]):
                out_degrees[t1].add(t2)
            
            degree_counts = [len(targets) for targets in out_degrees.values()]
            
            # 3. Max Out-Degree
            X[i, 2] = max(degree_counts) if degree_counts else 0
            
            # 4. Out-Degree Variance
            X[i, 3] = np.var(degree_counts) if len(degree_counts) > 1 else 0
            
        return X

class ChampionPlusGraph(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.champ = ChampionFeatures(max_features=150000)
        self.graph = GraphTopologyFeatures()
        
    def fit(self, texts, y=None):
        self.champ.fit(texts, y)
        return self
        
    def transform(self, texts):
        X_champ = self.champ.transform(texts)
        X_c = self.graph.transform(texts)
        return sp.hstack([X_champ, sp.csr_matrix(X_c)], format="csr")
        
    def fit_transform(self, texts, y=None):
        X_champ = self.champ.fit_transform(texts, y)
        X_c = self.graph.transform(texts)
        return sp.hstack([X_champ, sp.csr_matrix(X_c)], format="csr")

def main():
    print("============================================================")
    print("  Phase C: Word-Transition Graph Topology Features")
    print("============================================================")
    
    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    scale_pos = 1.8483
    
    print("Evaluating Champion + Graph Topology Features via 5-Fold CV...")
    results, oof_preds = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=scale_pos),
        feature_factory=lambda: ChampionPlusGraph(),
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="PhaseC-Graph",
        random_state=42
    )
    
    cv_mean = results['mean_acc']
    target = 0.9438
    print(f"\n=> Phase C Final CV Accuracy: {cv_mean:.4f}")
    print(f"=> Noise Floor Hurdle to Beat: {target:.4f}")
    
    if cv_mean > target:
        print("\n>>> SUCCESS! We cleared the noise floor! <<<")
        print("Training final model on 100% of data...")
        test_preds = train_and_predict(
            model_factory=lambda: make_lgbm(scale_pos_weight=scale_pos),
            feature_factory=lambda: ChampionPlusGraph(),
            train_texts=train_texts,
            y_train=y_train,
            test_texts=test_texts
        )
        out_path = generate_submission(
            test_ids=test_ids,
            test_probs=test_preds,
            sample_sub_path="sample_submission.csv",
            out_dir="submissions",
            model_name="PhaseC-GraphTopology",
            threshold=0.5
        )
        print(f"Submission saved to: {out_path}")
    else:
        print("\n>>> WARNING: Did not clear the noise floor. Feature rejected. <<<")

if __name__ == "__main__":
    main()

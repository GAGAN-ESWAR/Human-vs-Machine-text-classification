import numpy as np
import scipy.sparse as sp
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_loader import load_train, load_test
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate, train_and_predict, generate_submission

class ShapeStatsFeatures(BaseEstimator, TransformerMixin):
    """
    Computes skewness, kurtosis, and repeated 3-gram rates.
    """
    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), 3), dtype=np.float32)
        for i, tokens in enumerate(texts):
            if len(tokens) < 3:
                continue
            
            # 1 & 2: Shape Stats
            arr = np.array(tokens)
            X[i, 0] = skew(arr)
            X[i, 1] = kurtosis(arr)
            
            # 3: Repeated 3-gram rate
            from collections import Counter
            trigrams = zip(tokens, tokens[1:], tokens[2:])
            counts = Counter(trigrams)
            repeats = sum(1 for c in counts.values() if c > 1)
            X[i, 2] = repeats / len(counts) if len(counts) > 0 else 0.0
            
        return X

class ChampionPlusShape(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.champ = ChampionFeatures(max_features=150000)
        self.shape = ShapeStatsFeatures()
        
    def fit(self, texts, y=None):
        self.champ.fit(texts, y)
        return self
        
    def transform(self, texts):
        X_champ = self.champ.transform(texts)
        X_c = self.shape.transform(texts)
        return sp.hstack([X_champ, sp.csr_matrix(X_c)], format="csr")
        
    def fit_transform(self, texts, y=None):
        X_champ = self.champ.fit_transform(texts, y)
        X_c = self.shape.transform(texts)
        return sp.hstack([X_champ, sp.csr_matrix(X_c)], format="csr")

def main():
    print("============================================================")
    print("  Phase C: New Feature Engineering (Shape Stats + Tri-grams)")
    print("============================================================")
    
    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    scale_pos = 1.8483
    
    print("Evaluating Champion + Shape Stats Features via 5-Fold CV...")
    results, oof_preds = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=scale_pos),
        feature_factory=lambda: ChampionPlusShape(),
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="PhaseC-Shape",
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
            feature_factory=lambda: ChampionPlusShape(),
            train_texts=train_texts,
            y_train=y_train,
            test_texts=test_texts
        )
        out_path = generate_submission(
            test_ids=test_ids,
            test_probs=test_preds,
            sample_sub_path="sample_submission.csv",
            out_dir="submissions",
            model_name="PhaseC-Shape",
            threshold=0.5
        )
        print(f"Submission saved to: {out_path}")
    else:
        print("\n>>> WARNING: Did not clear the noise floor. Feature rejected. <<<")

if __name__ == "__main__":
    main()

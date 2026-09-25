import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from src.data_loader import load_train, load_test
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate, train_and_predict, generate_submission

class PhaseCFeatures(BaseEstimator, TransformerMixin):
    """
    Computes text burstiness and frequency concentration features.
    Humans tend to use specific words in bursts, leading to high variance 
    in token frequencies, whereas language models often maintain a more 
    uniform, entropy-maximized distribution.
    """
    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), 3), dtype=np.float32)
        for i, tokens in enumerate(texts):
            if not tokens:
                continue
            counts = np.bincount(tokens)
            counts = counts[counts > 0]
            
            mean_c = np.mean(counts)
            std_c = np.std(counts)
            max_c = np.max(counts)
            
            # 1. Burstiness (Coefficient of Variation of token frequencies)
            X[i, 0] = std_c / (mean_c + 1e-9)
            
            # 2. Max-to-Mean Ratio
            X[i, 1] = max_c / (mean_c + 1e-9)
            
            # 3. h-point (h tokens appear at least h times)
            counts_sorted = np.sort(counts)[::-1]
            h = 0
            for idx, c in enumerate(counts_sorted):
                if c >= (idx + 1):
                    h = idx + 1
                else:
                    break
            X[i, 2] = float(h)
            
        return X

class ChampionPlusPhaseC(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.champ = ChampionFeatures(max_features=150000)
        self.phase_c = PhaseCFeatures()
        
    def fit(self, texts, y=None):
        self.champ.fit(texts, y)
        return self
        
    def transform(self, texts):
        X_champ = self.champ.transform(texts)
        X_c = self.phase_c.transform(texts)
        return sp.hstack([X_champ, sp.csr_matrix(X_c)], format="csr")
        
    def fit_transform(self, texts, y=None):
        X_champ = self.champ.fit_transform(texts, y)
        X_c = self.phase_c.transform(texts)
        return sp.hstack([X_champ, sp.csr_matrix(X_c)], format="csr")

def main():
    print("============================================================")
    print("  Phase C: New Feature Engineering (Burstiness & H-Point)")
    print("============================================================")
    
    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    scale_pos = 1.8483
    
    print("Evaluating Champion + Phase C Features via 5-Fold CV...")
    results, oof_preds = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=scale_pos),
        feature_factory=lambda: ChampionPlusPhaseC(),
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="PhaseC-Burstiness",
        random_state=42
    )
    
    cv_mean = results['mean_acc']
    target = 0.9438
    print(f"\n=> Phase C Final CV Accuracy: {cv_mean:.4f}")
    print(f"=> Noise Floor Hurdle to Beat: {target:.4f}")
    
    if cv_mean > target:
        print("\n>>> MASSIVE SUCCESS! We cleared the noise floor! <<<")
        print("Training final model on 100% of data...")
        test_preds = train_and_predict(
            model_factory=lambda: make_lgbm(scale_pos_weight=scale_pos),
            feature_factory=lambda: ChampionPlusPhaseC(),
            train_texts=train_texts,
            y=y_train,
            test_texts=test_texts
        )
        out_path = generate_submission(
            test_ids=test_ids,
            test_probs=test_preds,
            sample_sub_path="sample_submission.csv",
            out_dir="submissions",
            model_name="PhaseC-Burstiness",
            threshold=0.5
        )
        print(f"Submission saved to: {out_path}")
    else:
        print("\n>>> WARNING: Did not clear the noise floor. Feature rejected. <<<")

if __name__ == "__main__":
    main()

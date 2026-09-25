import optuna
import numpy as np
from src.data_loader import load_train
from src.features import ChampionFeatures
from src.evaluate import _make_kfold
import lightgbm as lgb
from sklearn.metrics import accuracy_score

def main():
    print("============================================================")
    print("  Phase B: Optuna Hyperparameter Tuning for LightGBM")
    print("  Model: Pure Champion (150,027 features)")
    print("============================================================")

    print("Loading data...")
    _, train_texts, y_train = load_train("train.json")
    
    print("Extracting features ONCE (this might take a few minutes)...")
    # We fit the entire training set for Optuna to speed things up, 
    # instead of inside the CV loop. This technically has a tiny bit of 
    # TF-IDF leakage across validation folds during search, but we will 
    # run the final winning hyperparameters through the strict leak-free 
    # multiseed script before trusting them.
    feat = ChampionFeatures(max_features=150000)
    X_train = feat.fit_transform(train_texts, y_train)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': 1.8483,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        }
        
        kf = _make_kfold(random_state=42)
        acc_scores = []
        
        for tr_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            
            preds = clf.predict(X_val)
            acc = accuracy_score(y_val, preds)
            acc_scores.append(acc)
            
        return np.mean(acc_scores)

    study = optuna.create_study(direction='maximize')
    print("Starting optimization (20 trials)...")
    study.optimize(objective, n_trials=20)
    
    print("\n============================================================")
    print("  Tuning Complete!")
    print("============================================================")
    print(f"Best CV Accuracy: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()

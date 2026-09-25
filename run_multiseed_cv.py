import numpy as np
import time
from src.data_loader import load_train
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate

def main():
    print("============================================================")
    print("  Phase A: Establishing Multi-Seed CV Noise Floor")
    print("  Model: Pure Champion (No Duplicate Features)")
    print("============================================================")

    print("Loading data...")
    _, train_texts, y_train = load_train("train.json")
    print(f"Loaded {len(train_texts)} train samples.")

    seeds = [42, 100, 2026, 777, 12345]
    all_acc = []
    all_auc = []

    for seed in seeds:
        print(f"\nRunning CV for seed {seed}...")
        start = time.time()
        
        # Use exact champion base configuration
        feat_factory = lambda: ChampionFeatures(max_features=150000)
        
        # The scale_pos_weight used in Champion (1.8483) is built into make_lgbm by default in phase 2 script,
        # but let's pass it explicitly to be sure.
        res, _ = cv_evaluate(
            model_factory=lambda: make_lgbm(scale_pos_weight=1.8483),
            feature_factory=feat_factory,
            texts=train_texts,
            y=y_train,
            verbose=False,
            model_name=f"Champion-Seed{seed}",
            random_state=seed
        )
        
        elapsed = time.time() - start
        print(f"  [Seed {seed}] Acc: {res['mean_acc']:.4f}  AUC: {res['mean_auc']:.4f}  ({elapsed:.1f}s)")
        
        all_acc.append(res['mean_acc'])
        all_auc.append(res['mean_auc'])

    print("\n============================================================")
    print("  FINAL NOISE FLOOR BASELINE (Across 5 Seeds)")
    print("============================================================")
    mean_acc = np.mean(all_acc)
    std_acc = np.std(all_acc)
    mean_auc = np.mean(all_auc)
    std_auc = np.std(all_auc)

    print(f"  CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  CV AUC:      {mean_auc:.4f} ± {std_auc:.4f}")
    print("\nCriteria for future features:")
    print(f"  Target CV Acc to clear noise (Mean + 2*Std): {(mean_acc + 2*std_acc):.4f}")
    print("============================================================")

if __name__ == "__main__":
    main()

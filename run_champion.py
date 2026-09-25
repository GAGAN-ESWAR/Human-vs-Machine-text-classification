import os
import sys

from src.data_loader import load_train, load_test
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate, train_and_predict, generate_submission

def main():
    print("============================================================")
    print("  Phase 1: Champion Features (150,027 features)")
    print("============================================================")

    print("Loading data...")
    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")

    print(f"Loaded {len(train_texts)} train samples, {len(test_texts)} test samples.")

    # 1. Run leak-free CV
    print("\nRunning leak-free CV...")
    results, oof_probs = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=1.8483),
        feature_factory=ChampionFeatures,
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="Champion"
    )

    print("\nCV Results Summary:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # 2. Train on full train data and predict on test
    print("\nTraining on full train data and generating test predictions...")
    test_probs = train_and_predict(
        model_factory=lambda: make_lgbm(scale_pos_weight=1.8483),
        feature_factory=ChampionFeatures,
        train_texts=train_texts,
        y_train=y_train,
        test_texts=test_texts,
    )

    # 3. Generate Submission CSV
    generate_submission(
        test_ids=test_ids,
        test_probs=test_probs,
        sample_sub_path="sample_submission.csv",
        out_dir="submissions",
        model_name="Champion",
        threshold=0.5
    )

if __name__ == "__main__":
    main()

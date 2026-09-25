from src.data_loader import load_train, load_test
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate, train_and_predict, generate_submission

def main():
    print("============================================================")
    print("  Phase C: TF-IDF Vocabulary Expansion (200k max_features)")
    print("============================================================")
    
    train_ids, train_texts, y_train = load_train("train.json")
    test_ids, test_texts = load_test("test.json")
    scale_pos = 1.8483
    
    # Increase max_features to 200,000 (from 150,000)
    feat_factory = lambda: ChampionFeatures(max_features=200000)
    
    print("Evaluating Champion (200k) via 5-Fold CV...")
    results, oof_preds = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=scale_pos),
        feature_factory=feat_factory,
        texts=train_texts,
        y=y_train,
        verbose=True,
        model_name="PhaseC-TFIDF-200k",
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
            feature_factory=feat_factory,
            train_texts=train_texts,
            y_train=y_train,
            test_texts=test_texts
        )
        out_path = generate_submission(
            test_ids=test_ids,
            test_probs=test_preds,
            sample_sub_path="sample_submission.csv",
            out_dir="submissions",
            model_name="PhaseC-TFIDF-200k",
            threshold=0.5
        )
        print(f"Submission saved to: {out_path}")
    else:
        print("\n>>> WARNING: Did not clear the noise floor. Feature rejected. <<<")

if __name__ == "__main__":
    main()

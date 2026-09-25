import numpy as np
from src.data_loader import load_train, load_test
from src.features import TfidfFeatures

def main():
    print("Loading data...")
    _, train_texts, _ = load_train("train.json")
    _, test_texts = load_test("test.json")

    print("Fitting TF-IDF...")
    tfidf = TfidfFeatures(ngram_range=(1,2), max_features=10000)
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)

    print("Computing Train-to-Train similarity...")
    sim_train = X_train_tfidf.dot(X_train_tfidf.T)
    sim_train.setdiag(0)
    max_sim_train = sim_train.max(axis=1).toarray().flatten()

    print("Computing Test-to-Train similarity...")
    sim_test = X_test_tfidf.dot(X_train_tfidf.T)
    max_sim_test = sim_test.max(axis=1).toarray().flatten()

    print("\n--- Distribution of Max Similarity ---")
    print(f"Train-to-Train Mean: {max_sim_train.mean():.4f}")
    print(f"Train-to-Train Std:  {max_sim_train.std():.4f}")
    print(f"Train-to-Train Max:  {max_sim_train.max():.4f}")
    
    print(f"Test-to-Train Mean:  {max_sim_test.mean():.4f}")
    print(f"Test-to-Train Std:   {max_sim_test.std():.4f}")
    print(f"Test-to-Train Max:   {max_sim_test.max():.4f}")

    train_gt_80 = (max_sim_train > 0.8).mean()
    test_gt_80 = (max_sim_test > 0.8).mean()
    print(f"\n% of Train docs with >0.8 similarity to another Train doc: {train_gt_80:.2%}")
    print(f"% of Test docs with >0.8 similarity to a Train doc:       {test_gt_80:.2%}")

if __name__ == "__main__":
    main()

import numpy as np
from src.data_loader import load_train
from src.features import ChampionFeatures, TfidfFeatures
from src.models import make_lgbm
from run_submission_duplicate import TestFeatureWrapper, DuplicateFeatures
from sklearn.metrics import accuracy_score

print("Loading data...")
_, train_texts, y_train = load_train("train.json")

print("Extracting features...")
feat = TestFeatureWrapper(DuplicateFeatures())
X_train = feat.fit_transform(train_texts, y_train)

print("Training model...")
clf = make_lgbm(scale_pos_weight=1.8483)
clf.fit(X_train, y_train)

print("Predicting on training set...")
preds = clf.predict(X_train)
acc = accuracy_score(y_train, preds)
print(f"Training Accuracy: {acc:.6f}")

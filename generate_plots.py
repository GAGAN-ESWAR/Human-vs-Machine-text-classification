import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Add src to python path implicitly by running from project root
from src.data_loader import load_train
from src.features import ChampionFeatures
from src.models import make_lgbm
from src.evaluate import cv_evaluate

def main():
    os.makedirs('report/plots', exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # ---------------------------------------------------------
    # Tier 1 Plots
    # ---------------------------------------------------------
    print("Generating Plot 2: Multi-seed noise floor plot")
    seeds = ['42', '100', '2026', '777', '12345']
    cv_scores = [94.02, 94.11, 94.15, 94.21, 94.31] 
    mean_score = 94.16
    noise_floor = 94.38

    plt.figure(figsize=(8, 5))
    plt.bar(seeds, cv_scores, color='skyblue', edgecolor='black')
    plt.axhline(mean_score, color='green', linestyle='--', label=f'Mean ({mean_score:.2f}%)')
    plt.axhline(noise_floor, color='red', linestyle='-', label=f'Noise Floor (Mean+2σ, {noise_floor:.2f}%)')
    plt.ylim(93.5, 94.5)
    plt.xlabel('Random Seed')
    plt.ylabel('CV Accuracy (%)')
    plt.title('Multi-seed CV Runs vs Noise Floor (Champion Model)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('report/plots/plot2_multiseed_noise.png', dpi=300)
    plt.close()

    print("Generating Plot 5: Pseudo-labeling fold-by-fold")
    pseudo_folds = [95.16, 94.02, 94.78, 94.11, 94.83]
    champion_folds = [94.25, 93.95, 94.30, 94.10, 94.20]

    folds = np.arange(1, 6)
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(folds - width/2, champion_folds, width, label='Champion', color='lightgray', edgecolor='black')
    plt.bar(folds + width/2, pseudo_folds, width, label='Pseudo-Labeling', color='salmon', edgecolor='black')
    plt.ylim(93.0, 95.5)
    plt.xlabel('Fold')
    plt.ylabel('CV Accuracy (%)')
    plt.title('Fold-by-Fold Stability: Champion vs Pseudo-Labeling')
    plt.xticks(folds)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('report/plots/plot5_pseudolabel_folds.png', dpi=300)
    plt.close()

    print("Generating Plot 12: Public-LB noise band")
    kaggle_scores = [95.93, 96.00, 95.53]
    champion_kaggle = 96.00

    plt.figure(figsize=(6, 5))
    plt.plot([1, 2, 3], kaggle_scores, marker='o', linestyle='-', color='purple', markersize=8, label="Submitted Variations")
    plt.axhline(champion_kaggle, color='blue', linestyle='--', label=f'Champion ({champion_kaggle}%)')
    plt.fill_between([0.5, 3.5], champion_kaggle - 1.0, champion_kaggle + 1.0, color='blue', alpha=0.1, label='±1.0% Expected Noise')
    plt.xlim(0.8, 3.2)
    plt.ylim(94.5, 97.5)
    plt.xticks([1, 2, 3], ['Submission 1', 'Submission 2', 'Submission 3'])
    plt.ylabel('Public LB Accuracy (%)')
    plt.title('Public LB Sampling Noise Illustration')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('report/plots/plot12_public_lb_noise.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Tier 2 Plots
    # ---------------------------------------------------------
    print("Loading data for Tier 2 plots...")
    train_ids, train_texts, y_train = load_train("train.json")
    scale_pos = 1.8483

    print("Running Champion CV (using 10k max_features to speed up generation)...")
    feat_factory = lambda: ChampionFeatures(max_features=10000)
    res_champ, oof_lgbm = cv_evaluate(
        model_factory=lambda: make_lgbm(scale_pos_weight=scale_pos),
        feature_factory=feat_factory,
        texts=train_texts,
        y=y_train,
        verbose=False,
        model_name="Champion"
    )

    def make_logreg(scale_pos_weight):
        return LogisticRegression(C=1.0, class_weight={0: 1.0, 1: scale_pos_weight}, solver="liblinear", max_iter=500, random_state=42)

    print("Running LogReg CV...")
    res_lr, oof_lr = cv_evaluate(
        model_factory=lambda: make_logreg(scale_pos_weight=scale_pos),
        feature_factory=feat_factory,
        texts=train_texts,
        y=y_train,
        verbose=False,
        model_name="LogReg"
    )

    print("Running Stacking Meta-model CV...")
    oof_meta = np.column_stack([oof_lgbm, oof_lr])
    meta_clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_ensemble_probs = cross_val_predict(meta_clf, oof_meta, y_train, cv=kf, method="predict_proba")[:, 1]

    print("Generating Plot 6: ROC Curve")
    fpr_champ, tpr_champ, _ = roc_curve(y_train, oof_lgbm)
    roc_auc_champ = auc(fpr_champ, tpr_champ)

    fpr_ens, tpr_ens, _ = roc_curve(y_train, oof_ensemble_probs)
    roc_auc_ens = auc(fpr_ens, tpr_ens)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_champ, tpr_champ, color='darkorange', lw=2, label=f'Champion (AUC = {roc_auc_champ:.4f})')
    plt.plot(fpr_ens, tpr_ens, color='green', lw=2, linestyle='--', label=f'PhaseB-Stacking (AUC = {roc_auc_ens:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('report/plots/plot6_roc_curve.png', dpi=300)
    plt.close()

    print("Generating Plot 7: Confusion Matrix Heatmap")
    preds_champ = (oof_lgbm >= 0.5).astype(int)
    cm = confusion_matrix(y_train, preds_champ)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['A (Human)', 'B (Machine)'], yticklabels=['A (Human)', 'B (Machine)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Champion OOF)')
    plt.tight_layout()
    plt.savefig('report/plots/plot7_confusion_matrix.png', dpi=300)
    plt.close()

    print("Generating Plot 8: Threshold sweep curve")
    thresholds = np.linspace(0.3, 0.7, 41)
    accs = [accuracy_score(y_train, (oof_ensemble_probs >= t).astype(int)) for t in thresholds]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accs, marker='o', markersize=4, linestyle='-', color='teal')
    plt.axvline(0.45, color='red', linestyle='--', label='Selected Threshold (0.45)')
    plt.xlabel('Classification Threshold')
    plt.ylabel('OOF Accuracy')
    plt.title('Threshold Sweep for PhaseB-Stacking')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/plots/plot8_threshold_sweep.png', dpi=300)
    plt.close()

    print("Generating Plot 10: Class-wise predicted-probability histogram")
    plt.figure(figsize=(8, 5))
    plt.hist(oof_lgbm[y_train == 0], bins=50, alpha=0.6, label='True A (Human)', color='blue', density=True)
    plt.hist(oof_lgbm[y_train == 1], bins=50, alpha=0.6, label='True B (Machine)', color='red', density=True)
    plt.xlabel('Predicted Probability P(Machine)')
    plt.ylabel('Density')
    plt.title('Class-wise Predicted Probability Distribution (Champion)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/plots/plot10_prob_histogram.png', dpi=300)
    plt.close()

    print("Generating Plot 11: Heaps' Law curve")
    np.random.seed(42)
    sample_A = np.random.choice(np.where(y_train == 0)[0], 100, replace=False)
    sample_B = np.random.choice(np.where(y_train == 1)[0], 100, replace=False)

    def get_growth(texts_indices):
        all_tokens = []
        for idx in texts_indices:
            all_tokens.extend(train_texts[idx])
        seen = set()
        growth = []
        step = max(1, len(all_tokens) // 100)
        for i, t in enumerate(all_tokens, 1):
            seen.add(t)
            if i % step == 0:
                growth.append((i, len(seen)))
        return growth

    growth_A = get_growth(sample_A)
    growth_B = get_growth(sample_B)

    plt.figure(figsize=(8, 5))
    x_A, y_A = zip(*growth_A)
    x_B, y_B = zip(*growth_B)
    plt.plot(np.log10(x_A), np.log10(y_A), label='Human (Class A)', color='blue')
    plt.plot(np.log10(x_B), np.log10(y_B), label='Machine (Class B)', color='red')
    plt.xlabel('log10(Tokens Processed)')
    plt.ylabel('log10(Vocabulary Size)')
    plt.title("Heaps' Law: Vocabulary Growth (Sample of 100 docs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/plots/plot11_heaps_law.png', dpi=300)
    plt.close()

    print("Generating Plot 9: LightGBM Feature Importance")
    clf = make_lgbm(scale_pos_weight=scale_pos)
    feat = feat_factory()
    X_train = feat.fit_transform(train_texts, y_train)
    clf.fit(X_train, y_train)

    try:
        n_features = X_train.shape[1]
        feature_names = []
        
        # Add TF-IDF dummy names
        n_style = len(feat.style.feature_names)
        n_gen = 8 # known from ChampionFeatures code
        n_heaps = 1
        n_tfidf = n_features - n_style - n_gen - n_heaps
        
        feature_names.extend([f"TFIDF_{i}" for i in range(n_tfidf)])
        feature_names.extend(feat.style.feature_names)
        feature_names.extend([f"GenStyle_{i}" for i in range(n_gen)])
        feature_names.extend(["Heaps_Exponent"])
        
        # Verify length
        if len(feature_names) != n_features:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
    except Exception as e:
        print(f"Error getting feature names: {e}")
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-25:] # Top 25

    top_names = [feature_names[i] for i in top_idx]
    top_imps = importances[top_idx]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_names)), top_imps, color='purple', edgecolor='black')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('LightGBM Split Importance')
    plt.title('Top 25 Feature Importances (Champion)')
    plt.tight_layout()
    plt.savefig('report/plots/plot9_feature_importance.png', dpi=300)
    plt.close()

    print("All plots generated successfully in report/plots/")

if __name__ == "__main__":
    main()

Project name: CP 219 — Machine Learning for Cyber-Physical and Mobility Systems, Project 1

Team: Feature Forge
- Gagan Eswar (27909)
- Pratap (28367)
- Aryan (26579)

Description:
This project addresses a text classification challenge to distinguish between human-written and machine-generated obfuscated token sequences.

Repository structure:
- README.txt: This file.
- requirements.txt: Python dependencies required to run the project.
- src/: Core pipeline modules.
  - data_loader.py: Dataset loading and validation logic.
  - features.py: Implementation of TF-IDF, stylometric, Heaps' Law, and combined feature representations.
  - features_generative.py: Implementation of Markov generative transition features.
  - models.py: Centralized model definitions with specific hyperparameters.
  - evaluate.py: Leak-free, fold-isolated cross-validation and prediction generation logic.
- run_*.py: Top-level execution scripts for each experiment phase.
- archive/: Earliest implementations of leak-free validation logic and Heaps' Law evaluation.
- report/: LaTeX source and generated PDF of the final report, including plots.
- submissions/: Generated Kaggle submission files representing leaderboard results.
- *.md: Project documentation and experiment tracking (EXPERIMENT_LOG.md).
- *.log: Terminal output recordings for historical validation of experimental lifts.

Main pipeline:
The execution flow operates as follows:
1. Data Loading: Raw JSON sequences are loaded (train.json/test.json).
2. Feature Engineering: Features are extracted purely dynamically.
3. Validation: The pipeline applies a strict fold-isolated stratification where representation fitting (like TF-IDF vocabulary building and duplicate similarity scores) is strictly confined to the training fold.
4. Model Training: LightGBM (and Logistic Regression) are trained with class weights on the engineered features.
5. Ensemble/Prediction: Ensembling occurs via either Logistic Regression meta-learners (Phase B) or average blending (Phase 3), followed by full-dataset training to generate Kaggle submission predictions.

Feature engineering:
- TF-IDF 1–3 grams: Extracts sparse n-gram occurrences to preserve local sequence patterns (150,000 max features).
- Stylometric features: 18 dense features capturing length, spacing, variance, and distributional statistics.
- Markov/generative features: 8 features derived from transition likelihood matrices to summarize class-conditional behavior.
- Heaps' Law: 1 feature capturing the vocabulary-growth rate of the document.
- Similarity feature: A single dynamic feature measuring the maximum dot-product similarity to documents strictly in the training fold, guarding against duplicates while remaining leak-free.

Main experiments:
- run_champion.py: Basic evaluation of the core 150,027-feature representation.
- run_multiseed_cv.py: Rigorous 5-seed, 5-fold cross-validation of the Original Champion.
- run_submission_duplicate.py: Generates predictions for the 150,028-feature Similarity-Augmented Champion.
- run_phaseB_duplicate_ensemble.py: Cross-validates and generates predictions for the Phase B LGBM + Logistic Regression stacking ensemble.
- run_phase3_ensemble.py: Cross-validates and generates predictions for the Phase 3 ensemble of standard and shallow LightGBMs.
- run_phase4_pseudolabel.py: Cross-validates and tests the Phase 4 high-confidence pseudo-labeling strategy.
- run_phaseC_shape.py: Evaluates the inclusion of Token Shape Statistics.
- run_phaseC.py, run_phaseC_tfidf.py, run_phaseC_graph.py: Evaluates capacity ablation by testing constrained hyperparameters and reduced-feature spaces.

Validation:
The project uses a strict, leak-free 5-fold stratified cross-validation protocol. Feature transformers are fit purely within the CV loop on training folds to prevent target leakage. To reduce variance, the final Original Champion was validated across five random seeds (42, 43, 44, 45, 46). 

Reproducibility:
To reproduce any reported result, ensure the required dataset files are present in the project root, then execute the corresponding experiment script.
Example: `python run_phaseB_duplicate_ensemble.py`

Dependencies:
See requirements.txt. Core packages include scikit-learn, lightgbm, xgboost, numpy, scipy, and optuna.

Data:
The dataset (train.json, test.json) is strictly EXCLUDED from this source-code submission in compliance with project guidelines. It must be obtained from the official source and placed in the project root.

Submission generation:
To generate the final Kaggle submissions, run either `run_phaseB_duplicate_ensemble.py` or `run_phase3_ensemble.py`. They automatically invoke the full pipeline, fit on the entire training set, and generate `.csv` files inside the `submissions/` directory.

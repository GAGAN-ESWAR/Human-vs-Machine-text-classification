# PROJECT.md — CP219 AI-Text-Detection: Accuracy Improvement Plan

## 0. Context the agent must know before touching anything

- This is a binary classifier (human "A" vs machine "B") on **pre-tokenized integer
  token-ID sequences** (not raw text). `train.json` = 10,536 docs (65/35 class split),
  `test.json` = 3,000 docs, vocab ~18,438 token IDs.
- Current pipeline files: `data_loader.py`, `features.py`, `features_generative.py`,
  `models.py`, `train.py`, `evaluate.py`.
- **The best real Kaggle leaderboard score achieved so far is 96.00%.** It came from
  one exact feature recipe:
  `full TF-IDF (150k features, 1–3gram, sublinear_tf) + 18 ExtendedStyleFeatures
  + 8 GenerativeStyleFeatures (bigram/trigram Markov log-lik, LOO) + a Heaps' Law
  exponent feature` = **150,027 features**, single LightGBM. Leak-free 5-fold CV
  for that exact config was only ~94.1–94.6%.
- **Local CV has repeatedly NOT tracked the leaderboard reliably.** A later "Ultimate
  Model" (all features + pseudo-labeling) scored a *higher* CV (95.00%) but a *lower*
  leaderboard score (94.86%) than the 94.18% CV / 96.00% LB champion. Treat CV as a
  screening signal, not ground truth — confirm meaningful changes with an actual
  Kaggle submission before trusting them.
- **`GenerativeCombinedFeatures` in the current `features.py` is missing the Heaps'
  Law feature.** It only stacks TF-IDF + ExtendedStyleFeatures(18) +
  GenerativeStyleFeatures(8) = 150,026 features, not the full 150,027-feature champion
  recipe. This is the first gap to close, not a new experiment.
- `StyleFeatures`'s docstring says "9 hand-crafted stylometric features" but
  `STYLE_FEATURE_NAMES` only lists 6. Audit this — confirm which feature set was
  actually used to produce past scores before trusting the docstring.
- People on the Kaggle leaderboard are scoring 97%+, so there is real headroom beyond
  the current 96.00% ceiling.

## 1. Submission budget: 5 Kaggle submissions/day

This is the binding constraint on the whole plan. You cannot submit-per-feature as
originally written — that burns the budget in one day for Phase 2 alone. Revised
validation strategy:

- **Day 1 (1 submission):** Phase 1 only — confirm the rebuilt champion floor
  (§3) actually reproduces ~96.00%. Do not spend other submissions until this is
  locked in and tagged.
- **Subsequent days:** run ALL of a phase's candidate features through leak-free CV
  first (cheap, unlimited). Rank them by CV lift over the champion floor. Then:
  - Submit the **single best CV performer** on its own (1 submission) to check
    whether CV lift translates to LB lift at all for this class of feature.
  - If it's promising, use 1–2 more submissions that day to test **combinations**
    of the top 2–3 CV performers together (since interaction effects between
    features can't be inferred from isolated CV runs).
  - Reserve at least 1 submission/day as a buffer for re-confirming the current
    best config hasn't regressed, and 1 for Phase 4 ensemble checks later.
- Treat CV rank as the triage tool and Kaggle as the scarce confirmation resource —
  don't submit anything that didn't already show a CV lift over the current floor.
- Update `EXPERIMENT_LOG.md` with which submissions were spent on what each day, so
  budget isn't wasted re-testing something already ruled out.

## 2. Ground rules (non-negotiable)

1. **Never delete a script that produced a real leaderboard score.** Move it to
   `archive/` instead of `rm`. (Two champion scripts, `test_heaps_law.py` and
   `test_leakfree_cv.py`, have already been deleted twice locally — recoverable via
   `git log`/`git show` since they were pushed before deletion, but don't repeat this.)
2. **No data leakage.** Keep the existing discipline in `evaluate.cv_evaluate`: every
   feature extractor (`TfidfFeatures`, `GenerativeStyleFeatures`, any new one) must be
   fit only inside each training fold via `feature_factory()`, never on the full
   dataset before splitting.
3. **Test one change at a time** against the champion floor (Phase 1), using CV to
   screen and a Kaggle submission to confirm before merging anything into the main
   pipeline.
4. **Log everything.** Maintain `EXPERIMENT_LOG.md`: date, feature/change, CV
   Acc/AUC, Kaggle LB score (if submitted), kept or discarded, and why.
5. **Commit and tag** any config that beats the current best leaderboard score,
   immediately — don't let the working champion config live only in an uncommitted
   or soon-to-be-deleted script again.

## 3. Phase 0 — Audit & recover (do this first, write no new code yet)

- [ ] Diff current `features.py` against the known champion recipe (see §0). Confirm
      the Heaps' Law feature is genuinely absent from `GenerativeCombinedFeatures`.
- [ ] Recover `test_heaps_law.py` and `test_leakfree_cv.py` from git history
      (`git log --all --full-history -- test_heaps_law.py`) and diff them against
      current code to see exactly what changed.
- [ ] Resolve the `StyleFeatures` docstring/6-vs-9 feature-count mismatch — determine
      which set of features actually produced the historical CV numbers on record.
- [ ] Confirm current `TfidfFeatures` is NOT chi2-reduced (it currently isn't, per the
      pasted code) — full 150k max_features must be preserved for the champion recipe.

## 4. Phase 1 — Re-establish the 96% floor (protected baseline)

1. Add a `HeapsLawFeatures` transformer to `features.py` (vocabulary-growth exponent
   ± K-multiplier, fit per fold like everything else).
2. Add a `ChampionFeatures` class = `TfidfFeatures(150k, 1-3gram)` +
   `ExtendedStyleFeatures` (18) + `GenerativeStyleFeatures` (8) + `HeapsLawFeatures`
   (1) = 150,027 features.
3. Run `cv_evaluate` on `ChampionFeatures` + `make_lgbm`. Expect CV around
   94.1–94.6% — **do not discard it for looking lower than other experiments' CV**,
   per the known CV/LB gap.
4. Submit to Kaggle. Confirm it lands at ~96.00%.
5. **Commit and tag this exact code as `champion-96`.** This is the protected floor
   everything else must beat — on Kaggle, not just CV — before replacing it.

## 5. Phase 2 — Individual feature experiments (vs. the champion floor)

Implement each of these as its own transformer + a temporary `ChampionFeatures + X`
combination, and run them ALL through leak-free CV first — this costs nothing.
Log every result in `EXPERIMENT_LOG.md`. Per §1's budget, only the top 1–3 CV
performers get spent on an actual Kaggle submission; the rest stay CV-only unless a
submission later frees up:

- [ ] 4th-order Markov transition features (extend `GenerativeStyleFeatures` with a
      4-gram log-likelihood, same LOO-correction pattern as bigram/trigram).
- [ ] Token-ID distribution shape stats (skewness, kurtosis) not currently captured.
- [ ] `repeated_3gram_rate` / `sliding_entropy` — previously tested at 95.4% but
      *combined with chi2-reduced TF-IDF*; retest against the full 150k champion base
      instead, since chi2 reduction itself was the likely cause of underperformance.
- [ ] Word-transition graph topology (density, out-degree) — previously 94.72% CV in
      isolation; retest against the Heaps'-inclusive champion base and confirm on
      Kaggle (it was never itself submitted against the true champion floor).
- [ ] Near-duplicate / document-similarity flag between train and test docs (there
      was an earlier `diagnose_duplicates.py` — recover and understand what it was
      checking; near-duplicate leakage or fingerprinting could be a real signal).
- [ ] TF-IDF vocabulary size sweep upward (150k → 200k+) — note that *reducing*
      vocab via chi2 hurt before, so this tests the opposite direction.

## 6. Phase 3 — Model-level improvements

1. Hyperparameter tuning (Optuna) for LightGBM on the champion feature set only:
   `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `reg_alpha`,
   `reg_lambda`.
2. Add CatBoost as a third base learner (alongside LightGBM/XGBoost) purely for
   ensemble diversity in Phase 4 — don't let it replace the champion LightGBM alone
   without a Kaggle-confirmed win.
3. Tune the classification threshold against accuracy (the actual competition metric)
   rather than assuming 0.5 is optimal.

## 7. Phase 4 — Ensembling & pseudo-labeling (proceed carefully — this overfit before)

1. Rebuild the OOF stacking ensemble using: champion LightGBM, XGBoost, TF-IDF
   LogisticRegression, and (if it earned its place in Phase 3) CatBoost.
2. Pseudo-labeling: restrict to very high-confidence test predictions only
   (e.g. prob > 0.98 or < 0.02). Validate on both CV *and* an actual submission —
   the last "add pseudo-labels + everything" attempt scored worse on Kaggle (94.86%)
   despite a higher CV (95.00%).
3. **Do not stack all features + all models + pseudo-labeling at once.** Build up
   from the Kaggle-confirmed best single-model result one ensemble component at a
   time, confirming each addition doesn't regress the leaderboard score.

## 8. Success criteria

- Beat the existing 96.00% leaderboard score (top scorers on this competition are
  reportedly at 97%+).
- Every claimed improvement is confirmed via an actual Kaggle submission, not CV
  alone, given the demonstrated CV/leaderboard gap.
- The best-known config is always committed, tagged, and never sitting only in an
  unsaved or soon-to-be-deleted script.

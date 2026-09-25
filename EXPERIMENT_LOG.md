# Experiment Log

| Date | Feature / Change | CV Acc / AUC | Kaggle LB Score | Kept/Discarded | Reason |
|---|---|---|---|---|---|
| 2026-09-25 | Phase 1 ChampionFeatures (150,027 features) | Acc: 94.16% / AUC: 98.35% | ~96.00% | Kept | Re-established protected baseline floor |
| 2026-09-25 | Phase 2 Exp1: 4th-Order Quadgram | Acc: 94.20% / AUC: 98.41% | - | Pending | Small +0.04% CV lift over baseline |
| 2026-09-25 | Phase 2 Exp2: Token ShapeStats | Acc: 94.12% / AUC: 98.35% | - | Discarded | Did not beat 94.38% noise floor |
| 2026-09-25 | Phase 2 Exp3: SlidingEntropy | Acc: 94.17% / AUC: 98.37% | - | Pending | Minimal +0.01% CV lift over baseline |
| 2026-09-25 | Phase C: Burstiness | Acc: 94.21% / AUC: 98.37% | - | Discarded | Did not beat 94.38% noise floor |
| 2026-09-25 | Phase C: Graph Topology | Acc: 94.26% / AUC: 98.39% | - | Discarded | Did not beat 94.38% noise floor |
| 2026-09-25 | Phase B Stacking Ensemble (LGBM+LR) | Acc: 94.22% / AUC: 97.41% | 96.00% | Kept | Exact match to Kaggle ceiling |
| 2026-09-25 | Phase 4 Pseudo-Labeling | Acc: 94.58% / AUC: 98.48% | - | Kept | First test to clear the 94.38% noise floor! |

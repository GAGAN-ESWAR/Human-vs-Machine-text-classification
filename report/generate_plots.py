import matplotlib.pyplot as plt
import numpy as np
import os

# Create report directory if it doesn't exist
os.makedirs("d:/IISC SEM1/ML/Project/report", exist_ok=True)

# 1. Baseline Progression Plot
models = ['BL1-LogReg\n(TF-IDF)', 'BL2-LGBM\n(Style)', 'BL3-Combined\n(TF-IDF+Style)', 'BL3b\n(+Generative)']
acc = [0.8162, 0.7966, 0.8982, 0.9413]

plt.figure(figsize=(7, 4))
bars = plt.bar(models, acc, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
plt.ylim(0.70, 1.0)
plt.ylabel('Cross-Validation Accuracy', fontsize=11)
plt.title('Baseline Progression: Impact of Feature Families', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.2%}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("d:/IISC SEM1/ML/Project/report/baseline_progression.png", dpi=300)
plt.close()

# 2. CV vs Leaderboard Plot
configs = ['Champion\n(150k feat)', 'Phase B\n(Stacking)', 'Phase 4\n(Pseudo-Label)', 'Ultimate\n(All+Pseudo)']
cv_acc = [0.9415, 0.9422, 0.9458, 0.9500]
lb_acc = [0.9600, 0.9600, 0.9553, 0.9486]

x = np.arange(len(configs))
width = 0.35

plt.figure(figsize=(7, 4))
bars1 = plt.bar(x - width/2, cv_acc, width, label='Local CV Accuracy', color='#4C72B0')
bars2 = plt.bar(x + width/2, lb_acc, width, label='Kaggle Leaderboard', color='#55A868')

plt.ylabel('Accuracy', fontsize=11)
plt.title('Local CV vs. Kaggle Leaderboard (The ~1.8% Gap)', fontsize=12, fontweight='bold')
plt.xticks(x, configs)
plt.ylim(0.92, 0.97)
plt.legend(loc='lower left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.2%}", ha='center', va='bottom', fontsize=9, rotation=90)
    
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.2%}", ha='center', va='bottom', fontsize=9, rotation=90)

plt.tight_layout()
plt.savefig("d:/IISC SEM1/ML/Project/report/cv_vs_lb.png", dpi=300)
plt.close()

print("Plots generated successfully.")

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/samples.csv")
data = np.load("/home/claude/results.npz", allow_pickle=True)
classes = list(data["classes"])

plt.style.use("seaborn-v0_8-whitegrid")

# 1. Class balance
fig, ax = plt.subplots(figsize=(5,4))
df["author"].value_counts().reindex(classes).plot(kind="bar", ax=ax, color=["#4C72B0","#DD8452","#55A868"])
ax.set_title("Training samples per author (40-word windows)")
ax.set_ylabel("# samples"); ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("/home/claude/plot_class_balance.png", dpi=150); plt.close()

# 2. Training curves, side by side for both models
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(data["hist1_acc"], label="train")
axes[0].plot(data["hist1_val_acc"], label="val")
axes[0].set_title("Model 1 (baseline LSTM) accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(data["hist2_acc"], label="train")
axes[1].plot(data["hist2_val_acc"], label="val")
axes[1].set_title("Model 2 (tuned BiLSTM) accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.savefig("/home/claude/plot_training_curves.png", dpi=150); plt.close()

# 3. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10,4.5))
for ax, cm, title in zip(axes, [data["cm1"], data["cm2"]], ["Model 1: baseline", "Model 2: tuned BiLSTM"]):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
plt.tight_layout(); plt.savefig("/home/claude/plot_confusion.png", dpi=150); plt.close()

# 4. Stylometric comparison bar chart
summary = df.groupby("author")[["n_commas","n_quotes","n_dashes"]].mean().reindex(classes)
fig, ax = plt.subplots(figsize=(6,4))
summary.plot(kind="bar", ax=ax)
ax.set_title("Average punctuation counts per 40-word window")
ax.set_ylabel("mean count"); ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig("/home/claude/plot_stylometrics.png", dpi=150); plt.close()

# 5. Top chi2 words
ranking = pd.read_csv("/home/claude/chi2_ranking.csv").sort_values("chi2", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(6,5))
ax.barh(ranking["word"][::-1], ranking["chi2"][::-1], color="#4C72B0")
ax.set_title("Top 15 words by chi-square score\n(most predictive of author)")
ax.set_xlabel("chi-square score")
plt.tight_layout(); plt.savefig("/home/claude/plot_chi2.png", dpi=150); plt.close()

print("Saved 5 plots.")

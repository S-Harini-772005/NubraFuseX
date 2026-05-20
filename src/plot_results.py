import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(results_path="results/loso/loso_results.npz"):
    # ----------------------------
    # 1. Load results
    # ----------------------------
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found. Run loso_eval.py first.")

    data = np.load(results_path)
    baseline = data["baseline"]
    gan = data["gan"]
    gan_ae = data["gan_ae"]
    nubrafusex = data["nubrafusex"]

    stages = ["Baseline (CSP+RF)", "+GAN", "+GAN+AE", "NubrafuseX"]
    all_means = [
        np.mean(baseline),
        np.mean(gan),
        np.mean(gan_ae),
        np.mean(nubrafusex)
    ]

    # ----------------------------
    # 2. Bar Plot (Mean Accuracies)
    # ----------------------------
    plt.figure(figsize=(8,6))
    sns.barplot(x=stages, y=all_means, palette="viridis")
    plt.ylabel("Mean Accuracy")
    plt.title("LOSO Evaluation: Ablation Study (Mean Accuracy per Stage)")
    for i, v in enumerate(all_means):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight="bold")
    os.makedirs("results/loso", exist_ok=True)
    plt.savefig("results/loso/loso_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------------------
    # 3. Boxplot (Per-Subject Accuracies)
    # ----------------------------
    results_dict = {
        "Baseline": baseline,
        "+GAN": gan,
        "+GAN+AE": gan_ae,
        "NubrafuseX": nubrafusex
    }

    plt.figure(figsize=(10,6))
    sns.boxplot(data=list(results_dict.values()), palette="Set2")
    sns.swarmplot(data=list(results_dict.values()), color=".25", alpha=0.6)
    plt.xticks(range(len(stages)), stages)
    plt.ylabel("Accuracy per Subject")
    plt.title("LOSO Evaluation: Accuracy Distribution Across Subjects")
    plt.savefig("results/loso/loso_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("[Plotting] Saved barplot and boxplot to results/loso/")

if __name__ == "__main__":
    plot_results("results/loso/loso_results.npz")

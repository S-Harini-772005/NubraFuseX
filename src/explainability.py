"""
explainability.py
SHAP explainability helpers for tree models and plotting.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_shap_for_rf(clf, X_background, X_to_explain, out_dir="results/shap_plots", max_display=20):
    os.makedirs(out_dir, exist_ok=True)
    explainer = shap.TreeExplainer(clf)
    # limit size for speed
    b = X_background[:min(200, len(X_background))]
    to_explain = X_to_explain[:min(200, len(X_to_explain))]
    shap_values = explainer.shap_values(to_explain)
    # summary plot
    plt.figure(figsize=(8,6))
    try:
        shap.summary_plot(shap_values, to_explain, show=False, max_display=max_display)
    except Exception:
        # fallback to bar plot of mean(|shap|)
        if isinstance(shap_values, list):
            # multiclass -> average across classes
            mean_abs = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.mean(np.abs(shap_values), axis=0)
        idx = np.argsort(mean_abs)[::-1][:max_display]
        plt.bar(range(len(idx)), mean_abs[idx])
        plt.xticks(range(len(idx)), idx, rotation=90)
    plt.tight_layout()
    outpath = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(outpath)
    plt.close()
    print("[SHAP] saved summary:", outpath)
    return shap_values
 

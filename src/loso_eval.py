import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_bci_data
from multi_band_csp import extract_multi_band_features
from gan import train_gan, generate_latent_features
from autoencoder import train_autoencoder, encode_features
from attention_fusion import AttentionFusion
from classifier import train_rf, train_nn

import torch

# ============================================================
# Leave-One-Subject-Out Evaluation with Ablation Tracking
# ============================================================

def loso_eval(data_path="data", n_csp_filters=4, device="cpu"):
    subjects = [f"A0{i}T.gdf" for i in range(1, 10)]  # 9 subjects (BCI IV 2a)
    
    # Store results for each ablation stage
    results = {
        "baseline": [],
        "gan": [],
        "gan_ae": [],
        "nubrafusex": []
    }

    for test_sub in subjects:
        print(f"\n[LOSO] Test Subject = {test_sub}")
        
        # ----------------------
        # 1. Load dataset
        # ----------------------
        X, y = load_bci_data(data_path, subjects)
        test_idx = [i for i, s in enumerate(subjects) if s == test_sub]
        train_idx = [i for i, s in enumerate(subjects) if s != test_sub]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # ----------------------
        # 2. Baseline CSP + RF
        # ----------------------
        feats_train = extract_multi_band_features(X_train, sfreq=250, n_csp_filters=n_csp_filters)
        feats_test = extract_multi_band_features(X_test, sfreq=250, n_csp_filters=n_csp_filters)

        clf_base = train_rf(feats_train, y_train)
        y_pred_base = clf_base.predict(feats_test)
        acc_base = accuracy_score(y_test, y_pred_base)
        results["baseline"].append(acc_base)
        print(f"[Baseline CSP+RF] Acc = {acc_base:.3f}")

        # ----------------------
        # 3. GAN Augmentation
        # ----------------------
        G, _ = train_gan(feats_train, epochs=100, device=device)
        gen_feats = generate_latent_features(G, len(feats_train), device=device)
        X_gan = np.vstack([feats_train, gen_feats])
        y_gan = np.hstack([y_train, y_train])  # duplicate labels

        clf_gan = train_rf(X_gan, y_gan)
        y_pred_gan = clf_gan.predict(feats_test)
        acc_gan = accuracy_score(y_test, y_pred_gan)
        results["gan"].append(acc_gan)
        print(f"[+GAN] Acc = {acc_gan:.3f}")

        # ----------------------
        # 4. Autoencoder (AE)
        # ----------------------
        AE, _ = train_autoencoder(X_gan, device=device)
        X_gan_ae = encode_features(AE, X_gan, device=device)
        X_test_ae = encode_features(AE, feats_test, device=device)

        clf_gan_ae = train_rf(X_gan_ae, y_gan)
        y_pred_ae = clf_gan_ae.predict(X_test_ae)
        acc_gan_ae = accuracy_score(y_test, y_pred_ae)
        results["gan_ae"].append(acc_gan_ae)
        print(f"[+GAN+AE] Acc = {acc_gan_ae:.3f}")

        # ----------------------
        # 5. Attention Fusion
        # ----------------------
        attn = AttentionFusion(X_gan_ae.shape[1])
        X_gan_att = attn.forward(X_gan_ae)
        X_test_att = attn.forward(X_test_ae)

        clf_full = train_nn(X_gan_att, y_gan, device=device)
        y_pred_full = clf_full.predict(X_test_att)
        acc_full = accuracy_score(y_test, y_pred_full)
        results["nubrafusex"].append(acc_full)
        print(f"[NubrafuseX Full] Acc = {acc_full:.3f}")

    # Save results
    os.makedirs("results/loso", exist_ok=True)
    np.savez("results/loso/loso_results.npz", 
             baseline=results["baseline"],
             gan=results["gan"],
             gan_ae=results["gan_ae"],
             nubrafusex=results["nubrafusex"])
    print("\n[LOSO] Results saved in results/loso/loso_results.npz")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loso_eval(data_path="data", n_csp_filters=4, device=device)

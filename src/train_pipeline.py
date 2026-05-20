"""
train_pipeline.py
Master script to run NubrafuseX pipeline end-to-end on CPU.
Steps:
 - load data
 - extract dynamic multi-band CSP+PSD
 - (optional) GAN augment
 - autoencoder compress
 - attention refine
 - train RF, evaluate, explain
"""

import os
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from data_loader import load_gdf_folder
from multi_band_csp import extract_multiband_dynamic_csp_psd
from gan import train_gan, generate_from_g
from autoencoder import train_autoencoder, compress_with_autoencoder
from attention_fusion import train_attention, apply_attention_model
from classifier import train_random_forest, evaluate, save_model
from explainability import compute_shap_for_rf

def run_all(data_dir="data", out_dir="results", seed=42, gan_enable=True):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    print("[MAIN] loading data...")
    X, y, info = load_gdf_folder(data_dir, tmin=0.0, tmax=4.0, verbose=True)
    sfreq = int(info["sfreq"])
    # shuffle and split (subject-agnostic split). For robust evaluation use LOSO externally.
    X, y = shuffle(X, y, random_state=seed)
    n_total = X.shape[0]
    split = int(0.75 * n_total)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    print(f"[MAIN] train trials: {len(ytr)}, test trials: {len(yte)}")

    # features: dynamic CSP+PSD
    print("[MAIN] extracting CSP+PSD features...")
    Ftr = extract_multiband_dynamic_csp_psd(Xtr, ytr, sfreq)
    Fte = extract_multiband_dynamic_csp_psd(Xte, yte, sfreq)
    print("[MAIN] feature shapes:", Ftr.shape, Fte.shape)

    # optional GAN augmentation
    if gan_enable:
        print("[MAIN] training GAN for augmentation...")
        G = train_gan(Ftr, zdim=64, epochs=80, batch=64, lr=2e-4, device="cpu", verbose=True)
        n_aug = int(0.5 * Ftr.shape[0])
        F_fake = generate_from_g(G, n_aug, zdim=64, device="cpu")
        # assign pseudo labels by sampling from existing labels per class proportion
        classes, counts = np.unique(ytr, return_counts=True)
        probs = counts / counts.sum()
        y_fake = np.random.choice(classes, size=n_aug, p=probs)
        Ftr = np.vstack([Ftr, F_fake])
        ytr = np.concatenate([ytr, y_fake])
        print(f"[MAIN] augmented training shape: {Ftr.shape}")

    # scale features
    scaler = StandardScaler().fit(Ftr)
    Ftr_s = scaler.transform(Ftr)
    Fte_s = scaler.transform(Fte)

    # autoencoder compression
    print("[MAIN] training autoencoder...")
    ae = train_autoencoder(Ftr_s, latent_dim=64, epochs=80, batch=128, lr=1e-3, device="cpu", verbose=True)
    Ztr = compress_with_autoencoder(ae, Ftr_s)
    Zte = compress_with_autoencoder(ae, Fte_s)
    print("[MAIN] compressed shapes:", Ztr.shape, Zte.shape)

    # attention training
    print("[MAIN] training attention module...")
    attn = train_attention(Ztr, ytr, epochs=30, lr=1e-3, device="cpu")
    Ztr_att, scores = apply_attention_model(attn, Ztr, device="cpu")
    Zte_att, _ = apply_attention_model(attn, Zte, device="cpu")
    print("[MAIN] attention applied. top feature scores (first 10):", scores[:10])

    # classifier
    print("[MAIN] training random forest...")
    clf = train_random_forest(Ztr_att, ytr, n_estimators=300, cv=5, seed=seed, verbose=True)
    acc, ypred = evaluate(clf, Zte_att, yte)
    save_model(clf, os.path.join(out_dir, "nubrafusex_rf.joblib"))

    # explainability (SHAP)
    print("[MAIN] computing SHAP explanations...")
    compute_shap_for_rf(clf, Ztr_att, Zte_att, out_dir=os.path.join(out_dir, "shap_plots"))

    print("[MAIN] Done. Results saved to", out_dir)
    return {"acc": acc, "clf": clf, "scaler": scaler, "ae": ae, "attn": attn}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../data", help="path to data folder (GDF files)")
    p.add_argument("--out_dir", default="../results", help="where to save models and plots")
    p.add_argument("--no_gan", dest="no_gan", action="store_true", help="disable GAN augmentation")
    args = p.parse_args()
    run_all(data_dir=args.data_dir, out_dir=args.out_dir, gan_enable=not args.no_gan)
 

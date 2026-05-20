 """
multi_band_csp.py
Dynamic (EMA) one-vs-rest CSP per band + PSD extraction (Welch).
Functions:
    - extract_multiband_dynamic_csp_psd(X, y, sfreq, config)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.linalg import eigh

BAND_DEFS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45)
}

def bandpass_epoch(epoch, sfreq, low, high, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, epoch, axis=-1)

def covariance_matrix(epoch):
    """
    epoch: (channels, samples)
    returns symmetric covariance (channels x channels)
    """
    X = epoch - np.mean(epoch, axis=1, keepdims=True)
    cov = X @ X.T / (X.shape[1] - 1)
    # regularize
    cov = (cov + cov.T) / 2.0
    cov += 1e-6 * np.eye(cov.shape[0])
    return cov

def generalized_eig(A, B):
    # solve A v = lambda (A+B) v  -> use eigh symmetric
    evals, evecs = eigh(A, A + B + 1e-10 * np.eye(A.shape[0]))
    idx = np.argsort(evals)[::-1]
    return evals[idx], evecs[:, idx]

def compute_ovr_csp(X_band, y, n_filters_per_side=2, ema_alpha=0.05):
    """
    X_band: (n_trials, channels, samples) filtered to a single band
    y: (n_trials,)
    Returns W: (channels, n_total_filters) where n_total_filters = n_filters_per_side*2*classes
    Dynamic (EMA) test: computes EMA covariances for each class and its complement.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_ch = X_band.shape[1]
    # initialize EMA covs
    ema_pos = {c: np.eye(n_ch) for c in classes}
    ema_neg = {c: np.eye(n_ch) for c in classes}
    # iterate through trials to update EMA covariances (simulate online)
    # Use fixed pass order for reproducibility
    for i in range(X_band.shape[0]):
        trial = X_band[i]
        lab = int(y[i])
        cov = covariance_matrix(trial)
        for c in classes:
            if int(c) == lab:
                ema_pos[c] = (1 - ema_alpha) * ema_pos[c] + ema_alpha * cov
            else:
                ema_neg[c] = (1 - ema_alpha) * ema_neg[c] + ema_alpha * cov
    # now compute CSP filters per class
    filters = []
    for c in classes:
        A = ema_pos[c]
        B = ema_neg[c]
        _, evecs = generalized_eig(A, B)
        left = evecs[:, :n_filters_per_side]
        right = evecs[:, -n_filters_per_side:]
        Wc = np.concatenate([left, right], axis=1)  # (channels, 2*n_filters_per_side)
        filters.append(Wc)
    W = np.concatenate(filters, axis=1)
    return W

def apply_filters_and_psd(X_band, W, sfreq, nperseg=256):
    """
    X_band: (n_trials, channels, samples)
    W: (channels, n_filters)
    Returns PSD features: (n_trials, n_filters) where each feature is mean power across Welch
    """
    n_trials = X_band.shape[0]
    n_filters = W.shape[1]
    feats = np.zeros((n_trials, n_filters), dtype=np.float32)
    for t in range(n_trials):
        # filtered channels -> apply W^T @ epoch for each trial
        projected = W.T @ X_band[t]  # (n_filters, samples)
        # for each projected component compute mean PSD
        for f in range(n_filters):
            _, pxx = welch(projected[f], fs=sfreq, nperseg=min(nperseg, projected.shape[1]))
            feats[t, f] = np.mean(pxx)
    return feats

def extract_multiband_dynamic_csp_psd(X, y, sfreq, bands=BAND_DEFS,
                                      n_filters_per_side=2, ema_alpha=0.05,
                                      nperseg=256):
    """
    X: (n_trials, channels, samples)
    y: (n_trials,)
    Returns:
        F: (n_trials, total_features) where total_features = sum(num_filters_per_band)
    """
    band_feats = []
    for bname, (lo, hi) in bands.items():
        Xb = np.zeros_like(X)
        # bandpass each trial (vectorized)
        for i in range(X.shape[0]):
            Xb[i] = bandpass_epoch(X[i], sfreq, lo, hi)
        W = compute_ovr_csp(Xb, y, n_filters_per_side, ema_alpha)
        feats = apply_filters_and_psd(Xb, W, sfreq, nperseg=nperseg)
        band_feats.append(feats)
    F = np.concatenate(band_feats, axis=1)
    return F


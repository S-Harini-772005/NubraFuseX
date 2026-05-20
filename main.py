"""
main.py - Complete Production-level EEG loader, preprocessor, and CSV exporter for BCI IV 2a
Features:
- Loads training and evaluation datasets using updated data_loader.py
- Robust logging & QC of corrupted/truncated files
- Converts data to PyTorch tensors for ML/DL
- Handles CPU/GPU automatically
- Optional plotting of a sample EEG trial
- Exports processed training and evaluation data to CSV for PowerBI
- Supports predicted labels export for evaluation CSV
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_PATH = r"C:\Users\shari\Videos\nubrafusex\data\BCICIV_2a_gdf"
VERBOSE = True
PLOT_SAMPLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
CSV_EXPORT_PATH = r"C:\Users\shari\nubrafusex\output"

# ----------------------------
# LOGGING SETUP
# ----------------------------
logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info(f"Running on device: {DEVICE}")

# ----------------------------
# CREATE OUTPUT DIRECTORY
# ----------------------------
os.makedirs(CSV_EXPORT_PATH, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
try:
    train_loader, eval_loader, X_train_np, y_train_np = prepare_dataloaders(
        data_path=DATA_PATH, batch_size=BATCH_SIZE, plot_trials_flag=PLOT_SAMPLE
    )
    # Load evaluation NumPy array
    X_eval_np, _ = prepare_dataloaders(
        data_path=DATA_PATH, batch_size=BATCH_SIZE, plot_trials_flag=False
    )[2:4]
    logging.info(f"Datasets loaded: X_train={X_train_np.shape}, y_train={y_train_np.shape}, X_eval={X_eval_np.shape}")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    raise SystemExit(1)

# ----------------------------
# CONVERT TO TORCH TENSORS
# ----------------------------
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train_np, dtype=torch.long).to(DEVICE)
X_eval = torch.tensor(X_eval_np, dtype=torch.float32).to(DEVICE)
logging.info(f"Converted training and evaluation data to PyTorch tensors on {DEVICE}")

# ----------------------------
# SAMPLE DATA PLOT
# ----------------------------
if PLOT_SAMPLE:
    trial_idx = 0
    plt.figure(figsize=(12, 6))
    for ch in range(X_train_np.shape[1]):
        plt.plot(X_train_np[trial_idx, ch] + ch * 50, label=f"Ch {ch}")
    plt.title(f"Sample EEG Training Trial #{trial_idx}")
    plt.xlabel("Time samples")
    plt.ylabel("Amplitude (offset for visualization)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

# ----------------------------
# EXPORT TO CSV FOR POWERBI
# ----------------------------
try:
    # ----- Training CSV -----
    n_trials, n_ch, n_times = X_train_np.shape
    X_train_flat = X_train_np.reshape(n_trials, n_ch * n_times)
    df_train = pd.DataFrame(X_train_flat)
    df_train['label'] = y_train_np
    csv_train_file = os.path.join(CSV_EXPORT_PATH, "BCI_IV2a_train.csv")
    df_train.to_csv(csv_train_file, index=False)
    logging.info(f"Exported training data to CSV: {csv_train_file}")

    # ----- Evaluation CSV (with optional predictions placeholder) -----
    n_eval_trials, n_ch, n_times = X_eval_np.shape
    X_eval_flat = X_eval_np.reshape(n_eval_trials, n_ch * n_times)
    df_eval = pd.DataFrame(X_eval_flat)
    df_eval['label'] = np.nan  # No ground-truth labels for evaluation
    df_eval['predicted_label'] = np.nan  # Optional column for predicted labels after inference
    csv_eval_file = os.path.join(CSV_EXPORT_PATH, "BCI_IV2a_eval.csv")
    df_eval.to_csv(csv_eval_file, index=False)
    logging.info(f"Exported evaluation data to CSV: {csv_eval_file}")

except Exception as e:
    logging.error(f"Failed to export CSV: {e}")

# ----------------------------
# DATA READY FOR ML/DL
# ----------------------------
logging.info(f"Total training trials: {X_train.shape[0]}, Channels: {X_train.shape[1]}, Samples: {X_train.shape[2]}")
logging.info(f"Labels range: {y_train.min().item()} - {y_train.max().item()}")
logging.info(f"Total evaluation trials: {X_eval.shape[0]}, Channels: {X_eval.shape[1]}, Samples: {X_eval.shape[2]}")

# ----------------------------
# OPTIONAL: VERIFY ONE BATCH
# ----------------------------
for batch_X, batch_y in train_loader:
    logging.info(f"Sample batch shapes - X: {batch_X.shape}, y: {batch_y.shape}")
    break

for batch_X_eval in eval_loader:
    logging.info(f"Sample evaluation batch shapes - X: {batch_X_eval.shape}")
    break

logging.info("main.py executed successfully - training and evaluation CSV ready for PowerBI")

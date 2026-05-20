"""
data_loader.py - Master Dataset Loader for BCI Competition IV-2a
Features:
- AdvancedMasterDatasetLoader class for CSV-based pipelines (cleaning, saving, training, SHAP)
- Unified prepare_dataloaders function for GDF EEG files
- Handles training (T.gdf) and evaluation (E.gdf) sets
- Safe fallbacks: uses dummy data if EEG utils not available
- Robust logging and PyTorch DataLoader integration
"""

import os
import logging
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Optional SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Explainability will be skipped.")

warnings.filterwarnings("ignore")  # cleaner output

# -------------------- Advanced CSV Loader --------------------
class AdvancedMasterDatasetLoader:
    def __init__(self, file_paths, target_column='event_code', test_size=0.2, random_state=42):
        self.file_paths = file_paths
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.df = self._load_and_merge()
        self._clean_standardize()
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None

    def _load_and_merge(self):
        files = []
        if isinstance(self.file_paths, str) and os.path.isdir(self.file_paths):
            files = [os.path.join(self.file_paths, f) for f in os.listdir(self.file_paths) if f.endswith('.csv')]
        elif isinstance(self.file_paths, list):
            files = self.file_paths
        else:
            raise ValueError("Provide folder path or list of CSV files")

        if len(files) == 0:
            raise FileNotFoundError("No CSV files found in the given path(s).")

        df_list = [pd.read_csv(f) for f in files]
        combined_df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {len(df_list)} files, total rows: {combined_df.shape[0]}")
        return combined_df

    def _clean_standardize(self):
        df = self.df.copy()
        df.drop_duplicates(inplace=True)
        df.dropna(subset=[self.target_column], inplace=True)

        # Fill numeric
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Fill categorical
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

        # Standardize MI labels
        df[self.target_column] = df[self.target_column].astype(str).str.upper()

        def standardize_code(code):
            if 'MI' in code:
                return 'MI'
            elif 'NORMAL' in code:
                return 'NORMAL'
            else:
                return 'OTHER'

        df[self.target_column] = df[self.target_column].apply(standardize_code)
        df = df[df[self.target_column] != 'OTHER']
        self.df = df.reset_index(drop=True)
        print(f"After cleaning: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")

    def split_train_test(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(f"Train size: {self.X_train.shape[0]}, Test size: {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_cleaned_dataset(self, output_path='cleaned_dataset.csv'):
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")

    def train_model(self, n_estimators=200, max_depth=None):
        if self.X_train is None:
            self.split_train_test()
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")
        y_pred = self.model.predict(self.X_test)
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def explain_model(self, sample_size=200):
        if not SHAP_AVAILABLE:
            print("SHAP not installed. Skipping explainability.")
            return
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        sample = self.X_train.sample(min(sample_size, self.X_train.shape[0]), random_state=self.random_state)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(sample)
        print("SHAP values computed.")
        shap.summary_plot(shap_values, sample, show=True)

    def run_full_pipeline(self, save_path='cleaned_dataset.csv', train=True, explain=True):
        self.save_cleaned_dataset(save_path)
        if train:
            self.train_model()
        if explain:
            self.explain_model()

# -------------------- PyTorch Dataset + Loader --------------------
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BCICIVDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# -------------------- Unified Dataloader --------------------
def prepare_dataloaders(data_path, batch_size=32, plot_trials_flag=False):
    """
    Loads GDF EEG data if utils available, else falls back to dummy random data.
    Returns: train_loader, eval_loader, X_train, y_train
    """
    try:
        from src.eeg_utils import load_gdf_files, plot_trials
    except ImportError:
        logging.warning("EEG utils not available. Using dummy data.")
        X_train = np.random.randn(100, 22, 1000)  # fake EEG
        y_train = np.random.randint(0, 2, 100)
        train_dataset = BCICIVDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, None, X_train, y_train

    # Load Training Data
    try:
        X_train, y_train = load_gdf_files(data_path, "T.gdf")
    except Exception as e:
        logging.error(f"T.gdf loading failed: {e}")
        raise

    # Load Evaluation Data (optional)
    try:
        X_eval, _ = load_gdf_files(data_path, "E.gdf")
    except Exception as e:
        logging.warning(f"E.gdf loading failed: {e}")
        X_eval = None

    # Torch Dataset + Loader
    train_dataset = BCICIVDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    eval_loader = None
    if X_eval is not None:
        eval_dataset = BCICIVDataset(X_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Optional Plotting
    if plot_trials_flag:
        try:
            plot_trials(X_train, y_train)
        except Exception as e:
            logging.warning(f"Plotting failed: {e}")

    return train_loader, eval_loader, X_train, y_train


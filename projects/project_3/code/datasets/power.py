# code/datasets/power.py
"""
PowerDataset – Household Electricity Consumption (hourly)
Features: lag1, lag24, lag168, hour, day-of-week
Target: Global_active_power (next hour)
Includes proper time-based splits and test timestamps.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import load_household_power, get_project_root


class PowerDataset:
    def __init__(self, data_dir=None, seed=1993):
        self.seed = seed
        self.data_dir = data_dir or get_project_root() / "data"
        self.scaler = StandardScaler()
        self.load_and_preprocess()

    def load_and_preprocess(self):
        print("Loading and preprocessing household power consumption data...")
        df = load_household_power(self.data_dir)

        # Feature engineering
        df = df.copy()
        df['lag1']   = df['Global_active_power'].shift(1)
        df['lag24']  = df['Global_active_power'].shift(24)   # yesterday same hour
        df['lag168'] = df['Global_active_power'].shift(168)  # one week ago
        df['hour']   = df.index.hour
        df['dow']    = df.index.dayofweek  # 0=Monday

        # Drop rows with NaN (from shifting)
        df = df.dropna().copy()

        # Features and target
        feature_cols = ['lag1', 'lag24', 'lag168', 'hour', 'dow']
        X = df[feature_cols].values
        y = df['Global_active_power'].values.reshape(-1, 1)

        # Time-based split (chronological – no leakage!)
        n = len(df)
        train_end = int(0.70 * n)
        val_end   = int(0.85 * n)

        # Split data
        X_train_raw = X[:train_end]
        X_val_raw   = X[train_end:val_end]
        X_test_raw  = X[val_end:]

        y_train = y[:train_end]
        y_val   = y[train_end:val_end]
        y_test  = y[val_end:]

        # Fit scaler on training data only
        self.scaler.fit(X_train_raw)

        # Transform all splits
        self.X_train = self.scaler.transform(X_train_raw)
        self.X_val   = self.scaler.transform(X_val_raw)
        self.X_test  = self.scaler.transform(X_test_raw)

        self.y_train = y_train
        self.y_val   = y_val
        self.y_test  = y_test

        # Save full info
        self.timestamps_test = df.index[val_end:]
        self.df_full = df
        self.feature_names = feature_cols

        print(f"PowerDataset ready!")
        print(f"   • Train: {len(self.X_train):,} samples ({df.index[:train_end].min().date()} → {df.index[train_end-1].date()})")
        print(f"   • Val:   {len(self.X_val):,} samples")
        print(f"   • Test:  {len(self.X_test):,} samples ({self.timestamps_test[0].date()} → {self.timestamps_test[-1].date()})")
        print(f"   • Features: {', '.join(feature_cols)}")
        print(f"   • Target: Global_active_power (kW)\n")


    # Helper: for creating sequences for LSTM (used in part_c)
    def create_sequences(self, X, y, seq_len=24):
        """
        Create sequences for LSTM training/inference.
        Returns: X_seq (n_samples, seq_len, n_features), y_seq (n_samples,)
        """
        seqs, targets = [], []
        for i in range(len(X) - seq_len):
            seqs.append(X[i:i + seq_len])
            targets.append(y[i + seq_len])
        return np.array(seqs), np.array(targets).ravel()

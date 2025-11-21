# code/datasets/trade.py
"""
TradeDataset – Bilateral Trade Flows (1870–2014)
Task: Predict whether trade between two countries will grow >10% next year
Features: lagged trade flow + previous growth rate
Target: binary (1 if growth >10%, else 0)
Time-based split: train <1990, val 1990–1999, test ≥2000
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import load_trade_data, get_project_root

class TradeDataset:
    def __init__(self, data_dir=None, seed=1993):
        """
        Initialize TradeDataset.
        """
        self.seed = seed
        np.random.seed(seed)

        self.data_dir = data_dir or get_project_root() / "data"
        self.scaler = StandardScaler()

        self.load_and_preprocess()
        print(f"TradeDataset initialized (seed={seed})")


    def load_and_preprocess(self):
        """
        Load data, engineer features, create binary target, and split chronologically.
        """
        print("Loading and preprocessing bilateral trade data...")

        # Load raw data
        df = load_trade_data(self.data_dir)

        # Ensure sorted by time for proper lagging
        df = df.sort_values(['ccode1', 'ccode2', 'year']).reset_index(drop=True)

        # Feature 1: Lagged trade flow (last year's flow)
        df['lag_flow'] = df.groupby(['ccode1', 'ccode2'])['flow'].shift(1)

        # Feature 2: Previous year's growth rate
        df['growth_lag'] = (
            df.groupby(['ccode1', 'ccode2'])['flow']
              .pct_change()
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0)
        )

        # Binary target: 1 if next year's growth >10%, else 0
        # Compute next year's growth first
        df['growth_next'] = df.groupby(['ccode1', 'ccode2'])['flow'].pct_change().shift(-1)
        df['target'] = (df['growth_next'] > 0.10).astype(int)

        # Drop rows where lag_flow is missing (first observation per dyad)
        df = df.dropna(subset=['lag_flow']).copy()

        # Optional: log-transform lag_flow to reduce skewness (highly recommended!)
        df['lag_flow_log'] = np.log1p(df['lag_flow'])  # log(1+x) to handle zeros

        # Final features
        feature_cols = ['lag_flow_log', 'growth_lag']
        X = df[feature_cols].values
        y = df['target'].values.astype(int)  # Ensure integer labels

        # Chronological train/val/test split by year
        train_mask = df['year'] < 1990
        val_mask   = (df['year'] >= 1990) & (df['year'] < 2000)
        test_mask  = df['year'] >= 2000

        # Fit scaler only on training data
        X_train_raw = X[train_mask]
        self.scaler.fit(X_train_raw)

        # Transform all splits
        self.X_train = self.scaler.transform(X[train_mask])
        self.X_val   = self.scaler.transform(X[val_mask])
        self.X_test  = self.scaler.transform(X[test_mask])

        self.y_train = y[train_mask]
        self.y_val   = y[val_mask]
        self.y_test  = y[test_mask]

        # Save full dataframe and test subset for analysis
        self.df_full = df.copy()
        self.df_test = df[test_mask].copy().reset_index(drop=True)

        # Feature names for interpretability (SHAP, plots, etc.)
        self.feature_names = feature_cols

        # Stats
        pos_ratio = self.y_train.mean()
        n_train = len(self.X_train)
        n_val   = len(self.X_val)
        n_test  = len(self.X_test)

        print(f"TradeDataset ready!")
        print(f"   • Train: {n_train:,} samples (before 1990) | Positives: {pos_ratio:.2%}")
        print(f"   • Val:   {n_val:,} samples (1990–1999)")
        print(f"   • Test:  {n_test:,} samples (2000–2014)")
        print(f"   • Features: {', '.join(feature_cols)}")
        print(f"   • Target: 1 if trade growth >10% next year")
        print(f"   • Lag flow is log-transformed for stability\n")

    def get_feature_names(self):
        """Return list of feature names (useful for plots/SHAP)"""
        return self.feature_names

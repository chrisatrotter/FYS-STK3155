# code/utils.py
"""
Utility functions for Project 3 – FYS-STK3155 / FYS4155
- Directory management
- Plot saving
- Project root detection
- Safe data loading (power + trade)
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────
# Global plot style (applied once)
# ──────────────────────────────────────────────────────────────
sns.set(style="whitegrid", font_scale=1.25, palette="deep", rc={
    "grid.alpha": 0.3,
    "axes.labelsize": 13,
    "axes.titlesize": 16,
    "figure.figsize": (12, 7),
    "legend.fontsize": 12,
})
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
})

def breakpoint() -> None:
    print(f"\n{'═' * 90}")

# ──────────────────────────────────────────────────────────────
# Directory & Path Helpers
# ──────────────────────────────────────────────────────────────
def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_plot(fig: plt.Figure | None = None,
              name: str = "plot",
              part: str = "a") -> None:
    """
    Save current or given figure to figures/part_{part}/{name}.pdf
    Automatically closes the figure.
    """
    if fig is None:
        fig = plt.gcf()

    save_path = Path("figures") / f"part_{part}" / f"{name}.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"   → Saved: {save_path}")


def get_project_root() -> Path:
    """
    Robust project root detection.
    Works when:
      - data/ is at project root (project_3/data/)
      - data/ is inside code/     (project_3/code/data/) ← YOUR CASE
    """
    current = Path(__file__).resolve()
    # If we're inside the 'code' directory → root is 'code/'
    if current.parent.name == "code":
        return current.parent
    # Otherwise assume standard layout (code/ is inside root)
    return current.parent.parent


# ──────────────────────────────────────────────────────────────
# Data Loaders – Safe, Informative, No Side Effects
# ──────────────────────────────────────────────────────────────
def load_household_power(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load and preprocess household power consumption dataset.
    Returns hourly resampled DataFrame with Global_active_power.
    """
    data_dir = Path(data_dir or get_project_root() / "data")
    path = data_dir / "household_power_consumption.csv"

    if not path.exists():
        raise FileNotFoundError(f"Power data not found: {path}\n"
                                "   → Download from UCI and place in data/")

    print(f"Loading household power consumption data from: data/household_power_consumption.csv")

    # Read only needed columns
    df = pd.read_csv(
        path,
        sep=';',
        usecols=['Date', 'Time', 'Global_active_power'],
        na_values='?',
        dtype={'Global_active_power': 'float32'},
        low_memory=False
    )

    # Combine date and time → datetime index
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        dayfirst=True,
        errors='coerce'
    )

    # Drop parsing failures and set index
    df = df.dropna(subset=['datetime', 'Global_active_power'])
    df = df.set_index('datetime')[['Global_active_power']]

    # Resample to hourly mean, forward-fill small gaps, then interpolate
    df = df.resample('h').mean()
    df = df.interpolate(method='linear')

    print(f"Power data loaded → {len(df):,} hourly observations "
          f"({df.index.min().date()} → {df.index.max().date()})")
    return df


def load_trade_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load COW Dyadic trade dataset (v4.0).
    Returns clean DataFrame with flow and target.
    """
    data_dir = Path(data_dir or get_project_root() / "data")
    path = data_dir / "Dyadic_COW_4.0.csv"

    if not path.exists():
        raise FileNotFoundError(f"Trade data not found: {path}\n"
                                "   → Download from Correlates of War and place in data/")

    print(f"Loading bilateral trade data from: data/Dyadic_COW_4.0.csv")

    df = pd.read_csv(
        path,
        usecols=['ccode1', 'ccode2', 'year', 'flow1', 'flow2'],
        dtype={'ccode1': int, 'ccode2': int, 'year': int}
    )

    # Average directed flows (flow1: 1→2, flow2: 2→1)
    df['flow'] = df[['flow1', 'flow2']].mean(axis=1)

    # Clean and sort
    df = df[['ccode1', 'ccode2', 'year', 'flow']].dropna()
    df = df[df['flow'] > 0]  # Remove zero-trade dyads
    df = df.sort_values(['ccode1', 'ccode2', 'year']).reset_index(drop=True)

    # Growth next year
    df['growth_next'] = df.groupby(['ccode1', 'ccode2'])['flow'].pct_change().shift(-1)

    # Binary target: 1 if next year growth >10%
    df['target'] = (df['growth_next'] > 0.10).astype(int)

    # Drop last year per dyad (no future → no target)
    df = df.dropna(subset=['target', 'growth_next']).reset_index(drop=True)

    n_obs = len(df)
    pos_rate = df['target'].mean()

    print(f"Trade data loaded → {n_obs:,} dyad-years "
          f"| Years: {df['year'].min()}–{df['year'].max()} "
          f"| >10% growth: {pos_rate:.2%}")
    return df

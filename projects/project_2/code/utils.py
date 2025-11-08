# utils.py
"""
Utility toolbox for Project 2 – Neural Networks from Scratch.
Combines Project-1 helpers (Runge data, polynomial features, CSV I/O)
with Project-2 helpers (MNIST, scaling, saving .npz, plotting).
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import fetch_openml

# ----------------------------------------------------------------------
# 1. GLOBAL SETTINGS
# ----------------------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 300

# ----------------------------------------------------------------------
# 2. RUNGE FUNCTION & DATA GENERATION (Project 1 + Project 2)
# ----------------------------------------------------------------------
def runge_function(x: np.ndarray) -> np.ndarray:
    """True Runge function f(x) = 1/(1+25x²)."""
    return 1.0 / (1.0 + 25.0 * x**2)


def generate_runge_data(
    n_samples: int = 500,
    noise: float = 0.05,
    seed: int = 1993,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Runge data **without** automatic split.
    Returns X (n_samples,1) and y (n_samples,1) – exactly what
    ``run_project2.py`` expects.

    Parameters
    ----------
    n_samples : int
        Number of points.
    noise : float
        Std-dev of Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X, y : np.ndarray
        Feature matrix and target vector.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n_samples)
    y_clean = runge_function(x)
    y = y_clean + rng.normal(0.0, noise, n_samples)
    return x.reshape(-1, 1), y.reshape(-1, 1)


def save_runge_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_samples: int,
    save_dir: str,
) -> None:
    """Save Runge splits as **CSV** (kept from Project 1)."""
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(X_train, columns=["x"]).to_csv(
        os.path.join(save_dir, f"X_train_n{n_samples}.csv"), index=False
    )
    pd.DataFrame(X_test, columns=["x"]).to_csv(
        os.path.join(save_dir, f"X_test_n{n_samples}.csv"), index=False
    )
    pd.DataFrame(y_train, columns=["y"]).to_csv(
        os.path.join(save_dir, f"y_train_n{n_samples}.csv"), index=False
    )
    pd.DataFrame(y_test, columns=["y"]).to_csv(
        os.path.join(save_dir, f"y_test_n{n_samples}.csv"), index=False
    )


def load_runge_data(n_samples: int, data_dir: str) -> Tuple[np.ndarray, ...]:
    """Load the CSV files created by ``save_runge_data``."""
    X_train = pd.read_csv(os.path.join(data_dir, f"X_train_n{n_samples}.csv")).values
    X_test = pd.read_csv(os.path.join(data_dir, f"X_test_n{n_samples}.csv")).values
    y_train = pd.read_csv(os.path.join(data_dir, f"y_train_n{n_samples}.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, f"y_test_n{n_samples}.csv")).values.ravel()
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# 3. POLYNOMIAL FEATURES (Project 1)
# ----------------------------------------------------------------------
def create_polynomial_features(
    X: np.ndarray, degree: int, include_bias: bool = False
) -> np.ndarray:
    """Wrap ``PolynomialFeatures`` – handy for OLS/Ridge/Lasso."""
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    return poly.fit_transform(X)


# ----------------------------------------------------------------------
# 4. MNIST LOADING (Project 2)
# ----------------------------------------------------------------------
def load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST from OpenML, scale pixels to [0,1] and return
    ``X (70000,784) float32`` and ``y (70000,) int32``.
    """
    print("Downloading MNIST (this may take a moment)...")
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="auto",
    )
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int32)
    print(f"MNIST loaded – {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


# ----------------------------------------------------------------------
# 5. TRAIN/TEST SPLIT + OPTIONAL SCALING (Project 2)
# ----------------------------------------------------------------------
def split_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    scale: bool = True,
    seed: int = 1993,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if y.ndim == 1 else None,
    )

    scaler: Optional[StandardScaler] = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return X_train, X_test, y_train, y_test, scaler


# ----------------------------------------------------------------------
# 6. SAVE .npz (used by ``run_project2.py``)
# ----------------------------------------------------------------------
def save_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    name: str,
    data_dir: str,
) -> None:
    """
    Store a train/test split in a compressed ``.npz`` file.
    ``run_project2.py`` calls this for both Runge and MNIST.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, name)
    np.savez_compressed(
        path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    print(f"Data saved to {path}")


# ----------------------------------------------------------------------
# 7. PLOTTING HELPERS
# ----------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    """Create directory if missing and return the full path."""
    os.makedirs(path, exist_ok=True)
    return path


def plot_learning_curve(
    history: list[float],
    title: str = "Learning Curve",
    save_path: Optional[str] = None,
) -> None:
    """Plot metric (MSE or accuracy) vs. epoch."""
    plt.figure(figsize=(8, 5))
    plt.plot(history, linewidth=2, color="tab:blue")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE" if "MSE" in title else "Accuracy")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning curve → {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Confusion matrix for MNIST (10×10)."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.title("MNIST Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix → {save_path}")
    plt.close()


def create_heatmap(
    data: np.ndarray,
    x_labels: list,
    y_labels: list,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str] = None,
    annot: bool = True,
) -> None:
    """Generic heatmap (e.g. λ₁ vs λ₂ or η vs λ)."""
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        data,
        annot=annot,
        fmt=".3f" if annot else "",
        cmap="viridis",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={"label": "MSE"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap → {save_path}")
    plt.close()


def plot_runge_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_samples: int,
    save_dir: str,
) -> None:
    """Scatter plot of Runge data + true function (Project 1)."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, c="steelblue", label="Train", alpha=0.6, s=20)
    plt.scatter(X_test, y_test, c="crimson", marker="x", label="Test", s=30)

    x_smooth = np.linspace(-1, 1, 1000).reshape(-1, 1)
    plt.plot(x_smooth, runge_function(x_smooth), "k-", linewidth=2, label="True Runge")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Runge Function Data (n={n_samples})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(save_dir, f"runge_data_n{n_samples}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Runge data plot → {path}")
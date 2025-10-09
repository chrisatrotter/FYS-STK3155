import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os

def runge_function(x: np.ndarray) -> np.ndarray:
    """Calculate the 1D Runge function."""
    return 1 / (1 + 25 * x**2)

def generate_runge_data(n_samples: int, noise_sigma: float = 0.05, random_state: int = None) -> tuple:
    """Generate Runge function data with optional Gaussian noise."""
    if random_state is not None:
        np.random.seed(random_state)
    x = np.random.uniform(-1, 1, n_samples)
    y = runge_function(x) + noise_sigma * np.random.randn(n_samples)
    X = x.reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=random_state)

def save_runge_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
                    n_samples: int, save_dir: str):
    """Save Runge data to CSV files."""
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame(X_train, columns=['x']).to_csv(os.path.join(save_dir, f'X_train_n{n_samples}.csv'), index=False)
    pd.DataFrame(X_test, columns=['x']).to_csv(os.path.join(save_dir, f'X_test_n{n_samples}.csv'), index=False)
    pd.DataFrame(y_train, columns=['y']).to_csv(os.path.join(save_dir, f'y_train_n{n_samples}.csv'), index=False)
    pd.DataFrame(y_test, columns=['y']).to_csv(os.path.join(save_dir, f'y_test_n{n_samples}.csv'), index=False)

def load_runge_data(n_samples: int, data_dir: str) -> tuple:
    """Load Runge data from CSV files."""
    X_train = pd.read_csv(os.path.join(data_dir, f'X_train_n{n_samples}.csv')).values
    X_test = pd.read_csv(os.path.join(data_dir, f'X_test_n{n_samples}.csv')).values
    y_train = pd.read_csv(os.path.join(data_dir, f'y_train_n{n_samples}.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, f'y_test_n{n_samples}.csv')).values.ravel()
    return X_train, X_test, y_train, y_test

def scale_features(X_train: np.ndarray, X_test: np.ndarray, scaler: StandardScaler = None) -> tuple:
    """Scale features using provided scaler or return unscaled."""
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    return X_train, X_test

def create_polynomial_features(X: np.ndarray, degree: int, include_bias: bool = False) -> np.ndarray:
    """Create polynomial features for given degree."""
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    return poly.fit_transform(X)

def plot_runge_data(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
                    n_samples: int, save_dir: str):
    """Plot the initial Runge data (train and test) with the true Runge function."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    # Plot training and test data
    plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.5, s=20)
    plt.scatter(X_test, y_test, color='red', label='Test Data', marker='x', s=20)
    
    # Plot true Runge function
    x_smooth = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y_true = runge_function(x_smooth)
    plt.plot(x_smooth, y_true, 'k-', label='True Runge Function')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Runge Function Data (n={n_samples})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'runge_data_n{n_samples}.png'), dpi=300)
    plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.preprocessing import StandardScaler
from utils import load_runge_data, scale_features, create_polynomial_features
from regression_methods import RegressionModel

def run_ols(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
             degree: int, scaler: StandardScaler = None) -> dict:
    """Run OLS regression for a given degree."""
    X_train_poly = create_polynomial_features(X_train, degree)
    X_test_poly = create_polynomial_features(X_test, degree)
    X_train_poly, X_test_poly = scale_features(X_train_poly, X_test_poly, scaler)

    model = RegressionModel()
    try:
        theta = model.ols_fit(X_train_poly, y_train)
        y_train_pred = model.predict(X_train_poly, theta)
        y_test_pred = model.predict(X_test_poly, theta)
        mse_train, r2_train = model.compute_metrics(y_train, y_train_pred)
        mse_test, r2_test = model.compute_metrics(y_test, y_test_pred)
        return {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'theta': theta
        }
    except np.linalg.LinAlgError:
        print(f"[Warning] OLS failed at degree {degree}")
        return {
            'mse_train': np.nan,
            'mse_test': np.nan,
            'r2_train': np.nan,
            'r2_test': np.nan,
            'theta': np.zeros(X_train_poly.shape[1])
        }

def plot_results(degrees: range, results: dict, n_samples: int, save_dir: str):
    """Plot MSE, R2, and coefficient progression in a single figure with subplots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    ax1.plot(degrees, results['mse_train'], 'o-', label='Train MSE')
    ax1.plot(degrees, results['mse_test'], 's-', label='Test MSE')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title(f'Runge Function: OLS MSE (n={n_samples})')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(degrees, results['r2_train'], 'o-', label='Train R²')
    ax2.plot(degrees, results['r2_test'], 's-', label='Test R²')
    ax2.set_ylabel('R² Score')
    ax2.set_title(f'Runge Function: OLS R² (n={n_samples})')
    ax2.legend()
    ax2.grid(True)
    
    max_betas = max(len(results['theta'][d-1]) for d in degrees)
    betas_to_plot = min(10, max_betas)
    for beta_idx in range(betas_to_plot):
        beta_vals = [results['theta'][d-1][beta_idx] if beta_idx < len(results['theta'][d-1]) else np.nan 
                     for d in degrees]
        ax3.plot(degrees, beta_vals, 'o-', label=f'β{beta_idx}', markersize=3)
    ax3.set_xlabel('Polynomial Degree')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title(f'Runge Function: OLS Coefficient Progression (n={n_samples})')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'runge_ols_combined_n{n_samples}.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="OLS regression on Runge function")
    parser.add_argument("-n", "--nsamples", type=int, default=100, help="Number of samples")
    parser.add_argument("-d", "--degree", type=int, default=15, help="Max polynomial degree")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level")
    parser.add_argument("--noscale", action="store_true", help="Disable feature scaling")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing saved data")
    args = parser.parse_args()

    if not (0 <= args.noise <= 1):
        raise ValueError("Noise must be between 0 and 1")
    if args.degree <= 0:
        raise ValueError("Polynomial degree must be positive")

    scaler = None if args.noscale else StandardScaler()
    n_samples_list = [100, 500, 1000]
    degrees = range(1, args.degree + 1)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures/part_a")
    res_dir = os.path.join(os.path.dirname(__file__), "results/part_a")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    for n_samples in n_samples_list:
        X_train, X_test, y_train, y_test = load_runge_data(n_samples, args.data_dir)
        ols_results = {'mse_train': [], 'mse_test': [], 'r2_train': [], 'r2_test': [], 'theta': []}
        
        for d in degrees:
            result = run_ols(X_train, X_test, y_train, y_test, d, scaler=scaler)
            ols_results['mse_train'].append(result['mse_train'])
            ols_results['mse_test'].append(result['mse_test'])
            ols_results['r2_train'].append(result['r2_train'])
            ols_results['r2_test'].append(result['r2_test'])
            ols_results['theta'].append(result['theta'])

        plot_results(degrees, ols_results, n_samples, fig_dir)
        pd.DataFrame({
            'Degree': degrees,
            'MSE_Train': ols_results['mse_train'],
            'MSE_Test': ols_results['mse_test'],
            'R2_Train': ols_results['r2_train'],
            'R2_Test': ols_results['r2_test']
        }).to_csv(os.path.join(res_dir, f'runge_ols_n{n_samples}.csv'), index=False)

if __name__ == '__main__':
    main()

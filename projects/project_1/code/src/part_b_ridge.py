import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.preprocessing import StandardScaler
from utils import load_runge_data, scale_features, create_polynomial_features
from regression_methods import RegressionModel

def run_ridge(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
              degree: int, lambda_val: float, scaler: StandardScaler = None) -> dict:
    """Run Ridge regression for a given degree and lambda."""
    X_train_poly = create_polynomial_features(X_train, degree)
    X_test_poly = create_polynomial_features(X_test, degree)
    X_train_poly, X_test_poly = scale_features(X_train_poly, X_test_poly, scaler)

    if not (np.all(np.isfinite(X_train_poly)) and np.all(np.isfinite(X_test_poly))):
        print(f"[Warning] Non-finite values detected in scaled features at degree {degree}, lambda={lambda_val}")
        return {
            'mse_train': np.nan,
            'mse_test': np.nan,
            'r2_train': np.nan,
            'r2_test': np.nan,
            'theta': np.zeros(X_train_poly.shape[1])
        }

    model = RegressionModel()
    try:
        theta = model.ridge_fit(X_train_poly, y_train, lambda_val)
        if not np.all(np.isfinite(theta)):
            print(f"[Warning] Non-finite coefficients detected at degree {degree}, lambda={lambda_val}")
            return {
                'mse_train': np.nan,
                'mse_test': np.nan,
                'r2_train': np.nan,
                'r2_test': np.nan,
                'theta': np.zeros(X_train_poly.shape[1])
            }
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
        print(f"[Warning] Ridge failed at degree {degree}, lambda={lambda_val}")
        return {
            'mse_train': np.nan,
            'mse_test': np.nan,
            'r2_train': np.nan,
            'r2_test': np.nan,
            'theta': np.zeros(X_train_poly.shape[1])
        }

def plot_results(degrees: range, results: dict, n_samples: int, lambdas: list, save_dir: str):
    """Plot MSE, R², and coefficient progression in a single figure with subplots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(10, 12))
    
    ax1 = plt.subplot(3, 1, 1)
    for lam in lambdas:
        mse_vals = [results[(d, lam)]['mse_test'] for d in degrees]
        ax1.plot(degrees, mse_vals, 'o-', label=f'λ={lam}')
    ax1.set_ylabel('Test MSE')
    ax1.set_title(f'Runge Function: Ridge Test MSE (n={n_samples})')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(3, 1, 2)
    for lam in lambdas:
        r2_vals = [results[(d, lam)]['r2_test'] for d in degrees]
        ax2.plot(degrees, r2_vals, 'o-', label=f'λ={lam}')
    ax2.set_ylabel('Test R²')
    ax2.set_title(f'Runge Function: Ridge Test R² (n={n_samples})')
    ax2.legend()
    ax2.grid(True)
    
    ax3 = plt.subplot(3, 1, 3)
    max_betas = min(5, max(len(results[(d, lambdas[0])]['theta']) for d in degrees))
    colors = ['b', 'r', 'g', 'm', 'c']
    for lam_idx, lam in enumerate(lambdas):
        for beta_idx in range(max_betas):
            beta_vals = [results[(d, lam)]['theta'][beta_idx] if beta_idx < len(results[(d, lam)]['theta']) else np.nan 
                         for d in degrees]
            ax3.plot(degrees, beta_vals, color=colors[beta_idx % len(colors)], linestyle=['-', '--', ':', '-.'][lam_idx % 4], 
                     label=f'β{beta_idx}, λ={lam}', markersize=3)
    ax3.set_xlabel('Polynomial Degree')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title(f'Runge Function: Ridge Coefficient Progression (n={n_samples})')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'runge_ridge_combined_n{n_samples}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Ridge regression on Runge function")
    parser.add_argument("-n", "--nsamples", type=int, default=100, help="Number of samples")
    parser.add_argument("-d", "--degree", type=int, default=10, help="Max polynomial degree")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level")
    parser.add_argument("--lambdas", nargs='+', type=float, default=[0.01, 0.1, 1.0, 10.0], help="Regularization parameters")
    parser.add_argument("--noscale", action="store_true", help="Disable feature scaling")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing saved data")
    args = parser.parse_args()

    if not (0 <= args.noise <= 1):
        raise ValueError("Noise must be between 0 and 1")
    if args.degree <= 0:
        raise ValueError("Polynomial degree must be positive")
    if args.degree > 10:
        print(f"[Warning] Maximum polynomial degree {args.degree} is high. Consider reducing degree to avoid numerical instability.")

    scaler = None if args.noscale else StandardScaler()
    n_samples_list = [100, 500, 1000]
    degrees = range(1, args.degree + 1)
    lambdas = args.lambdas

    fig_dir = os.path.join(os.path.dirname(__file__), "figures/part_b")
    res_dir = os.path.join(os.path.dirname(__file__), "results/part_b")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    for n_samples in n_samples_list:
        X_train, X_test, y_train, y_test = load_runge_data(n_samples, args.data_dir)
        ridge_results = {(d, lam): run_ridge(X_train, X_test, y_train, y_test, d, lam, scaler) 
                        for d in degrees for lam in lambdas}
        
        plot_results(degrees, ridge_results, n_samples, lambdas, fig_dir)
        pd.DataFrame([
            {'Degree': d, 'Lambda': lam, 'MSE_Train': ridge_results[(d, lam)]['mse_train'], 
             'MSE_Test': ridge_results[(d, lam)]['mse_test'], 
             'R2_Train': ridge_results[(d, lam)]['r2_train'], 
             'R2_Test': ridge_results[(d, lam)]['r2_test']}
            for d in degrees for lam in lambdas
        ]).to_csv(os.path.join(res_dir, f'runge_ridge_n{n_samples}.csv'), index=False)

if __name__ == '__main__':
    main()

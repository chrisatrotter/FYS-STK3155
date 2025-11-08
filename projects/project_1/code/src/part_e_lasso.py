import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.preprocessing import StandardScaler
from utils import load_runge_data, scale_features, create_polynomial_features
from regression_methods import RegressionModel

def run_lasso(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
              degree: int, lambda_val: float = 0.01, eta: float = 0.00001, scaler: StandardScaler = None, 
              use_gd: bool = False) -> dict:
    """Run Lasso regression for a given degree, using analytical or GD solution."""
    X_train_poly = create_polynomial_features(X_train, degree)
    X_test_poly = create_polynomial_features(X_test, degree)
    X_train_poly, X_test_poly = scale_features(X_train_poly, X_test_poly, scaler)

    model = RegressionModel()
    if use_gd:
        theta, epochs = model.gd_fit(X_train_poly, y_train, eta=eta, lambda_val=lambda_val, regression_type='lasso')
    else:
        theta = model.lasso_fit(X_train_poly, y_train, lambda_val)
        epochs = None
    y_pred = model.predict(X_test_poly, theta)
    mse, r2 = model.compute_metrics(y_test, y_pred)
    return {'mse': mse, 'r2': r2, 'epochs': epochs, 'theta': theta}

def plot_results(degrees: range, results: dict, n_samples: int, save_dir: str, use_gd: bool):
    """Plot MSE, R², and coefficient progression."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    ax1.plot(degrees, results['mse'], 'o-', label='Test MSE')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title(f'Runge Function: Lasso MSE (n={n_samples}, GD={use_gd})')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(degrees, results['r2'], 'o-', label='Test R²')
    ax2.set_ylabel('R² Score')
    ax2.set_title(f'Runge Function: Lasso R² (n={n_samples}, GD={use_gd})')
    ax2.legend()
    ax2.grid(True)
    
    max_betas = max(len(results['theta'][d-1]) for d in degrees)
    betas_to_plot = min(5, max_betas)
    for beta_idx in range(betas_to_plot):
        beta_vals = [results['theta'][d-1][beta_idx] if beta_idx < len(results['theta'][d-1]) else np.nan 
                     for d in degrees]
        ax3.plot(degrees, beta_vals, 'o-', label=f'β{beta_idx}', markersize=3)
    ax3.set_xlabel('Polynomial Degree')
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title(f'Runge Function: Lasso Coefficient Progression (n={n_samples}, GD={use_gd})')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'runge_lasso_combined_n{n_samples}_gd{use_gd}.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Lasso regression on Runge function")
    parser.add_argument("-n", "--nsamples", type=int, default=100, help="Number of samples")
    parser.add_argument("-d", "--degree", type=int, default=15, help="Max polynomial degree")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level")
    parser.add_argument("--lambda_val", type=float, default=0.01, help="Regularization parameter")
    parser.add_argument("--eta", type=float, default=0.00001, help="Learning rate for GD")
    parser.add_argument("--noscale", action="store_true", help="Disable feature scaling")
    parser.add_argument("--use_gd", action="store_true", help="Use gradient descent for Lasso")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing saved data")
    args = parser.parse_args()

    if not (0 <= args.noise <= 1):
        raise ValueError("Noise must be between 0 and 1")
    if args.degree <= 0:
        raise ValueError("Polynomial degree must be positive")
    if args.lambda_val < 0:
        raise ValueError("Lambda must be non-negative")
    if args.eta <= 0 and args.use_gd:
        raise ValueError("Learning rate must be positive when using GD")

    scaler = None if args.noscale else StandardScaler()
    n_samples_list = [100, 500, 1000]
    degrees = range(1, args.degree + 1)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures/part_e")
    res_dir = os.path.join(os.path.dirname(__file__), "results/part_e")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    for n_samples in n_samples_list:
        X_train, X_test, y_train, y_test = load_runge_data(n_samples, args.data_dir)
        lasso_results = {'mse': [], 'r2': [], 'theta': []}
        
        for d in degrees:
            result = run_lasso(X_train, X_test, y_train, y_test, d, lambda_val=args.lambda_val, 
                              eta=args.eta, scaler=scaler, use_gd=args.use_gd)
            lasso_results['mse'].append(result['mse'])
            lasso_results['r2'].append(result['r2'])
            lasso_results['theta'].append(result['theta'])
            if args.use_gd:
                print(f"Runge, Lasso GD, degree={d}, n_samples={n_samples}, Epochs: {result['epochs']}, MSE: {result['mse']:.4f}")
            else:
                print(f"Runge, Lasso Analytical, degree={d}, n_samples={n_samples}, MSE: {result['mse']:.4f}")

        plot_results(degrees, lasso_results, n_samples, fig_dir, args.use_gd)
        pd.DataFrame({
            'Degree': degrees,
            'MSE_Lasso': lasso_results['mse'],
            'R2_Lasso': lasso_results['r2']
        }).to_csv(os.path.join(res_dir, f'runge_lasso_n{n_samples}_gd{args.use_gd}.csv'), index=False)

if __name__ == '__main__':
    main()

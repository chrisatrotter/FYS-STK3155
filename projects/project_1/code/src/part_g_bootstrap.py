import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.preprocessing import StandardScaler
from utils import load_runge_data, scale_features, create_polynomial_features
from regression_methods import RegressionModel

def bootstrap(X: np.ndarray, y: np.ndarray, degree: int, n_bootstraps: int = 100, 
              scaler: StandardScaler = None) -> dict:
    """Perform bootstrap resampling for OLS."""
    model = RegressionModel()
    X_poly = create_polynomial_features(X, degree)
    X_poly = scale_features(X_poly, X_poly, scaler)[0]
    
    y_pred_boot = np.zeros((y.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        indices = np.random.choice(len(y), len(y), replace=True)
        X_boot, y_boot = X_poly[indices], y[indices]
        try:
            theta = model.ols_fit(X_boot, y_boot)
            y_pred_boot[:, i] = model.predict(X_poly, theta)
        except np.linalg.LinAlgError:
            y_pred_boot[:, i] = np.nan

    mse = np.nanmean((y[:, np.newaxis] - y_pred_boot) ** 2, axis=1).mean()
    bias = np.nanmean((y[:, np.newaxis] - np.nanmean(y_pred_boot, axis=1)[:, np.newaxis]) ** 2)
    variance = np.nanmean(np.nanvar(y_pred_boot, axis=1))
    return {'mse': mse, 'bias': bias, 'variance': variance}

def plot_results(degrees: range, results: dict, n_samples: int, save_dir: str):
    """Plot MSE, bias, and variance in a single figure."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, results['mse'], 'o-', label='MSE')
    plt.plot(degrees, results['bias'], 's-', label='Bias²')
    plt.plot(degrees, results['variance'], '^-', label='Variance')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error')
    plt.title(f'Runge Function: Bias-Variance Trade-off (n={n_samples})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'runge_bootstrap_n{n_samples}.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Bootstrap analysis on Runge function")
    parser.add_argument("-n", "--nsamples", type=int, default=100, help="Number of samples")
    parser.add_argument("-d", "--degree", type=int, default=15, help="Max polynomial degree")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level")
    parser.add_argument("--n_bootstraps", type=int, default=100, help="Number of bootstrap samples")
    parser.add_argument("--noscale", action="store_true", help="Disable feature scaling")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing saved data")
    args = parser.parse_args()

    if not (0 <= args.noise <= 1):
        raise ValueError("Noise must be between 0 and 1")
    if args.degree <= 0:
        raise ValueError("Polynomial degree must be positive")
    if args.n_bootstraps <= 0:
        raise ValueError("Number of bootstraps must be positive")

    scaler = None if args.noscale else StandardScaler()
    n_samples_list = [100, 500, 1000]
    degrees = range(1, args.degree + 1)

    fig_dir = os.path.join(os.path.dirname(__file__), "figures/part_g")
    res_dir = os.path.join(os.path.dirname(__file__), "results/part_g")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    for n_samples in n_samples_list:
        X_train, X_test, y_train, y_test = load_runge_data(n_samples, args.data_dir)
        X_full = np.vstack((X_train, X_test))
        y_full = np.hstack((y_train, y_test))
        bootstrap_results = {'mse': [], 'bias': [], 'variance': []}
        
        for d in degrees:
            result = bootstrap(X_full, y_full, d, n_bootstraps=args.n_bootstraps, scaler=scaler)
            bootstrap_results['mse'].append(result['mse'])
            bootstrap_results['bias'].append(result['bias'])
            bootstrap_results['variance'].append(result['variance'])
            print(f"Runge, Bootstrap, degree={d}, n_samples={n_samples}, MSE: {result['mse']:.4f}, "
                  f"Bias²: {result['bias']:.4f}, Variance: {result['variance']:.4f}")

        plot_results(degrees, bootstrap_results, n_samples, fig_dir)
        pd.DataFrame({
            'Degree': degrees,
            'MSE': bootstrap_results['mse'],
            'Bias2': bootstrap_results['bias'],
            'Variance': bootstrap_results['variance']
        }).to_csv(os.path.join(res_dir, f'runge_bootstrap_n{n_samples}.csv'), index=False)

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from utils import load_runge_data, scale_features, create_polynomial_features
from regression_methods import RegressionModel

def run_cv(X: np.ndarray, y: np.ndarray, degree: int, k: int = 5, lambda_val: float = 0.01, 
           scaler: StandardScaler = None, regression_type: str = 'ols') -> dict:
    """Run k-fold cross-validation for OLS, Ridge, or Lasso."""
    X_poly = create_polynomial_features(X, degree)
    X_poly = scale_features(X_poly, X_poly, scaler)[0]

    model = RegressionModel()
    kf = KFold(n_splits=k, shuffle=True, random_state=1993)
    mse_scores = []
    
    for train_idx, test_idx in kf.split(X_poly):
        X_train, X_test = X_poly[train_idx], X_poly[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        try:
            if regression_type == 'ols':
                theta = model.ols_fit(X_train, y_train)
            elif regression_type == 'ridge':
                theta = model.ridge_fit(X_train, y_train, lambda_val)
            elif regression_type == 'lasso':
                theta = model.lasso_fit(X_train, y_train, lambda_val)
            y_pred = model.predict(X_test, theta)
            mse, _ = model.compute_metrics(y_test, y_pred)
            mse_scores.append(mse)
        except np.linalg.LinAlgError:
            mse_scores.append(np.nan)

    return {'mse': np.nanmean(mse_scores)}

def plot_results(degrees: range, results: dict, n_samples: int, save_dir: str):
    """Plot cross-validation MSE for different regression types in a single figure."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for reg_type in ['ols', 'ridge', 'lasso']:
        mse_vals = [results[(d, reg_type)]['mse'] for d in degrees]
        plt.plot(degrees, mse_vals, label=reg_type.upper())
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validation MSE')
    plt.title(f'Runge Function: Cross-Validation MSE (n={n_samples})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'runge_cv_n{n_samples}.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Cross-validation on Runge function")
    parser.add_argument("-n", "--nsamples", type=int, default=100, help="Number of samples")
    parser.add_argument("-d", "--degree", type=int, default=15, help="Max polynomial degree")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--lambda_val", type=float, default=0.01, help="Regularization parameter for Ridge/Lasso")
    parser.add_argument("--noscale", action="store_true", help="Disable feature scaling")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing saved data")
    args = parser.parse_args()

    if not (0 <= args.noise <= 1):
        raise ValueError("Noise must be between 0 and 1")
    if args.degree <= 0:
        raise ValueError("Polynomial degree must be positive")
    if args.k <= 1:
        raise ValueError("Number of folds must be at least 2")
    if args.lambda_val < 0:
        raise ValueError("Lambda must be non-negative")

    scaler = None if args.noscale else StandardScaler()
    n_samples_list = [100, 500, 1000]
    degrees = range(1, args.degree + 1)
    regression_types = ['ols', 'ridge', 'lasso']

    fig_dir = os.path.join(os.path.dirname(__file__), "figures/part_h")
    res_dir = os.path.join(os.path.dirname(__file__), "results/part_h")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    for n_samples in n_samples_list:
        X_train, X_test, y_train, y_test = load_runge_data(n_samples, args.data_dir)
        X_full = np.vstack((X_train, X_test))
        y_full = np.hstack((y_train, y_test))
        cv_results = {(d, reg_type): run_cv(X_full, y_full, d, k=args.k, lambda_val=args.lambda_val, scaler=scaler, regression_type=reg_type) 
                      for d in degrees for reg_type in regression_types}
        
        for d in degrees:
            for reg_type in regression_types:
                print(f"Runge, CV, {reg_type.upper()}, degree={d}, n_samples={n_samples}, MSE: {cv_results[(d, reg_type)]['mse']:.4f}")

        plot_results(degrees, cv_results, n_samples, fig_dir)
        pd.DataFrame([
            {'Degree': d, 'Regression': reg_type, 'MSE': cv_results[(d, reg_type)]['mse']}
            for d in degrees for reg_type in regression_types
        ]).to_csv(os.path.join(res_dir, f'runge_cv_n{n_samples}.csv'), index=False)

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.preprocessing import StandardScaler
from utils import load_runge_data, scale_features, create_polynomial_features
from regression_methods import RegressionModel

def run_sgd(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
            degree: int, method: str, eta: float = 0.00001, lambda_val: float = 0.01, 
            scaler: StandardScaler = None, regression_type: str = 'ridge') -> dict:
    """Run stochastic gradient descent for a given method and regression type."""
    X_train_poly = create_polynomial_features(X_train, degree)
    X_test_poly = create_polynomial_features(X_test, degree)
    X_train_poly, X_test_poly = scale_features(X_train_poly, X_test_poly, scaler)

    model = RegressionModel()
    theta, epochs = model.sgd_fit(X_train_poly, y_train, eta=eta, lambda_val=lambda_val, 
                                  method=method, regression_type=regression_type)
    y_pred = model.predict(X_test_poly, theta)
    mse, r2 = model.compute_metrics(y_test, y_pred)
    return {'mse': mse, 'r2': r2, 'epochs': epochs}

def plot_results(methods: list, results: dict, n_samples: int, save_dir: str, regression_type: str):
    """Plot MSE and epochs for different SGD methods."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.bar(methods, [results[regression_type][m]['mse'][0] for m in methods])
    ax1.set_ylabel('Test MSE')
    ax1.set_title(f'Runge Function: SGD {regression_type.upper()} MSE (n={n_samples})')
    ax1.grid(True)
    
    ax2.bar(methods, [results[regression_type][m]['epochs'][0] for m in methods])
    ax2.set_ylabel('Epochs')
    ax2.set_title(f'Runge Function: SGD {regression_type.upper()} Epochs (n={n_samples})')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'runge_sgd_{regression_type}_n{n_samples}.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Stochastic gradient descent on Runge function")
    parser.add_argument("-n", "--nsamples", type=int, default=100, help="Number of samples")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level")
    parser.add_argument("--eta", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--noscale", action="store_true", help="Disable feature scaling")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing saved data")
    args = parser.parse_args()

    if not (0 <= args.noise <= 1):
        raise ValueError("Noise must be between 0 and 1")
    if args.eta <= 0:
        raise ValueError("Learning rate must be positive")

    scaler = None if args.noscale else StandardScaler()
    n_samples_list = [100, 500, 1000]
    sgd_methods = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']
    regression_types = ['ols', 'ridge', 'lasso']

    fig_dir = os.path.join(os.path.dirname(__file__), "figures/part_f")
    res_dir = os.path.join(os.path.dirname(__file__), "results/part_f")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    for n_samples in n_samples_list:
        X_train, X_test, y_train, y_test = load_runge_data(n_samples, args.data_dir)
        results = {reg_type: {method: {'mse': [], 'r2': [], 'epochs': []} for method in sgd_methods} 
                   for reg_type in regression_types}
        
        for reg_type in regression_types:
            for method in sgd_methods:
                result = run_sgd(X_train, X_test, y_train, y_test, degree=5, method=method, 
                                 eta=args.eta, scaler=scaler, regression_type=reg_type)
                results[reg_type][method]['mse'].append(result['mse'])
                results[reg_type][method]['r2'].append(result['r2'])
                results[reg_type][method]['epochs'].append(result['epochs'])
                print(f"Runge, SGD Method: {method}, Type: {reg_type}, n_samples={n_samples}, Epochs: {result['epochs']}, MSE: {result['mse']:.4f}")
            
            plot_results(sgd_methods, results, n_samples, fig_dir, reg_type)

        pd.DataFrame([
            {'Regression': reg_type, 'Method': method, 'MSE': results[reg_type][method]['mse'][0], 
             'R2': results[reg_type][method]['r2'][0], 
             'Epochs': results[reg_type][method]['epochs'][0]}
            for reg_type in regression_types for method in sgd_methods
        ]).to_csv(os.path.join(res_dir, f'sgd_n{n_samples}.csv'), index=False)

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler
from utils import load_runge_data, scale_features, create_polynomial_features
from regression_methods import RegressionModel

def run_advanced_gd(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
                    degree: int, method: str, eta: float = 0.00001, lambda_val: float = 0.01, 
                    scaler: StandardScaler = None) -> dict:
    """Run advanced gradient descent for a given method."""
    X_train_poly = create_polynomial_features(X_train, degree)
    X_test_poly = create_polynomial_features(X_test, degree)
    X_train_poly, X_test_poly = scale_features(X_train_poly, X_test_poly, scaler)

    model = RegressionModel()
    theta, epochs = model.gd_fit(X_train_poly, y_train, eta=eta, lambda_val=lambda_val, method=method, 
                                 regression_type='ridge')
    y_pred = model.predict(X_test_poly, theta)
    mse, _ = model.compute_metrics(y_test, y_pred)
    return {'mse': mse, 'epochs': epochs}

def main():
    parser = argparse.ArgumentParser(description="Advanced gradient descent on Runge function")
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
    gd_methods = ['vanilla', 'momentum', 'adagrad', 'rmsprop', 'adam']

    res_dir = os.path.join(os.path.dirname(__file__), "results/part_d")
    os.makedirs(res_dir, exist_ok=True)

    for n_samples in n_samples_list:
        X_train, X_test, y_train, y_test = load_runge_data(n_samples, args.data_dir)
        results = {method: {'mse': [], 'epochs': []} for method in gd_methods}
        
        for method in gd_methods:
            result = run_advanced_gd(X_train, X_test, y_train, y_test, degree=5, method=method, 
                                    eta=args.eta, scaler=scaler)
            results[method]['mse'].append(result['mse'])
            results[method]['epochs'].append(result['epochs'])
            print(f"Runge, GD Method: {method}, n_samples={n_samples}, Epochs: {result['epochs']}, MSE: {result['mse']:.4f}")

        pd.DataFrame({
            'Method': gd_methods,
            'MSE': [results[m]['mse'][0] for m in gd_methods],
            'Epochs': [results[m]['epochs'][0] for m in gd_methods]
        }).to_csv(os.path.join(res_dir, f'gd_advanced_n{n_samples}.csv'), index=False)

if __name__ == '__main__':
    main()

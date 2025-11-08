import subprocess
import argparse
import os
from utils import generate_runge_data, save_runge_data, plot_runge_data

def run_script(script_name: str, args: list, part: str) -> None:
    """Run a Python script with the given arguments and handle errors."""
    args = [arg for arg in args if arg]
    print(f"Running {part} ({script_name})...")
    try:
        result = subprocess.run(['python3', script_name] + args, check=True, capture_output=True, text=True)
        print(f"{part} completed successfully.")
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors for {part}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {part}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Run all parts of Project 1 (OLS, Ridge, GD, etc.) on Runge function")
    parser.add_argument("-n", "--nsamples", type=int, default=100, help="Number of samples (default: 100)")
    parser.add_argument("-d", "--degree", type=int, default=15, help="Max polynomial degree (default: 15)")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level for normal distribution (default: 0.05)")
    parser.add_argument("--eta", type=float, default=0.00001, help="Learning rate for GD/SGD (default: 0.00001)")
    parser.add_argument("--lambdas", nargs='+', type=float, default=[0.01, 0.1, 1.0, 10.0], help="Regularization parameters for Ridge (default: [0.01, 0.1, 1.0, 10.0])")
    parser.add_argument("--lambda_val", type=float, default=0.01, help="Regularization parameter for Lasso/CV (default: 0.01)")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross-validation (default: 5)")
    parser.add_argument("--n_bootstraps", type=int, default=100, help="Number of bootstrap samples (default: 100)")
    parser.add_argument("--noscale", action="store_true", help="Disable feature scaling (default: False)")
    parser.add_argument("--use_gd", action="store_true", help="Use gradient descent for Lasso (default: False)")
    args = parser.parse_args()

    # Validate arguments
    if not (0 <= args.noise <= 1):
        raise ValueError("Noise must be between 0 and 1")
    if args.degree <= 0:
        raise ValueError("Polynomial degree must be positive")
    if args.eta <= 0:
        raise ValueError("Learning rate must be positive")
    if args.lambda_val < 0:
        raise ValueError("Lambda must be non-negative")
    if args.k <= 1:
        raise ValueError("Number of folds must be at least 2")
    if args.n_bootstraps <= 0:
        raise ValueError("Number of bootstraps must be positive")

    # Base directory for data, results, and figures
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    fig_dir = os.path.join(base_dir, "figures/data")
    for part in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        os.makedirs(os.path.join(base_dir, f"results/part_{part}"), exist_ok=True)
        if part in ['a', 'b', 'e', 'f', 'g', 'h']:  # Parts with plots
            os.makedirs(os.path.join(base_dir, f"figures/part_{part}"), exist_ok=True)

    # Generate, save, and plot data for each n_samples
    n_samples_list = [100, 500, 1000]
    for n_samples in n_samples_list:
        X_train, X_test, y_train, y_test = generate_runge_data(n_samples=n_samples, noise_sigma=args.noise, random_state=1993)
        save_runge_data(X_train, X_test, y_train, y_test, n_samples, data_dir)
        plot_runge_data(X_train, X_test, y_train, y_test, n_samples, fig_dir)

    # Common arguments
    common_args = [
        f"--nsamples={args.nsamples}",
        f"--noise={args.noise}",
        f"--noscale" if args.noscale else "",
        f"--data_dir={data_dir}"
    ]

    # Run each part with specific arguments
    run_script("part_a_ols.py", common_args + [f"--degree={args.degree}"], "Part A (OLS)")
    run_script("part_b_ridge.py", common_args + [f"--degree={args.degree}", f"--lambdas"] + [str(l) for l in args.lambdas], "Part B (Ridge)")
    run_script("part_c_gd.py", common_args + [f"--eta={args.eta}"], "Part C (Gradient Descent)")
    run_script("part_d_gd_advanced.py", common_args + [f"--eta={args.eta}"], "Part D (Advanced GD)")
    run_script("part_e_lasso.py", common_args + [f"--degree={args.degree}", f"--lambda_val={args.lambda_val}", f"--eta={args.eta}", f"--use_gd" if args.use_gd else ""], "Part E (Lasso)")
    run_script("part_f_sgd.py", common_args + [f"--eta={args.eta}"], "Part F (Stochastic GD)")
    run_script("part_g_bootstrap.py", common_args + [f"--degree={args.degree}", f"--n_bootstraps={args.n_bootstraps}"], "Part G (Bootstrap)")
    run_script("part_h_cv.py", common_args + [f"--degree={args.degree}", f"--k={args.k}", f"--lambda_val={args.lambda_val}"], "Part H (Cross-Validation)")

if __name__ == '__main__':
    main()
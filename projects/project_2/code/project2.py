# project2.py
import argparse
import os
import sys
import subprocess
from typing import List, Dict

from utils import generate_runge_data, save_data, load_mnist, split_scale
from generated_data import plot_runge_data

# ----------------------------------------------------------------------
# Helper – run a part script
# ----------------------------------------------------------------------
def run_script(script_name: str, part_args: List[str], part: str) -> None:
    part_args = [a for a in part_args if a]  # drop empty strings
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    print(f"\n{'=' * 60}")
    print(f"RUNNING {part.upper()} → {script_name}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [sys.executable, script_path] + part_args,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr.strip():
            print("Warnings/Stderr:\n", result.stderr)
    except FileNotFoundError:
        print(f"ERROR: Script not found → {script_path}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {part}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # 1. Argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Project 2 – FFNN on Runge + MNIST")
    parser.add_argument("-n", "--nsamples", type=int, default=1000, help="Runge samples")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise in Runge")
    parser.add_argument("--hidden", nargs="+", type=int, default=[50, 100], help="Hidden nodes (Part B)")
    parser.add_argument("--degree", type=int, default=10, help="Polynomial degree from Project 1")
    parser.add_argument("--act", type=str, default="relu", choices=["sigmoid", "relu", "leaky_relu"], help="Activation (Part D)")
    parser.add_argument("--eta", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="NN epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization (Part E)")
    parser.add_argument("--l1", type=float, default=0.0, help="L1 regularization (Part E)")
    parser.add_argument("--seed", type=int, default=1993, help="Random seed")
    parser.add_argument("--part", type=str, choices=["a", "b", "c", "d", "e", "f", "g"],
                        help="Run only this part")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Paths & data
    # ------------------------------------------------------------------
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Runge data (all regression parts)
    runge_file = f"runge_{args.nsamples}.npz"
    X, y = generate_runge_data(n_samples=args.nsamples, noise=args.noise, seed=args.seed)
    X_train, X_test, y_train, y_test, _ = split_scale(X, y, scale=True, seed=args.seed)
    save_data(X_train, X_test, y_train, y_test, runge_file, data_dir)

    # Plot the Runge data
    plot_runge_data(X, y, data_dir, args.nsamples, args.noise)

    # MNIST data (classification parts)
    mnist_file = "mnist_scaled.npz"
    X_mnist, y_mnist = load_mnist()
    X_train_m, X_test_m, y_train_m, y_test_m, _ = split_scale(X_mnist, y_mnist, scale=True, seed=args.seed)
    save_data(X_train_m, X_test_m, y_train_m, y_test_m, mnist_file, data_dir)

    # ------------------------------------------------------------------
    # 3. Base argument sets
    # ------------------------------------------------------------------
    # Common to regression (a–e)
    base_regression = [
        f"--data_dir={data_dir}",
        f"--runge_file={runge_file}",
        f"--figures_dir={figures_dir}",
        f"--seed={args.seed}",
        f"--epochs={args.epochs}",
        f"--batch={args.batch}",
        f"--eta={args.eta}",
    ]

    # Common to classification (f–g)
    base_classification = [
        f"--data_dir={data_dir}",
        f"--mnist_file={mnist_file}",
        f"--figures_dir={figures_dir}",
        f"--seed={args.seed}",
        f"--epochs={args.epochs}",
        f"--batch={args.batch}",
        f"--eta={args.eta}",
    ]

    # ------------------------------------------------------------------
    # 4. Part-specific arguments
    # ------------------------------------------------------------------
    parts: Dict[str, List[str]] = {
        # Part A: analytical (no data needed beyond dir)
        "a": [f"--data_dir={data_dir}"],

        # Part B: architecture search
        "b": [
            f"--data_dir={data_dir}",
            f"--runge_file={runge_file}",
            f"--figures_dir={figures_dir}",
            f"--seed={args.seed}",
            f"--epochs={args.epochs}",
            f"--batch={args.batch}",
            f"--degree=10",
            "--hidden"
        ] + [str(h) for h in args.hidden],

        # Part C: validation curves
        "c": base_regression,

        # Part D: activation functions
        "d": base_regression + [f"--act={args.act}"],

        # Part E: regularization
        "e": base_regression + [f"--l1={args.l1}", f"--l2={args.l2}"],

        # Part F: classification (MNIST)
        "f": base_classification,

        # Part G: evaluation (same as F)
        "g": base_classification,
    }

    # ------------------------------------------------------------------
    # 5. Script name mapping
    # ------------------------------------------------------------------
    script_map = {
        "a": "part_a_analytical.py",
        "b": "part_b_regression.py",
        "c": "part_c_validation.py",
        "d": "part_d_activation.py",
        "e": "part_e_regularized.py",
        "f": "part_f_classification.py",
        "g": "part_g_evaluation.py",
    }

    # ------------------------------------------------------------------
    # 6. Run selected parts
    # ------------------------------------------------------------------
    to_run = [args.part.lower()] if args.part else list(parts.keys())
    print(f"\nRunning {'ONLY Part ' + args.part.upper() if args.part else 'ALL parts (a–g)'}")

    for letter in to_run:
        script = script_map[letter]
        part_args = parts[letter]
        run_script(script, part_args, f"Part {letter.upper()}")

if __name__ == "__main__":
    main()

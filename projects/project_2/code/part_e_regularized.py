# part_e_regularized.py
# Part E: L1 + L2 Regularization + Comparison with Ridge/Lasso (Project 1)
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress Lasso(alpha=0) warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from models import NeuralNetwork, Adam, RegressionModel
from utils import create_polynomial_features

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Part E – L1/L2 Regularization + Project 1 Comparison")
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--runge_file", type=str, required=True)
parser.add_argument("--figures_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--eta", type=float, default=0.001)
parser.add_argument("--l1", type=float, default=0.0)
parser.add_argument("--l2", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=1993)
args = parser.parse_args()

# ------------------------------------------------------------------
# Load Runge data
# ------------------------------------------------------------------
runge_path = os.path.join(args.data_dir, args.runge_file)
data = np.load(runge_path)
X_train_raw = data["X_train"]
X_test_raw  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

# Polynomial features (degree 5) – same as Project 1
degree = 5
X_train_poly = create_polynomial_features(X_train_raw, degree=degree, include_bias=True)
X_test_poly  = create_polynomial_features(X_test_raw,  degree=degree, include_bias=True)

# ------------------------------------------------------------------
# Best architecture from Part D
# ------------------------------------------------------------------
layers = [X_train_poly.shape[1], 100, 100, 1]
activations = ["leaky_relu", "leaky_relu", "linear"]

# ------------------------------------------------------------------
# Prepare output directories
# ------------------------------------------------------------------
fig_dir = os.path.join(args.figures_dir, "part_e")
data_out_dir = os.path.join(args.data_dir, "part_e")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_out_dir, exist_ok=True)

# ------------------------------------------------------------------
# Grid search over λ₁, λ₂, η
# ------------------------------------------------------------------
l1_vals = [0.0, 1e-5, 1e-4, 1e-3]
l2_vals = [0.0, 1e-5, 1e-4, 1e-3]
eta_vals = [1e-4, 1e-3, 1e-2]

results = []

print("\nGrid search over λ₁, λ₂, η...")
for l1 in l1_vals:
    for l2 in l2_vals:
        for eta in eta_vals:
            print(f"  → λ₁={l1:.0e}, λ₂={l2:.0e}, η={eta:.0e}...", end=" ")

            nn = NeuralNetwork(
                layers=layers,
                activations=activations,
                cost="mse",
                seed=args.seed
            )

            history = nn.fit(
                X_train_poly, y_train,
                epochs=args.epochs,
                batch_size=args.batch,
                optimizer=Adam(eta=eta),
                l1=l1,
                l2=l2,
                verbose=False
            )

            y_train_pred = nn.predict(X_train_poly)
            y_test_pred  = nn.predict(X_test_poly)

            mse_train = np.mean((y_train - y_train_pred) ** 2)
            mse_test  = np.mean((y_test - y_test_pred) ** 2)

            results.append({
                "l1": l1,
                "l2": l2,
                "eta": eta,
                "mse_train": mse_train,
                "mse_test": mse_test,
                "history": history
            })

            print(f"Test MSE: {mse_test:.6f}")

# ------------------------------------------------------------------
# Find best NN model
# ------------------------------------------------------------------
best = min(results, key=lambda x: x["mse_test"])
print(f"\nBest NN: λ₁={best['l1']:.0e}, λ₂={best['l2']:.0e}, η={best['eta']:.0e} → Test MSE: {best['mse_test']:.6f}")

# ------------------------------------------------------------------
# Compare with Project 1: Ridge + Lasso
# ------------------------------------------------------------------
ridge_lambdas = [1e-5, 1e-4, 1e-3, 1e-2]  # Skip 0.0 to avoid warning
lasso_lambdas = [1e-5, 1e-4, 1e-3]        # Skip 0.0

ridge_results = []
lasso_results = []

print("\nRunning Ridge (closed-form)...")
for lam in ridge_lambdas:
    theta = RegressionModel.ridge_fit(X_train_poly, y_train, lambda_val=lam)
    y_pred = RegressionModel.predict(X_test_poly, theta)
    mse = np.mean((y_test - y_pred) ** 2)
    ridge_results.append({"lambda": lam, "mse": mse})
    print(f"  Ridge λ={lam:.0e} → MSE: {mse:.6f}")

print("Running Lasso (scikit-learn)...")
for lam in lasso_lambdas:
    theta = RegressionModel.lasso_fit(X_train_poly, y_train, lambda_val=lam)
    y_pred = RegressionModel.predict(X_test_poly, theta)
    mse = np.mean((y_test - y_pred) ** 2)
    lasso_results.append({"lambda": lam, "mse": mse})
    print(f"  Lasso λ={lam:.0e} → MSE: {mse:.6f}")

best_ridge = min(ridge_results, key=lambda x: x["mse"])
best_lasso = min(lasso_results, key=lambda x: x["mse"])

print(f"\nBest Ridge: λ={best_ridge['lambda']:.0e} → MSE: {best_ridge['mse']:.6f}")
print(f"Best Lasso: λ={best_lasso['lambda']:.0e} → MSE: {best_lasso['mse']:.6f}")

# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------

# 1. NN Heatmap: λ₁ vs λ₂ (best η)
df_nn = pd.DataFrame(results)
df_pivot = df_nn[df_nn["eta"] == best["eta"]].pivot_table(
    values="mse_test", index="l1", columns="l2", aggfunc="min"
)
plt.figure(figsize=(7, 5))
sns.heatmap(df_pivot, annot=True, fmt=".4f", cmap="viridis", cbar_kws={"label": "Test MSE"})
plt.title(f"NN Test MSE (η={best['eta']:.0e})")
plt.xlabel("λ₂ (L2)")
plt.ylabel("λ₁ (L1)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "nn_l1_l2_heatmap.pdf"), dpi=300)
plt.close()
print(f"Heatmap → {os.path.join(fig_dir, 'nn_l1_l2_heatmap.pdf')}")

# 2. Ridge vs Lasso vs NN
plt.figure(figsize=(8, 5))
plt.semilogx([r["lambda"] for r in ridge_results], [r["mse"] for r in ridge_results],
             marker='o', label="Ridge (Project 1)", color="tab:blue")
plt.semilogx([r["lambda"] for r in lasso_results], [r["mse"] for r in lasso_results],
             marker='s', label="Lasso (Project 1)", color="tab:orange")
plt.axhline(best["mse_test"], color="tab:green", linestyle="--", label=f"Best NN ({best['mse_test']:.4f})")
plt.xlabel("λ")
plt.ylabel("Test MSE")
plt.title("Ridge vs Lasso vs Best NN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "ridge_lasso_vs_nn.pdf"), dpi=300)
plt.close()
print(f"Comparison plot → {os.path.join(fig_dir, 'ridge_lasso_vs_nn.pdf')}")

# 3. Learning curve of best NN
plt.figure(figsize=(8, 5))
plt.plot(best["history"], label="Best NN (Train MSE)", color="tab:green")
plt.axhline(best["mse_test"], color="tab:red", linestyle="--", label=f"Test MSE: {best['mse_test']:.4f}")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"Best NN Learning Curve\nλ₁={best['l1']:.0e}, λ₂={best['l2']:.0e}, η={best['eta']:.0e}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "best_nn_learning.pdf"), dpi=300)
plt.close()
print(f"Learning curve → {os.path.join(fig_dir, 'best_nn_learning.pdf')}")

# 4. Save CSV summary (data/part_e)
summary = {
    "method": ["Best NN", "Best Ridge", "Best Lasso"],
    "l1": [best["l1"], 0.0, best_lasso["lambda"]],
    "l2": [best["l2"], best_ridge["lambda"], 0.0],
    "eta": [best["eta"], None, None],
    "test_mse": [best["mse_test"], best_ridge["mse"], best_lasso["mse"]]
}
csv_path = os.path.join(data_out_dir, "comparison_summary.csv")
pd.DataFrame(summary).to_csv(csv_path, index=False)
print(f"Summary CSV → {csv_path}")

print("\nAll results saved in:")
print(f"  Figures → {fig_dir}")
print(f"  Data    → {data_out_dir}")
print("Part E completed.")

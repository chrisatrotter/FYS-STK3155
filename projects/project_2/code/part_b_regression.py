# part_b_regression.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from models import NeuralNetwork, SGD, RMSprop, Adam, RegressionModel

# ------------------------------------------------------------------
# Argument parsing – ONLY the arguments that project2.py passes
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Part B – OLS vs NN (Sigmoid) on Runge")
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--runge_file", type=str, required=True)
parser.add_argument("--figures_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--seed", type=int, default=1993)
parser.add_argument("--degree", type=int, default=10, help="Polynomial degree from Project 1")
parser.add_argument("--eta", type=float, default=0.01, help="Learning rate (used only for ADAM default)")
parser.add_argument("--hidden", nargs="+", type=int, default=[100, 100], help="Hidden nodes per layer")
args = parser.parse_args()

# ------------------------------------------------------------------
# Load Runge data (already split & scaled in project2.py)
# ------------------------------------------------------------------
data = np.load(os.path.join(args.data_dir, args.runge_file))
X_train_raw = data["X_train"]
X_test_raw  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

# ------------------------------------------------------------------
# 1. Polynomial features (same as Project 1)
# ------------------------------------------------------------------
poly = PolynomialFeatures(degree=args.degree, include_bias=True)
X_train_poly = poly.fit_transform(X_train_raw)
X_test_poly  = poly.transform(X_test_raw)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_poly)
X_test  = scaler.transform(X_test_poly)

# ------------------------------------------------------------------
# 2. OLS baseline
# ------------------------------------------------------------------
theta_ols = RegressionModel.ols_fit(X_train, y_train)
y_ols = RegressionModel.predict(X_test, theta_ols)
mse_ols, r2_ols = RegressionModel.compute_metrics(y_test, y_ols)
print(f"\n[OLS] MSE: {mse_ols:.6f} | R²: {r2_ols:.6f}")

# ------------------------------------------------------------------
# 3. NN architectures (Part B requirement)
# ------------------------------------------------------------------
configs = []
for nodes in args.hidden:
    if len(args.hidden) == 1:
        # one hidden layer
        layers = [X_train.shape[1], nodes, 1]
        act    = ["sigmoid", "linear"]
    else:
        # two hidden layers (same size)
        layers = [X_train.shape[1], nodes, nodes, 1]
        act    = ["sigmoid", "sigmoid", "linear"]
    configs.append({"name": f"{len(layers)-2}×{nodes}", "layers": layers, "act": act})

# ------------------------------------------------------------------
# 4. Optimizers & learning rates (sweep)
# ------------------------------------------------------------------
optimizers = {
    "SGD":     lambda eta: SGD(eta=eta),
    "RMSprop": lambda eta: RMSprop(eta=eta),
    "ADAM":    lambda eta: Adam(eta=eta),
}
etas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

# ------------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------------
results = []
learning_curves = {}
rng = np.random.default_rng(args.seed)

for cfg in configs:
    print(f"\n=== {cfg['name']} ===")
    for opt_name, opt_factory in optimizers.items():
        for eta in etas:
            print(f"  → {opt_name} η={eta:.0e}", end="")
            nn = NeuralNetwork(
                layers=cfg["layers"],
                activations=cfg["act"],
                cost="mse",
                seed=args.seed,
            )
            optimizer = opt_factory(eta)
            history = nn.fit(
                X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch,
                optimizer=optimizer,
                l1=0.0, l2=0.0,
                verbose=False,
            )
            y_pred = nn.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            print(f" | MSE: {mse:.6f}")
            results.append({
                "arch": cfg["name"],
                "optimizer": opt_name,
                "eta": eta,
                "mse": mse,
            })
            # keep learning curve for best ADAM run
            if opt_name == "ADAM" and eta == 1e-3:
                learning_curves[cfg["name"]] = history

# ------------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------------
out_dir = os.path.join(args.data_dir, "part_b")
fig_dir = os.path.join(args.figures_dir, "part_b")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

df = pd.DataFrame(results)
csv_path = os.path.join(out_dir, "nn_vs_ols.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved → {csv_path}")

# ------------------------------------------------------------------
# 7. Plot learning curves (ADAM η=1e-3)
# ------------------------------------------------------------------
plt.figure(figsize=(10, 5))
for name, hist in learning_curves.items():
    plt.plot(hist, label=f"{name} (ADAM η=1e-3)")
plt.axhline(y=mse_ols, color='k', linestyle='--', label=f"OLS MSE = {mse_ols:.6f}")
plt.title("Part B – Learning Curves (Sigmoid Hidden, Linear Output)")
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.legend()
plt.grid(True, alpha=0.3)
curve_path = os.path.join(fig_dir, "learning_curves.pdf")
plt.savefig(curve_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Learning curves → {curve_path}")

# ------------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------------
best = df.loc[df["mse"].idxmin()]
print("\n" + "="*60)
print("BEST NEURAL NETWORK")
print(f"Architecture : {best['arch']}")
print(f"Optimizer    : {best['optimizer']}")
print(f"η            : {best['eta']:.0e}")
print(f"Test MSE     : {best['mse']:.6f}")
print(f"vs OLS MSE   : {mse_ols:.6f} ({100*(best['mse']-mse_ols)/mse_ols:+.2f}%)")
print("="*60)

print("\nPart B done.")

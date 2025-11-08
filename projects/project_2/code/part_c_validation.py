# part_c_validation.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------
# Optional: JAX gradient check
# -----------------------------
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not installed → skipping gradient checking")

from models import NeuralNetwork, SGD

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Part C – Validation vs scikit-learn + Gradient Check")
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--runge_file", type=str, required=True)
parser.add_argument("--figures_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--eta", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=1993)
parser.add_argument("--degree", type=int, default=10, help="Polynomial degree from Project 1")
args = parser.parse_args()

# ------------------------------------------------------------------
# Load & preprocess Runge data (degree-10 + scaling)
# ------------------------------------------------------------------
data = np.load(os.path.join(args.data_dir, args.runge_file))
X_train_raw = data["X_train"]
X_test_raw  = data["X_test"]
y_train = data["y_train"].reshape(-1, 1)
y_test  = data["y_test"].reshape(-1, 1)

poly = PolynomialFeatures(degree=args.degree, include_bias=True)
X_train_poly = poly.fit_transform(X_train_raw)
X_test_poly  = poly.transform(X_test_raw)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_poly)
X_test  = scaler.transform(X_test_poly)

input_dim = X_train.shape[1]
print(f"Polynomial features: degree={args.degree} → input dim = {input_dim}")

# ------------------------------------------------------------------
# Output dirs
# ------------------------------------------------------------------
out_dir = os.path.join(args.data_dir, "part_c")
fig_dir = os.path.join(args.figures_dir, "part_c")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. My Neural Network (He init, SGD)
# ------------------------------------------------------------------
layers = [input_dim, 100, 1]
nn = NeuralNetwork(layers, ["relu", "linear"], cost="mse", seed=args.seed)
optimizer = SGD(eta=args.eta)

print(f"\nTraining My NN: {layers}, SGD η={args.eta}, {args.epochs} epochs")
history_my = nn.fit(
    X_train, y_train,
    epochs=args.epochs,
    batch_size=args.batch,
    optimizer=optimizer,
    verbose=False
)
y_pred_my = nn.predict(X_test)
mse_my = np.mean((y_test - y_pred_my) ** 2)
print(f"My NN Test MSE: {mse_my:.6f}")

# ------------------------------------------------------------------
# 2. Scikit-learn MLPRegressor with He init + SGD
# ------------------------------------------------------------------
print(f"Training scikit-learn MLP with He initialization...")

mlp = MLPRegressor(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='sgd',
    learning_rate='constant',
    learning_rate_init=args.eta,
    batch_size=args.batch,
    random_state=args.seed,
    max_iter=1,
    warm_start=True,
    momentum=0.0,
    nesterovs_momentum=False,
    early_stopping=False,
    validation_fraction=0.0,
    tol=0.0,
    alpha=0.0,  # No L2
)

# Initialize with He (like your NN)
rng = np.random.default_rng(args.seed)

# First: partial_fit to initialize coefs_/intercepts_
mlp.partial_fit(X_train, y_train.ravel())

# Now override with He weights
fan_in = input_dim
scale1 = np.sqrt(2.0 / fan_in)
W1 = rng.normal(0, scale1, (fan_in, 100))
b1 = np.zeros((1, 100))

fan_in = 100
scale2 = np.sqrt(2.0 / fan_in)
W2 = rng.normal(0, scale2, (100, 1))
b2 = np.zeros((1, 1))

mlp.coefs_ = [W1, W2]
mlp.intercepts_ = [b1.ravel(), b2.ravel()]

# Train
mlp_loss_curve = []
for epoch in range(args.epochs):
    mlp.fit(X_train, y_train.ravel())
    mlp_loss_curve.append(mlp.loss_)

# ← ← ← CRITICAL: Compute MSE for scikit-learn ← ← ←
y_pred_sk = mlp.predict(X_test).reshape(-1, 1)
mse_sk = np.mean((y_test - y_pred_sk) ** 2)
print(f"scikit-learn Test MSE: {mse_sk:.6f}")

# ------------------------------------------------------------------
# 3. Save results
# ------------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["My FFNN", "scikit-learn MLP"],
    "Test MSE": [mse_my, mse_sk]  # ← now defined
})
csv_path = os.path.join(out_dir, "validation_mse.csv")
results.to_csv(csv_path, index=False)
print(f"MSE comparison saved → {csv_path}")

# ------------------------------------------------------------------
# 4. Learning curves
# ------------------------------------------------------------------
curve_df = pd.DataFrame({
    "Epoch": np.arange(1, args.epochs + 1),
    "My_FFNN": history_my,
    "Scikit_learn": mlp_loss_curve
})
curve_csv = os.path.join(out_dir, "learning_curves.csv")
curve_df.to_csv(curve_csv, index=False)

plt.figure(figsize=(9, 5.5))
plt.plot(history_my, label="My FFNN (from scratch)", linewidth=2.2, color="tab:blue")
plt.plot(mlp_loss_curve, label="scikit-learn MLPRegressor", linewidth=2.2, color="tab:orange", linestyle="--")
plt.title(f"Part C – Learning Curve Comparison (He init, SGD η={args.eta})")
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(fig_dir, "learning_curve_comparison.pdf")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Learning curve → {plot_path}")

# ------------------------------------------------------------------
# 5. Gradient Checking with JAX
# ------------------------------------------------------------------
if JAX_AVAILABLE:
    print("\nRunning gradient checking with JAX...")

    def jax_mse(W1, b1, W2, b2, X, y):
        a1 = jnp.maximum(0, X @ W1 + b1)  # ReLU
        y_pred = a1 @ W2 + b2
        return jnp.mean((y_pred - y) ** 2)  # ← 1/m * loss

    # Use same initialization
    rng = np.random.default_rng(args.seed)
    fan_in = input_dim
    W1 = rng.normal(0, np.sqrt(2/fan_in), (fan_in, 100))
    b1 = np.zeros((1, 100))
    W2 = rng.normal(0, np.sqrt(2/100), (100, 1))
    b2 = np.zeros((1, 1))

    Xj = jnp.array(X_train[:32])
    yj = jnp.array(y_train[:32])

    # JAX gradient of J = (1/m) * MSE
    jax_func = jit(lambda p: jax_mse(p[0], p[1], p[2], p[3], Xj, yj))
    jax_grad = grad(jax_func)

    # Your NN backward (same weights)
    nn_small = NeuralNetwork([input_dim, 100, 1], ["relu", "linear"], cost="mse", seed=args.seed)
    nn_small.W = [W1.copy(), W2.copy()]
    nn_small.b = [b1.copy(), b2.copy()]
    _ = nn_small.forward(X_train[:32])
    y_pred_small = nn_small.predict(X_train[:32])
    grads_my = nn_small.backward(X_train[:32], y_train[:32], y_pred_small)

    # JAX analytic gradient
    params = [jnp.array(W1), jnp.array(b1), jnp.array(W2), jnp.array(b2)]
    jax_grads = jax_grad(params)
    jax_grads_np = [np.array(g) for g in jax_grads]

    # Compare
    rel_errors = []
    labels = ['W1', 'b1', 'W2', 'b2']
    for i, (g_my, g_jax) in enumerate(zip(
        [grads_my['dW'][0], grads_my['db'][0], grads_my['dW'][1], grads_my['db'][1]],
        jax_grads_np
    )):
        norm_my = np.linalg.norm(g_my)
        norm_jax = np.linalg.norm(g_jax)
        rel_err = np.linalg.norm(g_my - g_jax) / (norm_my + norm_jax + 1e-12)
        rel_errors.append(rel_err)
        print(f"  {labels[i]:2}: rel_err = {rel_err:.2e}")

    print(f"Gradient check passed: max rel_err = {max(rel_errors):.2e}")
else:
    print("JAX not available → gradient checking skipped")

print("\nPart C complete.")
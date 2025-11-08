# part_d_activation.py
# Part D: Test activation functions + network depth/width
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import NeuralNetwork, Adam

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Part D – Activation & Depth Study")
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--runge_file", type=str, required=True)
parser.add_argument("--figures_dir", type=str, required=True)
parser.add_argument("--act", type=str, default="relu", choices=["sigmoid", "relu", "leaky_relu"])
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--eta", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=1993)
args = parser.parse_args()

# ------------------------------------------------------------------
# Load Runge data
# ------------------------------------------------------------------
runge_path = os.path.join(args.data_dir, args.runge_file)
data = np.load(runge_path)
X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

# ------------------------------------------------------------------
# Prepare output directories
# ------------------------------------------------------------------
fig_dir = os.path.join(args.figures_dir, "part_d")
data_out_dir = os.path.join(args.data_dir, "part_d")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_out_dir, exist_ok=True)

# ------------------------------------------------------------------
# Configurations to test
# ------------------------------------------------------------------
activations = ["sigmoid", "relu", "leaky_relu"]
depths = [1, 2, 3]           # number of hidden layers
widths = [50, 100, 200]      # nodes per hidden layer

results = []

# ------------------------------------------------------------------
# Train & evaluate
# ------------------------------------------------------------------
for act in activations:
    print(f"\nTesting activation: {act.upper()}")
    for depth in depths:
        for width in widths:
            print(f"  → Depth {depth}, Width {width}...", end=" ")

            # Build layer list: [1, width, width, ..., 1]
            layers = [1] + [width] * depth + [1]
            act_list = [act] * depth + ["linear"]

            nn = NeuralNetwork(
                layers=layers,
                activations=act_list,
                cost="mse",
                seed=args.seed
            )

            history = nn.fit(
                X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch,
                optimizer=Adam(eta=args.eta),
                verbose=False
            )

            # Train & test MSE
            y_train_pred = nn.predict(X_train)
            y_test_pred = nn.predict(X_test)
            mse_train = np.mean((y_train - y_train_pred) ** 2)
            mse_test  = np.mean((y_test - y_test_pred) ** 2)

            # Overfitting indicator
            overfit = mse_test - mse_train

            results.append({
                "act": act,
                "depth": depth,
                "width": width,
                "mse_train": mse_train,
                "mse_test": mse_test,
                "overfit": overfit,
                "history": history  # for plotting
            })

            print(f"Train MSE: {mse_train:.5f}, Test MSE: {mse_test:.5f}, Δ={overfit:.5f}")

# ------------------------------------------------------------------
# Save results to CSV (data/part_d)
# ------------------------------------------------------------------
df = pd.DataFrame([{
    "act": r["act"],
    "depth": r["depth"],
    "width": r["width"],
    "mse_train": r["mse_train"],
    "mse_test": r["mse_test"],
    "overfit": r["overfit"]
} for r in results])
csv_path = os.path.join(data_out_dir, "results.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved → {csv_path}")

# ------------------------------------------------------------------
# Plot: MSE vs Width (per activation & depth)
# ------------------------------------------------------------------
plt.figure(figsize=(12, 8))
for act in activations:
    for depth in depths:
        subset = [r for r in results if r["act"] == act and r["depth"] == depth]
        widths_list = [r["width"] for r in subset]
        test_mse = [r["mse_test"] for r in subset]
        label = f"{act}, {depth} hidden"
        plt.plot(widths_list, test_mse, marker='o', label=label)

plt.xlabel("Hidden Nodes per Layer")
plt.ylabel("Test MSE")
plt.title("Test MSE vs Network Width (by Activation & Depth)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "mse_vs_width.pdf"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved → {os.path.join(fig_dir, 'mse_vs_width.pdf')}")

# ------------------------------------------------------------------
# Plot: Learning curves (best model per activation)
# ------------------------------------------------------------------
best_models = {}
for act in activations:
    candidates = [r for r in results if r["act"] == act]
    best = min(candidates, key=lambda x: x["mse_test"])
    best_models[act] = best

plt.figure(figsize=(10, 6))
for act, r in best_models.items():
    plt.plot(r["history"], label=f"{act} (depth={r['depth']}, width={r['width']})")
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Learning Curves – Best Model per Activation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "learning_curves.pdf"), dpi=300)
plt.close()
print(f"Plot saved → {os.path.join(fig_dir, 'learning_curves.pdf')}")

# ------------------------------------------------------------------
# Overfitting heatmap
# ------------------------------------------------------------------
pivot = df.pivot_table(
    index="depth", columns="width", values="overfit", aggfunc='mean'
)
plt.figure(figsize=(8, 6))
im = plt.imshow(pivot.values, cmap="Reds", aspect='auto')
plt.colorbar(im, label="Overfitting (Test MSE - Train MSE)")
plt.xticks(range(len(widths)), widths)
plt.yticks(range(len(depths)), depths)
plt.xlabel("Width (nodes)")
plt.ylabel("Depth (layers)")
plt.title("Overfitting Heatmap (Avg over Activations)")
for i in range(len(depths)):
    for j in range(len(widths)):
        plt.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "overfitting_heatmap.pdf"), dpi=300)
plt.close()
print(f"Plot saved → {os.path.join(fig_dir, 'overfitting_heatmap.pdf')}")

print("\nPart D completed.")

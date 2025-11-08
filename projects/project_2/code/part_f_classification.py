# part_f_classification.py
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from models import NeuralNetwork, Adam
from utils import plot_learning_curve, plot_confusion_matrix


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Part F – MNIST classification")
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing data")
parser.add_argument("--mnist_file", type=str, required=True, help="MNIST .npz file")
parser.add_argument("--figures_dir", type=str, required=True, help="Directory to save figures")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--batch", type=int, default=64, help="Mini-batch size")
parser.add_argument("--eta", type=float, default=0.01, help="Learning rate")
parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization strength")
parser.add_argument("--seed", type=int, default=1993, help="Random seed")
args = parser.parse_args()


# ----------------------------------------------------------------------
# Load MNIST data from .npz
# ----------------------------------------------------------------------
mnist_path = os.path.join(args.data_dir, args.mnist_file)
data = np.load(mnist_path)

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

y_train = y_train.astype(np.int32)
y_test  = y_test.astype(np.int32)

print(f"MNIST loaded – train: {X_train.shape}, test: {X_test.shape}")
print(f"y_train dtype: {y_train.dtype}, sample: {y_train[:5]}")
print(f"y_test  dtype: {y_test.dtype}, sample: {y_test[:5]}")


# ----------------------------------------------------------------------
# Prepare output directories
# ----------------------------------------------------------------------
fig_dir = os.path.join(args.figures_dir, "part_f")
data_out_dir = os.path.join(args.data_dir, "part_f")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_out_dir, exist_ok=True)


# ----------------------------------------------------------------------
# Hyperparameter grid
# ----------------------------------------------------------------------
hidden_sizes = [100, 200, 300]
learning_rates = [1e-4, 5e-4, 1e-3]
l2_regs = [0.0, 1e-5, 1e-4]

results = []


# ----------------------------------------------------------------------
# Training function
# ----------------------------------------------------------------------
def train_nn(hidden: int, eta: float, l2: float):
    layers = [X_train.shape[1], hidden, 10]  # input → hidden → 10 classes
    activations = ["relu", "softmax"]
    nn = NeuralNetwork(
        layers=layers,
        activations=activations,
        cost="cross_entropy",
        seed=args.seed
    )

    history = nn.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        optimizer=Adam(eta=eta),
        l1=0.0,
        l2=l2,
        verbose=False
    )

    y_pred = nn.predict(X_test).ravel()
    acc = accuracy_score(y_test, y_pred)
    return nn, history, acc


# ----------------------------------------------------------------------
# Grid search
# ----------------------------------------------------------------------
print("\n=== NN grid search ===")
best_acc = 0.0
best_cfg = None
best_nn = None
best_history = None

for hidden in hidden_sizes:
    for eta in learning_rates:
        for l2 in l2_regs:
            print(f"  hidden={hidden:3d}  η={eta:.0e}  λ₂={l2:.0e} ...", end=" ")
            nn, hist, acc = train_nn(hidden, eta, l2)
            print(f"test-acc = {acc:.4f}")

            results.append({
                "hidden": hidden,
                "eta": eta,
                "l2": l2,
                "test_acc": acc,
                "history": hist
            })

            if acc > best_acc:
                best_acc = acc
                best_cfg = (hidden, eta, l2)
                best_nn = nn
                best_history = hist


print(f"\nBest NN → hidden={best_cfg[0]}, η={best_cfg[1]:.0e}, λ₂={best_cfg[2]:.0e}  →  test-acc = {best_acc:.4f}")


# ----------------------------------------------------------------------
# Save best NN results (figures)
# ----------------------------------------------------------------------
plot_learning_curve(
    best_history,
    title=f"Best NN – Test Accuracy {best_acc:.4f}",
    save_path=os.path.join(fig_dir, "best_nn_learning_curve.pdf")
)

y_pred_best = best_nn.predict(X_test).ravel()
plot_confusion_matrix(
    y_test, y_pred_best,
    save_path=os.path.join(fig_dir, "best_nn_confusion.pdf")
)


# ----------------------------------------------------------------------
# Logistic Regression baseline
# ----------------------------------------------------------------------
print("\n=== Logistic Regression baseline ===")
logreg = LogisticRegression(
    penalty='l2',
    C=1.0 / (args.l2 + 1e-8),
    solver='saga',
    multi_class='multinomial',
    max_iter=1000,
    random_state=args.seed,
    n_jobs=-1
)
logreg.fit(X_train, y_train.ravel())
y_pred_lr = logreg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"LogisticRegression test-acc = {acc_lr:.4f}")

plot_confusion_matrix(
    y_test, y_pred_lr,
    save_path=os.path.join(fig_dir, "logreg_confusion.pdf")
)


# ----------------------------------------------------------------------
# Save summary CSV (data/part_f)
# ----------------------------------------------------------------------
summary = {
    "model": ["Best NN", "LogisticRegression"],
    "hidden": [best_cfg[0], None],
    "eta":    [best_cfg[1], None],
    "l2":     [best_cfg[2], args.l2],
    "test_accuracy": [best_acc, acc_lr]
}
csv_path = os.path.join(data_out_dir, "summary.csv")
pd.DataFrame(summary).to_csv(csv_path, index=False)
print(f"Summary CSV → {csv_path}")


# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------
print("\nAll results saved in:")
print(f"  Figures → {fig_dir}")
print(f"  Data    → {data_out_dir}")
print("Part F completed.")

# part_a_analytical.py
"""
Part a – Analytical warm-up
---------------------------
* MSE + L1 + L2 (regression)
* Binary cross-entropy + L1 + L2
* Multiclass softmax cross-entropy
* Activation functions: Sigmoid, ReLU, Leaky-ReLU
* Symbolic derivatives via sympy
* Plots saved as PDF + CSV data
"""

import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from typing import Tuple

# ----------------------------------------------------------------------
# 1. Symbolic variables
# ----------------------------------------------------------------------
y_true, y_pred = sp.symbols('y_true y_pred')
λ1, λ2 = sp.symbols('λ_1 λ_2', real=True, nonnegative=True)
p = sp.symbols('p')           # predicted probability (binary)
k = sp.symbols('k')           # class index (multiclass)
y_k = sp.symbols('y_k')       # one-hot true label
p_k = sp.symbols('p_k')       # softmax output for class k

# ----------------------------------------------------------------------
# 2. Cost functions + gradients
# ----------------------------------------------------------------------
def print_section(title: str):
    print("\n" + "="*70)
    print(title)
    print("="*70)

# --- MSE + L1 + L2 ---
print_section("1. MSE + L1 + L2 (Regression)")

C_mse = (y_true - y_pred)**2 + λ1*sp.Abs(y_pred) + λ2*y_pred**2
g_mse = sp.diff(C_mse, y_pred)

print("Cost:")
sp.pprint(C_mse)
print("\nGradient w.r.t. y_pred:")
sp.pprint(g_mse)
print("\nLaTeX:")
print("Cost: ", sp.latex(C_mse))
print("Grad:  ", sp.latex(g_mse))

# --- Binary Cross-Entropy + L1 + L2 ---
print_section("2. Binary Cross-Entropy + L1 + L2")

C_bin = -(y_true*sp.log(p) + (1-y_true)*sp.log(1-p)) + λ1*sp.Abs(p) + λ2*p**2
g_bin = sp.diff(C_bin, p)

print("Cost:")
sp.pprint(C_bin)
print("\nGradient w.r.t. p:")
sp.pprint(g_bin)
print("\nLaTeX:")
print("Cost: ", sp.latex(C_bin))
print("Grad:  ", sp.latex(g_bin))

# --- Multiclass Softmax Cross-Entropy ---
print_section("3. Multiclass Softmax Cross-Entropy")

C_multi = -sp.Sum(y_k * sp.log(p_k), (k, 0, 9))  # sum over K=10 classes
g_multi = p_k - y_k

print("Cost (per sample):")
sp.pprint(C_multi)
print("\nGradient w.r.t. p_k (softmax output):")
sp.pprint(g_multi)
print("\nLaTeX:")
print("Cost: ", sp.latex(C_multi))
print("Grad:  p_k - y_k")

# ----------------------------------------------------------------------
# 3. Activation functions + derivatives
# ----------------------------------------------------------------------
x = sp.symbols('x')
α = sp.symbols('α', real=True, positive=True)

# Sigmoid
σ = 1 / (1 + sp.exp(-x))
dσ = sp.diff(σ, x)

# ReLU
relu = sp.Piecewise((0, x < 0), (x, True))
drelu = sp.diff(relu, x)

# Leaky ReLU
leaky = sp.Piecewise((α*x, x < 0), (x, True))
dleaky = sp.diff(leaky, x)

print_section("4. Activation Functions")

print("Sigmoid:       ", σ)
print("dσ/dx:         ", dσ)
print("\nReLU:        ", relu)
print("dReLU/dx:      ", drelu)
print("\nLeaky ReLU:  ", leaky)
print("dLeaky/dx:     ", dleaky)

# ----------------------------------------------------------------------
# 4. Numerical plotting + CSV export
# ----------------------------------------------------------------------
def plot_and_save(func_np, name: str, xmin: float = -3, xmax: float = 3, n: int = 400):
    xs = np.linspace(xmin, xmax, n)
    ys = func_np(xs)
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, label=name, linewidth=2)
    plt.title(name, fontsize=14, fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save PDF
    fig_path = f"figures/part_a/{name.lower().replace(' ', '_')}.pdf"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot → {fig_path}")

    # Save CSV
    csv_path = f"data/part_a/{name.lower().replace(' ', '_')}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    np.savetxt(csv_path, np.column_stack((xs, ys)), delimiter=",", header="x,y", comments="")
    print(f"CSV  → {csv_path}")

# Numerical versions
def sigmoid_np(z): return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
def relu_np(z):    return np.maximum(0, z)
def leaky_np(z, a=0.01): return np.where(z > 0, z, a*z)

plot_and_save(sigmoid_np, "Sigmoid")
plot_and_save(relu_np,    "ReLU")
plot_and_save(leaky_np,   "Leaky ReLU", xmin=-3, xmax=3)

# ----------------------------------------------------------------------
# 5. Final message
# ----------------------------------------------------------------------
print("""
All symbolic expressions printed above.
LaTeX code ready for report.
Plots saved in figures/part_a/*.pdf
CSV data in data/part_a/*.csv
""")

print_section("Part a - DONE")
# generated_data.py
import os
import matplotlib.pyplot as plt
from utils import generate_runge_data, save_data, split_scale

def plot_runge_data(X, y, data_dir, nsamples, noise):
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, color='blue', alpha=0.6, label='Runge data')
    plt.title(f'Runge Function Samples (n={nsamples}, noise={noise})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    plot_file = os.path.join(data_dir, f'runge_{nsamples}.png')
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Runge data plot saved → {plot_file}")

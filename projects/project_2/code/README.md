# Project 2 – FYS-STK3155/FYS4155  
**Neural Networks from Scratch: Regression and Classification with Feed-Forward Networks**

**Group members:** Christopher A. Trotter\
**Deadline:** November 10, 2025 (Midnight)  
**Course:** [FYS-STK3155 / FYS4155 – Data Analysis and Machine Learning](https://www.uio.no/studier/emner/matnat/fys/FYS-STK3155/index.html)


## Overview

This repository contains a **fully modular, from-scratch implementation** of:
- **Regression**: OLS, Ridge, Lasso, Gradient Descent (GD/SGD)
- **Feed-Forward Neural Networks (FFNN)** with back-propagation
- **Classification**: Multiclass MNIST using Softmax + Cross-Entropy
- **Hyperparameter tuning**, **regularization**, **optimizers**, and **validation**

All code is structured in **one file per sub-task (a–g)**, controlled by a **single driver script** (`project2.py`) — exactly like previous project `project1_regression_analysis.py`.


## Repository Structure
```
Project2/
│
├─ code/
│   └─ requirements.txt
│   └─ README.md
│   └─ src/
│   └─ models.py              ← All models (OLS, Neural Net, Logistic)
│   └─ utils.py               ← Data loading, scaling, plotting
│   └─ run_project2.py        ← Master driver (run all parts)
│   └─ part_a_analytical.py
│   └─ part_b_regression.py
│   └─ part_c_validation.py
│   └─ part_d_activation.py
│   └─ part_e_regularized.py
│   └─ part_f_classification.py
│   └─ part_g_evaluation.py
│   └─ data/           ← Auto-generated datasets (optional to push)
│   └─ figures/        ← Saved figures and csv for the report
│
├─ Report/
│   └─ Project2_report.pdf
│   └─ figures/            ← Saved figures and csv for the report
```

---

## Quick Start

### 1. **Clone the Repository**
```bash
git clone git@github.com:chrisatrotter/FYS-STK3155.git
cd projects/project2/code
```

### 2. **Create a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
If you get errors with scikit-learn:
```bash
pip install scikit-learn==1.5.0
```

### 4. **Run the Full Project**
```bash
python project2.py -n 1000 --epochs 400 --eta 0.001
```

This will generate data, train models, and store plots + results.


#### **Run a specific Part Only**
```bash
python run_project2.py --part a                     # Only Part A
python run_project2.py --part b                     # Only Part B
python run_project2.py --part e --l2 1e-4 --l1 1e-5 # Only Part E with reg.
```

This will generate data, train models, and store plots + results.

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|---------|------|---------|-------------|
| `-n`, `--nsamples` | int | 500 | Number of data samples |
| `--noise` | float | 0.05 | Noise level |
| `--hidden` | int list | [50, 100] | Hidden layer sizes |
| `--layers` | int | 2 | Number of hidden layers |
| `--act` | str | relu | Activation function |
| `--eta` | float | 0.001 | Learning rate |
| `--epochs` | int | 300 | Training epochs |
| `--batch` | int | 64 | Mini-batch size |
| `--l2` | float | 0.0 | L2 regularization |
| `--l1` | float | 0.0 | L1 regularization |
| `--seed` | int | 42 | Reproducibility seed |
| `--part` | str | (none) | Run only this part: `a`, `b`, `c`, `d`, `e`, `f`, or `g` |

---

## Requirements
```
numpy>=1.24
scikit-learn>=1.5
matplotlib>=3.8
seaborn>=0.13
pandas>=2.2
tqdm>=4.66
sympy>=1.12
```

---

## Submission Checklist

- ✅ Report/Project2_report.pdf
- Code folder with all `.py` files
- README.md (this file)
- requirements.txt
- figures/ folder included

---


- Nielsen, M. *Neural Networks and Deep Learning* — http://neuralnetworksanddeeplearning.com
- Goodfellow, I. et al. *Deep Learning* — https://www.deeplearningbook.org
- Raschka, S. *Machine Learning with PyTorch* — https://sebastianraschka.com

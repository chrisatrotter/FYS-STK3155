# Polynomial Regression & Regularization Project

This project implements various regression methods (Ordinary Least Squares, Ridge, Lasso) and optimization techniques (Gradient Descent, Stochastic Gradient Descent, Bootstrap, Cross-Validation) to analyze the Runge function for the FYS-STK3155/FYS4155 Project 1 assignment. It includes data generation, preprocessing, bias-variance analysis, and visualization of results such as MSE, R², and model coefficients.

---

## 🧩 Requirements

Install dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, create and activate a virtual environment before installation:

```bash
python3 -m venv venv
source venv/bin/activate    # on macOS/Linux
venv\Scripts\activate       # on Windows
pip install -r requirements.txt
```

Required packages:
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pandas`

---

## 🚀 How to Run

1. **Navigate to the Source Directory**:
   ```bash
   cd src
   ```

2. **Run the Main Script**:
   The main script `project1_regression_analysis.py` generates Runge function data and runs all parts of the assignment (a–h). Execute it with default or custom parameters:

   ```bash
   python3 project1_regression_analysis.py
   ```

   - **Arguments**:
     - `-n/--nsamples`: Number of samples (default: 100).
     - `-d/--degree`: Maximum polynomial degree (default: 15).
     - `--noise`: Noise level for Runge function (default: 0.05).
     - `--eta`: Learning rate for GD/SGD (default: 0.00001).
     - `--lambdas`: Regularization parameters for Ridge (default: [0.01, 0.1, 1.0, 10.0]).
     - `--lambda_val`: Regularization parameter for Lasso/CV (default: 0.01).
     - `--k`: Number of folds for cross-validation (default: 5).
     - `--n_bootstraps`: Number of bootstrap samples (default: 100).
     - `--noscale`: Disable feature scaling (default: False).
     - `--use_gd`: Use gradient descent for Lasso (default: False).
     - `--data_dir`: Directory for saved data (default: `project1/data`).

3. **Run Individual Scripts** (Optional):
   Each part of the project corresponds to a specific script. Run them individually if needed:
   ```bash
   python part_a_ols.py --nsamples 100 --degree 15 --noise 0.05 --data_dir ../data
   python part_b_ridge.py --nsamples 100 --degree 10 --noise 0.05 --lambdas 0.01 0.1 1.0 10.0 --data_dir ../data
   python part_c_gd.py --nsamples 100 --noise 0.05 --eta 0.00001 --data_dir ../data
   python part_d_gd_advanced.py --nsamples 100 --noise 0.05 --eta 0.00001 --data_dir ../data
   python part_e_lasso.py --nsamples 100 --degree 15 --noise 0.05 --lambda_val 0.01 --eta 0.00001 --data_dir ../data
   python part_f_sgd.py --nsamples 100 --noise 0.05 --eta 0.00001 --data_dir ../data
   python part_g_bootstrap.py --nsamples 100 --degree 15 --noise 0.05 --n_bootstraps 100 --data_dir ../data
   python part_h_cv.py --nsamples 100 --degree 15 --noise 0.05 --k 5 --lambda_val 0.01 --data_dir ../data
   ```

4. **Output Locations**:
   - **Data**: Generated Runge function data (train/test splits for n=100, 500, 1000) are saved in `project1/data/`.
   - **Plots**: Visualization of results (e.g., MSE, R², coefficients, bias-variance) are saved in `project1/figures/part_X/` for parts a, b, e, g, and h.
   - **Results**: CSV files with metrics (e.g., MSE, R², epochs, bias, variance) are saved in `project1/results/part_X/` for all parts.
   - Note: Parts c, d, and f produce CSV files but no plots, so their respective `figures/part_X/` directories remain empty.

---

## 📁 Project Structure

```
project1/
├── data/
│   ├── X_train_n100.csv     # Training features for n=100
│   ├── X_test_n100.csv      # Test features for n=100
│   ├── y_train_n100.csv     # Training targets for n=100
│   ├── y_test_n100.csv      # Test targets for n=100
│   ├── ...                  # Similar files for n=500, 1000
├── figures/
│   ├── data/                # Plots of Runge function data
│   ├── part_a/              # OLS plots (MSE, R², coefficients)
│   ├── part_b/              # Ridge plots (MSE, R², coefficients vs λ)
│   ├── part_e/              # Lasso plots (MSE, R², coefficients)
│   ├── part_g/              # Bootstrap plots (bias-variance tradeoff)
│   ├── part_h/              # Cross-validation plots (MSE vs degree)
├── results/
│   ├── part_a/              # OLS results (CSV files)
│   ├── part_b/              # Ridge results (CSV files)
│   ├── part_c/              # GD results (CSV files)
│   ├── part_d/              # Advanced GD results (CSV files)
│   ├── part_e/              # Lasso results (CSV files)
│   ├── part_f/              # SGD results (CSV files)
│   ├── part_g/              # Bootstrap results (CSV files)
│   ├── part_h/              # Cross-validation results (CSV files)
├── src/
│   ├── project1_regression_analysis.py  # Main script to run all parts
│   ├── utils.py                        # Data generation and utility functions
│   ├── regression_methods.py           # Regression model implementations
│   ├── part_a_ols.py                   # OLS regression
│   ├── part_b_ridge.py                 # Ridge regression
│   ├── part_c_gd.py                    # Gradient Descent for OLS and Ridge
│   ├── part_d_gd_advanced.py           # Advanced GD methods (momentum, Adagrad, RMSProp, Adam)
│   ├── part_e_lasso.py                 # Lasso regression
│   ├── part_f_sgd.py                   # Stochastic Gradient Descent
│   ├── part_g_bootstrap.py             # Bootstrap for bias-variance analysis
│   ├── part_h_cv.py                    # K-fold cross-validation
├── requirements.txt
├── README.md
```

---

## 📋 Project Parts

The project implements the following parts for the Runge function \( f(x) = \frac{1}{1 + 25x^2} \) with \( x \in [-1, 1] \) and added Gaussian noise:

- **Part a (OLS)**: Implements Ordinary Least Squares regression using the pseudoinverse, analyzing MSE, R², and coefficient progression for polynomial degrees 1–15 across sample sizes (n=100, 500, 1000).
- **Part b (Ridge)**: Implements Ridge regression with regularization parameters λ=[0.01, 0.1, 1.0, 10.0], evaluating MSE, R², and coefficients for polynomial degrees up to 10.
- **Part c (GD)**: Implements vanilla Gradient Descent for OLS and Ridge (λ=0.01), reporting test MSE and epochs for a fixed degree (5) and sample sizes (n=100, 500, 1000).
- **Part d (Advanced GD)**: Implements advanced GD methods (vanilla, momentum, Adagrad, RMSProp, Adam) for Ridge regression, reporting test MSE and epochs for a fixed degree (5).
- **Part e (Lasso)**: Implements Lasso regression using scikit-learn’s coordinate descent or custom GD, evaluating MSE, R², and coefficients for polynomial degrees 1–15.
- **Part f (SGD)**: Implements Stochastic Gradient Descent with advanced methods (vanilla, momentum, Adagrad, RMSProp, Adam) for OLS, Ridge, and Lasso, reporting test MSE, R², and epochs for a fixed degree (5).
- **Part g (Bootstrap)**: Performs bootstrap resampling (100 samples) to analyze bias-variance tradeoff for OLS across polynomial degrees 1–15.
- **Part h (Cross-Validation)**: Implements k-fold cross-validation (k=5) for OLS, Ridge, and Lasso, comparing MSE across polynomial degrees 1–15.

---

## 📊 Output Files

Each script generates outputs in its respective `results/part_X/` and `figures/part_X/` directories:
- **CSV Files**: Contain metrics such as MSE, R², epochs, bias, and variance for sample sizes 100, 500, and 1000.
  - Example: `results/part_a/runge_ols_n100.csv` includes Degree, MSE_Train, MSE_Test, R2_Train, R2_Test.
  - Example: `results/part_g/runge_bootstrap_n500.csv` includes Degree, MSE, Bias2, Variance.
- **Plots**: Visualize metrics for the Runge function:
  - Part a: MSE, R², and coefficient progression vs degree (`figures/part_a/runge_ols_combined_n100.png`).
  - Part b: MSE, R², and coefficients vs degree for different λ (`figures/part_b/runge_ridge_combined_n100.png`).
  - Part e: MSE, R², and coefficients vs degree for Lasso (`figures/part_e/runge_lasso_combined_n100_gdFalse.png`).
  - Part g: Bias, variance, and MSE vs degree (`figures/part_g/runge_bootstrap_n100.png`).
  - Part h: Cross-validation MSE vs degree for OLS, Ridge, Lasso (`figures/part_h/runge_cv_n100.png`).
- Note: Parts c, d, and f generate CSV files but no plots.

---

## 📜 License

This project is open source and free to use for educational or research purposes.

---

## 📝 Notes for Submission

- **Report**: Include results from CSV files and plots in your report. Discuss trends in MSE, R², coefficients, bias-variance tradeoff, and the impact of scaling (using `--noscale` to compare). For part g, include the theoretical derivation of the bias-variance tradeoff in the theory section.
- **Data Scaling**: The code uses `StandardScaler` for polynomial features to ensure numerical stability. Discuss in your report why scaling is critical (e.g., prevents large coefficients in high-degree polynomials, improves GD convergence) and how it’s implemented (mean=0, std=1).
- **GitHub**: Upload all files to a GitHub repository and include the link in your report.
- **Running Tips**: Ensure the `data/` directory is created automatically when running `project1_regression_analysis.py`. Use the `--noscale` flag to experiment with unscaled features and discuss its impact.

For issues or questions, refer to the FYS-STK3155/FYS4155 project guidelines or contact the course instructor.
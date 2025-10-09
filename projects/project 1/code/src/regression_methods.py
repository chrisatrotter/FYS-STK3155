import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

class RegressionModel:
    @staticmethod
    def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ordinary Least Squares regression."""
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    @staticmethod
    def ridge_fit(X: np.ndarray, y: np.ndarray, lambda_val: float) -> np.ndarray:
        """Ridge regression."""
        XTX = X.T @ X
        return np.linalg.pinv(XTX + lambda_val * np.identity(XTX.shape[0])) @ X.T @ y

    @staticmethod
    def lasso_fit(X: np.ndarray, y: np.ndarray, lambda_val: float) -> np.ndarray:
        """Lasso regression."""
        clf = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=int(1e5), tol=1e-1)
        clf.fit(X, y)
        return clf.coef_

    @staticmethod
    def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Predict using model coefficients."""
        return X @ theta

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """Compute MSE and R2 scores."""
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

    @staticmethod
    def gd_fit(X: np.ndarray, y: np.ndarray, eta: float = 0.00001, max_iter: int = 10000, 
               tol: float = 1e-6, lambda_val: float = 0, method: str = 'vanilla', 
               regression_type: str = 'ridge') -> tuple:
        """Gradient descent with various optimization methods for OLS, Ridge, or Lasso."""
        theta = np.zeros(X.shape[1])
        v, m, s = np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta)
        t, beta1, beta2, gamma, epsilon = 1, 0.9, 0.999, 0.9, 1e-8

        for i in range(max_iter):
            grad = -2/X.shape[0] * X.T @ (y - X @ theta)
            if regression_type == 'ridge':
                grad += 2 * lambda_val * theta
            elif regression_type == 'lasso':
                grad += lambda_val * np.sign(theta)
            grad = np.clip(grad, -1e2, 1e2)
            
            if method == 'vanilla':
                theta_new = theta - eta * grad
            elif method == 'momentum':
                v = gamma * v - eta * grad
                theta_new = theta + v
            elif method == 'adagrad':
                s += grad**2
                theta_new = theta - eta * grad / (np.sqrt(s) + epsilon)
            elif method == 'rmsprop':
                s = beta2 * s + (1 - beta2) * grad**2
                theta_new = theta - eta * grad / (np.sqrt(s) + epsilon)
            elif method == 'adam':
                m = beta1 * m + (1 - beta1) * grad
                s = beta2 * s + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**t)
                s_hat = s / (1 - beta2**t)
                theta_new = theta - eta * m_hat / (np.sqrt(s_hat) + epsilon)
                t += 1
            else:
                raise ValueError(f"Unknown method: {method}")

            if np.linalg.norm(theta_new - theta) < tol:
                return theta_new, i + 1
            theta = theta_new
        return theta, max_iter

    @staticmethod
    def sgd_fit(X: np.ndarray, y: np.ndarray, eta: float = 0.00001, max_iter: int = 1000, 
                batch_size: int = 32, method: str = 'vanilla', lambda_val: float = 1e-4, 
                regression_type: str = 'ridge') -> tuple:
        """Stochastic gradient descent with various optimization methods for OLS, Ridge, or Lasso."""
        theta = np.zeros(X.shape[1])
        v, m, s = np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta)
        t, beta1, beta2, gamma, epsilon = 1, 0.9, 0.999, 0.9, 1e-8
        n_batches = max(1, X.shape[0] // batch_size)

        for i in range(max_iter):
            indices = np.random.permutation(X.shape[0])
            X_shuffled, y_shuffled = X[indices], y[indices]
            for j in range(n_batches):
                start = j * batch_size
                end = min(start + batch_size, X.shape[0])
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]
                grad = -2/X_batch.shape[0] * X_batch.T @ (y_batch - X_batch @ theta)
                if regression_type == 'ridge':
                    grad += 2 * lambda_val * theta
                elif regression_type == 'lasso':
                    grad += lambda_val * np.sign(theta)
                grad = np.clip(grad, -1e2, 1e2)
                
                if method == 'vanilla':
                    theta = theta - eta * grad
                elif method == 'momentum':
                    v = gamma * v - eta * grad
                    theta = theta + v
                elif method == 'adagrad':
                    s += grad**2
                    theta = theta - eta * grad / (np.sqrt(s) + epsilon)
                elif method == 'rmsprop':
                    s = beta2 * s + (1 - beta2) * grad**2
                    theta = theta - eta * grad / (np.sqrt(s) + epsilon)
                elif method == 'adam':
                    m = beta1 * m + (1 - beta1) * grad
                    s = beta2 * s + (1 - beta2) * grad**2
                    m_hat = m / (1 - beta1**t)
                    s_hat = s / (1 - beta2**t)
                    theta = theta - eta * m_hat / (np.sqrt(s_hat) + epsilon)
                    t += 1
        return theta, max_iter

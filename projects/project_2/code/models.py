# models.py
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import Lasso, LogisticRegression
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===================================================================
# 1. REGRESSION MODELS (OLS, Ridge, Lasso, GD, SGD)
# ===================================================================
class RegressionModel:
    @staticmethod
    def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    @staticmethod
    def ridge_fit(X: np.ndarray, y: np.ndarray, lambda_val: float) -> np.ndarray:
        I = np.eye(X.shape[1])
        return np.linalg.pinv(X.T @ X + lambda_val * I) @ X.T @ y

    @staticmethod
    def lasso_fit(X: np.ndarray, y: np.ndarray, lambda_val: float) -> np.ndarray:
        clf = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=100_000, tol=1e-4)
        clf.fit(X, y.ravel())
        return clf.coef_.reshape(-1, 1)

    @staticmethod
    def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return X @ theta

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

    @staticmethod
    def gd_fit(X: np.ndarray, y: np.ndarray, eta: float = 1e-3, max_iter: int = 10000,
               tol: float = 1e-6, lambda_val: float = 0.0, method: str = 'vanilla',
               regression_type: str = 'ridge') -> Tuple[np.ndarray, int]:
        theta = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        m, v, s = np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta)
        t = 1
        beta1, beta2, gamma, eps = 0.9, 0.999, 0.9, 1e-8

        for i in range(max_iter):
            grad = (2 / len(y)) * X.T @ (X @ theta - y)
            if regression_type == 'ridge':
                grad += 2 * lambda_val * theta
            elif regression_type == 'lasso':
                grad += lambda_val * np.sign(theta)

            if method == 'vanilla':
                theta_new = theta - eta * grad
            elif method == 'momentum':
                v = gamma * v + eta * grad
                theta_new = theta - v
            elif method == 'adagrad':
                s += grad**2
                theta_new = theta - eta * grad / (np.sqrt(s) + eps)
            elif method == 'rmsprop':
                s = beta2 * s + (1 - beta2) * grad**2
                theta_new = theta - eta * grad / (np.sqrt(s) + eps)
            elif method == 'adam':
                m = beta1 * m + (1 - beta1) * grad
                s = beta2 * s + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**t)
                s_hat = s / (1 - beta2**t)
                theta_new = theta - eta * m_hat / (np.sqrt(s_hat) + eps)
                t += 1
            else:
                raise ValueError(f"Unknown method: {method}")

            if np.linalg.norm(theta_new - theta) < tol:
                return theta_new, i + 1
            theta = theta_new
        return theta, max_iter

    @staticmethod
    def sgd_fit(X: np.ndarray, y: np.ndarray, eta: float = 1e-3, max_iter: int = 1000,
                batch_size: int = 32, method: str = 'vanilla', lambda_val: float = 0.0,
                regression_type: str = 'ridge') -> Tuple[np.ndarray, int]:
        theta = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        m, v, s = np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta)
        t = 1
        beta1, beta2, gamma, eps = 0.9, 0.999, 0.9, 1e-8
        n = X.shape[0]

        for _ in range(max_iter):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            for i in range(0, n, batch_size):
                Xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]
                grad = (2 / len(yb)) * Xb.T @ (Xb @ theta - yb)
                if regression_type == 'ridge':
                    grad += 2 * lambda_val * theta
                elif regression_type == 'lasso':
                    grad += lambda_val * np.sign(theta)

                if method == 'vanilla':
                    theta -= eta * grad
                elif method == 'momentum':
                    v = gamma * v + eta * grad
                    theta -= v
                elif method == 'adagrad':
                    s += grad**2
                    theta -= eta * grad / (np.sqrt(s) + eps)
                elif method == 'rmsprop':
                    s = beta2 * s + (1 - beta2) * grad**2
                    theta -= eta * grad / (np.sqrt(s) + eps)
                elif method == 'adam':
                    m = beta1 * m + (1 - beta1) * grad
                    s = beta2 * s + (1 - beta2) * grad**2
                    m_hat = m / (1 - beta1**t)
                    s_hat = s / (1 - beta2**t)
                    theta -= eta * m_hat / (np.sqrt(s_hat) + eps)
                    t += 1
        return theta, max_iter


# ===================================================================
# 2. CLASSIFICATION: Logistic Regression (multi-class)
# ===================================================================
class ClassificationModel:
    @staticmethod
    def logistic_fit(X: np.ndarray, y: np.ndarray, lambda_val: float = 0.0) -> np.ndarray:
        clf = LogisticRegression(penalty='l2', C=1/(lambda_val + 1e-8),
                                 multi_class='multinomial', max_iter=1000,
                                 solver='saga', tol=1e-4)
        clf.fit(X, y.ravel())
        return clf.coef_.T

    @staticmethod
    def softmax_predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        z = X @ theta
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def predict_class(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        prob = ClassificationModel.softmax_predict(X, theta)
        return np.argmax(prob, axis=1)

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def cross_entropy_loss(y_true_onehot: np.ndarray, prob: np.ndarray) -> float:
        eps = 1e-15
        return -np.mean(np.sum(y_true_onehot * np.log(prob + eps), axis=1))


# ===================================================================
# 3. NEURAL NETWORK (FFNN) – My own implementation
# ===================================================================
class NeuralNetwork:
    def __init__(self, layers: List[int], activations: List[str],
                 cost: str = 'mse', seed: int = 42):
        self.layers = layers
        self.L = len(layers) - 1
        self.cost = cost
        self.act_funcs, self.act_derivs = self._get_activations(activations)
        self.rng = np.random.default_rng(seed)
        self._initialize_weights()

    def _get_activations(self, names: List[str]):
        funcs = {
            'sigmoid': (self._sigmoid, self._dsigmoid),
            'relu': (self._relu, self._drelu),
            'leaky_relu': (self._leaky_relu, self._dleaky_relu),
            'linear': (lambda x: x, lambda x: np.ones_like(x)),
            'softmax': (self._softmax, lambda x: np.ones_like(x))
        }
        return ([funcs[n][0] for n in names], [funcs[n][1] for n in names])

    def _initialize_weights(self):
        self.W, self.b = [], []
        for i in range(self.L):
            fan_in, fan_out = self.layers[i], self.layers[i+1]
            if self.act_funcs[i].__name__ in ('_relu', '_leaky_relu'):
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.W.append(self.rng.normal(0, scale, (fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))

    @staticmethod
    def _sigmoid(z): 
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    @staticmethod
    def _dsigmoid(a): 
        return a * (1 - a)
    @staticmethod
    def _relu(z): 
        return np.maximum(0, z)
    @staticmethod
    def _drelu(a): 
        return (a > 0).astype(float)
    @staticmethod
    def _leaky_relu(z, alpha=0.01): 
        return np.where(z > 0, z, alpha * z)
    @staticmethod
    def _dleaky_relu(a, alpha=0.01): 
        return np.where(a > 0, 1.0, alpha)
    @staticmethod
    def _softmax(z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache = {'a0': X.copy()}
        a = X
        for l in range(self.L):
            z = a @ self.W[l] + self.b[l]
            a = self.act_funcs[l](z)
            self.cache[f'z{l+1}'] = z
            self.cache[f'a{l+1}'] = a
        return a

    def backward(self, X: np.ndarray, y_onehot: np.ndarray, y_pred: np.ndarray,
                 l1: float = 0.0, l2: float = 0.0) -> Dict[str, List[np.ndarray]]:
        m = X.shape[0]
        grads = {'dW': [], 'db': []}

        if self.cost == 'mse':
            delta = 2 * (y_pred - y_onehot) * self.act_derivs[-1](y_pred) / m
        else:
            delta = (y_pred - y_onehot) / m

        for l in reversed(range(self.L)):
            a_prev = self.cache[f'a{l}']
            dW = a_prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            if l2 > 0:
                dW += l2 * self.W[l]
            if l1 > 0:
                dW += l1 * np.sign(self.W[l])

            grads['dW'].insert(0, dW)
            grads['db'].insert(0, db)

            if l > 0:
                delta = delta @ self.W[l].T * self.act_derivs[l-1](self.cache[f'a{l}'])
        return grads

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
            optimizer: Any, l1: float = 0.0, l2: float = 0.0, verbose: bool = True):
        history = []
        m = X.shape[0]
        for epoch in range(1, epochs + 1):
            idx = self.rng.permutation(m)
            X_shuf, y_shuf = X[idx], y[idx]
            for i in range(0, m, batch_size):
                Xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]

                yb = yb.astype(np.int32)

                y_pred = self.forward(Xb)
                if self.cost == 'cross_entropy':
                    yb_onehot = np.eye(self.layers[-1])[yb.ravel()]
                else:
                    yb_onehot = yb
                grads = self.backward(Xb, yb_onehot, y_pred, l1, l2)
                optimizer.update(self.W, self.b, grads)

            y_full = self.forward(X)
            if self.cost == 'mse':
                metric = mean_squared_error(y, y_full)
            else:
                pred = np.argmax(y_full, axis=1)
                metric = accuracy_score(y.ravel(), pred)
            history.append(metric)
            if verbose and epoch % max(1, epochs//10) == 0:
                print(f"Epoch {epoch:4d} | metric: {metric:.5f}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.forward(X)
        return y_pred if self.cost == 'mse' else np.argmax(y_pred, axis=1).reshape(-1, 1)


# ===================================================================
# 4. OPTIMIZER CLASSES
# ===================================================================
class Optimizer:
    def update(self, W: List[np.ndarray], b: List[np.ndarray], grads: Dict): pass

class SGD(Optimizer):
    def __init__(self, eta: float = 0.01): self.eta = eta
    def update(self, W, b, grads):
        for l in range(len(W)):
            W[l] -= self.eta * grads['dW'][l]
            b[l] -= self.eta * grads['db'][l]

class RMSprop(Optimizer):
    def __init__(self, eta: float = 0.001, rho: float = 0.9, eps: float = 1e-8):
        self.eta, self.rho, self.eps = eta, rho, eps
        self.sW = self.sb = None
    def update(self, W, b, grads):
        if self.sW is None:
            self.sW = [np.zeros_like(w) for w in W]
            self.sb = [np.zeros_like(bb) for bb in b]
        for l in range(len(W)):
            self.sW[l] = self.rho * self.sW[l] + (1 - self.rho) * grads['dW'][l]**2
            self.sb[l] = self.rho * self.sb[l] + (1 - self.rho) * grads['db'][l]**2
            W[l] -= self.eta * grads['dW'][l] / (np.sqrt(self.sW[l]) + self.eps)
            b[l] -= self.eta * grads['db'][l] / (np.sqrt(self.sb[l]) + self.eps)

class Adam(Optimizer):
    def __init__(self, eta: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.eta, self.b1, self.b2, self.eps = eta, beta1, beta2, eps
        self.mW = self.vW = self.mb = self.vb = None
        self.t = 0
    def update(self, W, b, grads):
        self.t += 1
        if self.mW is None:
            self.mW = [np.zeros_like(w) for w in W]; self.vW = [np.zeros_like(w) for w in W]
            self.mb = [np.zeros_like(bb) for bb in b]; self.vb = [np.zeros_like(bb) for bb in b]
        for l in range(len(W)):
            self.mW[l] = self.b1 * self.mW[l] + (1 - self.b1) * grads['dW'][l]
            self.vW[l] = self.b2 * self.vW[l] + (1 - self.b2) * grads['dW'][l]**2
            self.mb[l] = self.b1 * self.mb[l] + (1 - self.b1) * grads['db'][l]
            self.vb[l] = self.b2 * self.vb[l] + (1 - self.b2) * grads['db'][l]**2
            m_hat = self.mW[l] / (1 - self.b1**self.t)
            v_hat = self.vW[l] / (1 - self.b2**self.t)
            mb_hat = self.mb[l] / (1 - self.b1**self.t)
            vb_hat = self.vb[l] / (1 - self.b2**self.t)
            W[l] -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps)
            b[l] -= self.eta * mb_hat / (np.sqrt(vb_hat) + self.eps)

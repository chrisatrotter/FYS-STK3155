# tasks/part_c_training.py
"""
Part C – Model Training & Implementation
FYS-STK3155 / FYS4155 – Project 3

Trains all required models for both datasets.
Run via: python code/project3.py --dataset power --part c
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
from pathlib import Path

# Import our consistent divider + utilities
from utils import ensure_dir, breakpoint
from models import NeuralNetwork, Adam, RegressionModel
from datasets.power import PowerDataset
from datasets.trade import TradeDataset


# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Part C – Model Training")
parser.add_argument("--dataset", type=str, choices=["power", "trade"], required=True)
parser.add_argument("--seed", type=int, default=1993)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--part", type=str, help="Internal flag")
args, _ = parser.parse_known_args()

# Set seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Output directories
model_dir = Path("models") / args.dataset
pred_dir  = Path("data/predictions") / args.dataset
ensure_dir(model_dir)
ensure_dir(pred_dir)
ensure_dir("figures/part_c")

breakpoint()
print(f" PART C – TRAINING MODELS: {args.dataset.upper()} DATASET")
breakpoint()


# ======================================================================
# POWER DATASET – REGRESSION
# ======================================================================
if args.dataset == "power":
    print("Loading PowerDataset...")
    data = PowerDataset(seed=args.seed)

    X_train, X_val, X_test = data.X_train, data.X_val, data.X_test
    y_train, y_val, y_test = data.y_train, data.y_val, data.y_test

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}\n")

    # ——— Ridge Regression (from scratch) ———
    print("Training Ridge Regression (λ=1.0)...")
    theta_ridge = RegressionModel.ridge_fit(X_train, y_train, lambda_val=1.0)
    pred_ridge = RegressionModel.predict(X_test, theta_ridge)
    np.save(pred_dir / "pred_ridge.npy", pred_ridge.ravel())
    print("  → Ridge saved")

    # ——— Lasso (sklearn) ———
    print("Training Lasso Regression (α=0.01)...")
    theta_lasso = RegressionModel.lasso_fit(X_train, y_train, lambda_val=0.01)
    pred_lasso = RegressionModel.predict(X_test, theta_lasso)
    np.save(pred_dir / "pred_lasso.npy", pred_lasso.ravel())
    print("  → Lasso saved")

    # ——— XGBoost Regressor ———
    print("Training XGBoost Regressor (early stopping)...")
    dtrain = xgb.DMatrix(X_train, label=y_train.ravel())
    dval   = xgb.DMatrix(X_val,   label=y_val.ravel())
    dtest  = xgb.DMatrix(X_test)

    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': args.seed,
        'eval_metric': 'rmse',
        'verbosity': 0
    }
    xgb_reg = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    xgb_reg.save_model(model_dir / "xgboost_reg.json")
    np.save(pred_dir / "pred_xgboost.npy", xgb_reg.predict(dtest))
    print(f"  → XGBoost saved (best iteration: {xgb_reg.best_iteration})")

    # ——— From-scratch FFNN ———
    print("Training From-Scratch FFNN (4-layer, Adam)...")
    ffnn = NeuralNetwork(
        layers=[X_train.shape[1], 128, 64, 32, 1],
        activations=["relu", "relu", "relu", "linear"],
        cost="mse",
        seed=args.seed
    )
    ffnn.fit(
        X=X_train.astype(np.float64),
        y=y_train.astype(np.float64),
        epochs=args.epochs,
        batch_size=args.batch,
        optimizer=Adam(eta=0.001),
        X_val=X_val.astype(np.float64),
        y_val=y_val.astype(np.float64),
        verbose=True
    )
    ffnn.save(model_dir / "ffnn_regression.pkl")
    np.save(pred_dir / "pred_ffnn.npy", ffnn.predict(X_test.astype(np.float64)).ravel())
    print("  → FFNN saved")

    # ——— PyTorch LSTM ———
    print("Training PyTorch LSTM (seq_len=24)...")
    class LSTMRegressor(nn.Module):
        def __init__(self, input_size=5, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1])
            return self.fc(out)

    seq_len = 24
    X_seq, y_seq = data.create_sequences(X_train, y_train.ravel(), seq_len=seq_len)
    X_val_seq, y_val_seq = data.create_sequences(X_val, y_val.ravel(), seq_len=seq_len)
    X_test_seq, _ = data.create_sequences(X_test, y_test.ravel(), seq_len=seq_len)

    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.float32)
    X_val_seq = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_seq = torch.tensor(y_val_seq, dtype=torch.float32)
    X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32)

    model = LSTMRegressor(input_size=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience = 15
    wait = 0

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_seq)
        loss = criterion(outputs.squeeze(), y_seq)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_seq)
                val_loss = criterion(val_out.squeeze(), y_val_seq).item()
                print(f"    Epoch {epoch:3d} → Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
                torch.save(model.state_dict(), model_dir / "lstm_reg.pth")
            else:
                wait += 1
                if wait >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

    model.load_state_dict(torch.load(model_dir / "lstm_reg.pth"))
    model.eval()
    with torch.no_grad():
        pred_lstm = model(X_test_seq).numpy().ravel()
        np.save(pred_dir / "pred_lstm.npy", pred_lstm)

    print("  → LSTM saved (with early stopping)")

    np.save(pred_dir / "y_test.npy", y_test.ravel())


# ======================================================================
# TRADE DATASET – CLASSIFICATION
# ======================================================================
else:
    print("Loading TradeDataset...")
    data = TradeDataset(seed=args.seed)

    X_train, X_val, X_test = data.X_train, data.X_val, data.X_test
    y_train, y_val, y_test = data.y_train.ravel(), data.y_val.ravel(), data.y_test.ravel()

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    print(f"Positive class: {pos_ratio:.3%} → scale_pos_weight = {scale_pos_weight:.1f}\n")

    # ——— Logistic Regression ———
    print("Training Logistic Regression...")
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=args.seed)
    logreg.fit(X_train, y_train)
    joblib.dump(logreg, model_dir / "logreg.pkl")
    np.save(pred_dir / "pred_logreg.npy", logreg.predict(X_test))
    print("  → Logistic Regression saved")

    # ——— XGBoost Classifier ———
    print("Training XGBoost Classifier...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test)

    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'seed': args.seed,
        'eval_metric': 'auc',
        'verbosity': 0
    }
    xgb_cls = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    xgb_cls.save_model(model_dir / "xgboost_cls.json")
    pred_proba = xgb_cls.predict(dtest)
    pred_hard = (pred_proba >= 0.5).astype(int)
    np.save(pred_dir / "pred_xgboost.npy", pred_hard)
    np.save(pred_dir / "pred_xgboost_proba.npy", pred_proba)
    print(f"  → XGBoost saved (best iteration: {xgb_cls.best_iteration})")

    # ——— From-scratch FFNN Classifier ———
    print("Training From-Scratch FFNN Classifier...")
    ffnn_cls = NeuralNetwork(
        layers=[X_train.shape[1], 128, 64, 32, 1],
        activations=["relu", "relu", "relu", "sigmoid"],
        cost="bce",
        seed=args.seed
    )
    ffnn_cls.fit(
        X=X_train.astype(np.float64),
        y=y_train.reshape(-1, 1).astype(np.float64),
        epochs=args.epochs,
        batch_size=args.batch,
        optimizer=Adam(eta=0.005),
        X_val=X_val.astype(np.float64),
        y_val=y_val.reshape(-1, 1).astype(np.float64),
        verbose=True
    )
    ffnn_cls.save(model_dir / "ffnn_classification.pkl")
    pred_proba_ffnn = ffnn_cls.forward(X_test.astype(np.float64)).ravel()
    pred_cls = (pred_proba_ffnn >= 0.5).astype(int)
    np.save(pred_dir / "pred_ffnn.npy", pred_cls)
    np.save(pred_dir / "pred_ffnn_proba.npy", pred_proba_ffnn)
    print("  → FFNN Classifier saved")

    np.save(pred_dir / "y_test.npy", y_test)


# ======================================================================
breakpoint()
print(f" PART C ({args.dataset.upper()}) – ALL MODELS TRAINED & SAVED!")
print(f" • Models → {model_dir}/")
print(f" • Predictions → {pred_dir}/")
print(" • Ready for Part D results!")
breakpoint()

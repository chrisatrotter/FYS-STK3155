# tasks/part_d_results.py
"""
Part D – Results & Visualisation
FYS-STK3155 / FYS4155 – Project 3
Full comparison across ALL models + trade classification
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, roc_curve, auc
)
import xgboost as xgb
import shap

from utils import ensure_dir, save_plot, breakpoint
from datasets.power import PowerDataset
from datasets.trade import TradeDataset


# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Part D – Results & Visualisation")
parser.add_argument("--dataset", type=str, choices=["power", "trade"], required=True)
parser.add_argument("--seed", type=int, default=1993)
parser.add_argument("--part", type=str, help="Internal flag")
args, _ = parser.parse_known_args()

pred_dir = Path("data/predictions") / args.dataset
model_dir = Path("models") / args.dataset
fig_dir = Path("figures/part_d")
ensure_dir(fig_dir)

sns.set(style="whitegrid", font_scale=1.4, palette="deep", rc={
    "grid.alpha": 0.3, "axes.titlesize": 18, "axes.labelsize": 14
})

breakpoint()
print(f" PART D – RESULTS & VISUALISATION: {args.dataset.upper()} DATASET")
breakpoint()


# ======================================================================
# POWER DATASET – FULL MODEL COMPARISON (15 plots)
# ======================================================================
if args.dataset == "power":
    print("Loading PowerDataset and all predictions...")
    data = PowerDataset(seed=args.seed)
    y_test = data.y_test.ravel()
    timestamps = data.timestamps_test

    # Load predictions
    pred_ridge = np.load(pred_dir / "pred_ridge.npy")
    pred_lasso = np.load(pred_dir / "pred_lasso.npy")
    pred_xgb   = np.load(pred_dir / "pred_xgboost.npy")
    pred_ffnn  = np.load(pred_dir / "pred_ffnn.npy")
    pred_lstm  = np.load(pred_dir / "pred_lstm.npy") if (pred_dir / "pred_lstm.npy").exists() else None

    # Align LSTM (drops first 24 hours)
    if pred_lstm is not None and len(pred_lstm) != len(y_test):
        aligned = np.full_like(y_test, np.nan)
        aligned[24:24 + len(pred_lstm)] = pred_lstm
        pred_lstm = aligned

    # Model registry
    model_configs = [
        ("Ridge",   pred_ridge, "#1f77b4"),
        ("Lasso",   pred_lasso, "#9467bd"),
        ("XGBoost", pred_xgb,   "#ff7f0e"),
        ("FFNN",    pred_ffnn,  "#2ca02c"),
    ]
    if pred_lstm is not None:
        model_configs.append(("LSTM", pred_lstm, "#d62728"))

    # Safe metrics
    def rmse(y_true, y_pred):
        mask = ~np.isnan(y_pred)
        return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])) if mask.any() else np.inf

    def r2(y_true, y_pred):
        mask = ~np.isnan(y_pred)
        return r2_score(y_true[mask], y_pred[mask]) if mask.any() else -np.inf

    # Performance table
    results = pd.DataFrame({
        'Model': [name for name, _, _ in model_configs],
        'RMSE':  [rmse(y_test, pred) for _, pred, _ in model_configs],
        'R²':    [r2(y_test, pred)   for _, pred, _ in model_configs]
    }).round(5)

    print("REGRESSION PERFORMANCE – Electricity Consumption Forecasting")
    print(results.to_string(index=False))

    # Bar plots
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    results.plot(x='Model', y='RMSE', kind='bar', ax=ax[0], legend=False, color="#1f77b4", edgecolor="black")
    ax[0].set_title("RMSE Comparison", fontweight="bold")
    ax[0].set_ylabel("RMSE (kW)")

    results.plot(x='Model', y='R²', kind='bar', ax=ax[1], legend=False, color="#2ca02c", edgecolor="black")
    ax[1].set_title("R² Score Comparison", fontweight="bold")
    ax[1].set_ylabel("R²")
    ax[1].set_ylim(-2, 1)
    plt.tight_layout()
    save_plot(fig, "01_regression_metrics_power", "d")

    # Find most typical week
    hours_per_week = 168
    weekly_means = [y_test[i*hours_per_week:(i+1)*hours_per_week].mean()
                    for i in range(len(y_test)//hours_per_week)]
    best_week_idx = np.argmin(np.abs(np.array(weekly_means) - y_test.mean()))
    week_start = best_week_idx * hours_per_week
    week_end = week_start + hours_per_week
    week_timestamps = timestamps[week_start:week_end]
    week_true = y_test[week_start:week_end]
    week_title = f"Typical Week ({week_timestamps[0]:%b %d} – {week_timestamps[-1]:%b %d, %Y})"

    print(f"   → Using typical test week: {week_title}")

    # === FULL, WEEKLY, DAILY FOR ALL MODELS ===
    print("\nGenerating 15 comparison plots (full, weekly, daily) for all models...")
    for name, pred, color in model_configs:
        valid = ~np.isnan(pred)

        # 1. Full period
        fig, ax = plt.subplots(figsize=(18, 7))
        ax.plot(timestamps, y_test, label="True", linewidth=1.6, color="#1f77b4", alpha=0.85)
        ax.plot(timestamps[valid], pred[valid], label=f"{name} Predicted", linewidth=1.8, color=color)
        ax.set_title(f"Full Test Period Forecast – {name}", fontweight="bold", pad=20)
        ax.set_ylabel("Power (kW)")
        ax.set_xlabel("Date")
        ax.legend(frameon=True, fancybox=True, shadow=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_plot(fig, f"02_full_forecast_{name.lower()}_power", "d")

        # 2. One-week (same style as Part A)
        week_pred = pred[week_start:week_end]
        week_valid = ~np.isnan(week_pred)
        fig, ax = plt.subplots(figsize=(16, 6.5))
        ax.plot(week_timestamps, week_true, label="True", linewidth=2.4, color="#1f77b4")
        ax.plot(week_timestamps[week_valid], week_pred[week_valid],
                label=f"{name} Predicted", linewidth=2.4, color=color, alpha=0.98)
        ax.set_title(f"One Week Forecast – {name}\n{week_title}", fontweight="bold", pad=20, fontsize=17)
        ax.set_ylabel("Power (kW)", fontsize=14)
        ax.set_xlabel("")
        day_starts = week_timestamps[::24]
        day_centers = day_starts + pd.Timedelta(hours=12)
        ax.set_xticks(day_centers)
        ax.set_xticklabels([t.strftime("%a %b %d") for t in day_starts], fontsize=12.5, fontweight="medium")
        for midnight in day_starts:
            ax.axvline(midnight, color="gray", linestyle="-", linewidth=0.7, alpha=0.35)
        ax.grid(True, alpha=0.35)
        ax.margins(x=0.01)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=13)
        plt.tight_layout()
        save_plot(fig, f"03_one_week_forecast_{name.lower()}_power", "d")

        # 3. Daily pattern
        true_daily = np.zeros(24)
        pred_daily = np.zeros(24)
        counts = np.zeros(24)
        for i, ts in enumerate(timestamps):
            if not valid[i]: continue
            h = ts.hour
            true_daily[h] += y_test[i]
            pred_daily[h] += pred[i]
            counts[h] += 1
        true_daily /= np.maximum(counts, 1)
        pred_daily /= np.maximum(counts, 1)
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.plot(np.arange(24), true_daily, 'o-', label="True Average", markersize=10, linewidth=4, color="#1f77b4")
        ax.plot(np.arange(24), pred_daily, 's--', label=f"{name} Predicted", markersize=10, linewidth=4, color=color)
        ax.set_title(f"Average Daily Pattern – {name}", fontweight="bold", pad=20)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Power (kW)")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(frameon=True, fancybox=True, fontsize=13)
        ax.grid(True, alpha=0.4)
        save_plot(fig, f"04_daily_pattern_{name.lower()}_power", "d")
    

    print("\nGenerating AVERAGE DAILY PATTERN with ALL models on the same graph...")
    fig, ax = plt.subplots(figsize=(13, 8))

    # Compute true daily average
    true_daily = np.zeros(24)
    counts = np.zeros(24)
    for i, ts in enumerate(timestamps):
        h = ts.hour
        true_daily[h] += y_test[i]
        counts[h] += 1
    true_daily /= counts

    ax.plot(np.arange(24), true_daily, 'o-', label="True Average",
            markersize=11, linewidth=4.5, color="#1f77b4", alpha=0.98)

    # Plot each model
    for name, pred, color in model_configs:
        pred_daily = np.zeros(24)
        counts = np.zeros(24)
        for i, ts in enumerate(timestamps):
            if np.isnan(pred[i]):
                continue
            h = ts.hour
            pred_daily[h] += pred[i]
            counts[h] += 1
        pred_daily /= np.maximum(counts, 1)
        ax.plot(np.arange(24), pred_daily, 's--', label=f"{name} Predicted",
                markersize=9, linewidth=3.5, color=color, alpha=0.92)

    ax.set_title("Average Daily Consumption Pattern – All Models", fontweight="bold", pad=20, fontsize=18)
    ax.set_xlabel("Hour of Day", fontsize=14)
    ax.set_ylabel("Average Power (kW)", fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=13)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_plot(fig, "04_daily_pattern_all_models_power", "d")

    print("\nALL 15 POWER MODEL PLOTS GENERATED SUCCESSFULLY!")

# ======================================================================
# TRADE DATASET – CLASSIFICATION + REAL-WORLD GROWTH ANALYSIS
# ======================================================================
else:
    breakpoint()
    print("TRADE DATASET – CLASSIFICATION RESULTS")
    breakpoint()

    data = TradeDataset(seed=args.seed)
    y_test = data.y_test.ravel()

    pred_logreg = np.load(pred_dir / "pred_logreg.npy")
    pred_xgb    = np.load(pred_dir / "pred_xgboost.npy")
    pred_xgb_proba = np.load(pred_dir / "pred_xgboost_proba.npy")
    pred_ffnn   = np.load(pred_dir / "pred_ffnn.npy")
    pred_ffnn_proba = np.load(pred_dir / "pred_ffnn_proba.npy") if (pred_dir / "pred_ffnn_proba.npy").exists() else None

    # Performance table
    results = pd.DataFrame({
        'Model': ['Logistic Reg', 'XGBoost', 'FFNN'],
        'Accuracy': [accuracy_score(y_test, p) for p in [pred_logreg, pred_xgb, pred_ffnn]],
        'F1-Score': [f1_score(y_test, p) for p in [pred_logreg, pred_xgb, pred_ffnn]]
    }).round(4)

    print("CLASSIFICATION PERFORMANCE – Predicting >10% Trade Growth")
    print(results.to_string(index=False))

    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    dfm = results.melt('Model', var_name='Metric', value_name='Score')
    sns.barplot(data=dfm, x='Model', y='Score', hue='Metric', ax=ax,
                palette="Set1", edgecolor="black", linewidth=1.5)
    ax.set_title("Classification Performance Comparison", fontweight="bold", pad=20)
    ax.set_ylim(0, 1)
    save_plot(fig, "05_classification_metrics_trade", "d")

    # Confusion Matrix (XGBoost)
    cm = confusion_matrix(y_test, pred_xgb)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax, linewidths=2,
                xticklabels=['No Growth', '>10% Growth'],
                yticklabels=['No Growth', '>10% Growth'])
    ax.set_title("Confusion Matrix – XGBoost Classifier", fontweight="bold", pad=20)
    save_plot(fig, "06_confusion_matrix_trade", "d")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, pred_xgb_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(fpr, tpr, color="#1f77b4", lw=4, label=f'XGBoost (AUC = {roc_auc:.3f})')
    if pred_ffnn_proba is not None:
        fpr_f, tpr_f, _ = roc_curve(y_test, pred_ffnn_proba)
        auc_f = auc(fpr_f, tpr_f)
        ax.plot(fpr_f, tpr_f, color="#ff7f0e", lw=3, label=f'FFNN (AUC = {auc_f:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve – Trade Growth Classification', fontweight="bold", pad=20)
    ax.legend(loc="lower right", frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)
    save_plot(fig, "07_roc_curve_trade", "d")

    # SHAP Plot
    print("Generating SHAP summary plot...")
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "xgboost_cls.json")
    X_sample = data.X_test[:2000]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=data.get_feature_names(),
                      show=False, plot_type="bar", color="#1f77b4")
    plt.title("SHAP Feature Importance – XGBoost Trade Growth Classifier", fontweight="bold", pad=20)
    save_plot(plt.gcf(), "08_shap_summary_trade", "d")
    plt.close('all')

    # ======================================================================
    # REAL-WORLD ANALYSIS: Which country pairs grew >10% in 2000–2014?
    # ======================================================================
    breakpoint()
    print("REAL-WORLD ANALYSIS: Which country pairs grew >10% in 2000–2014?")
    breakpoint()

    # Load country names
    try:
        from helpers.country_name_mapping import get_country_name
        name = get_country_name
        print("Using real country names")
    except Exception:
        print("Warning: country_name_mapping.py not found – using COW codes")
        def name(x): return f"Country {int(x)}"

    test_df = data.df_test.copy()
    test_df['country1'] = test_df['ccode1'].apply(name)
    test_df['country2'] = test_df['ccode2'].apply(name)

    # Actual growth cases
    growth_cases = test_df[test_df['target'] == 1].copy()
    total = len(test_df)
    grew = len(growth_cases)
    pct = grew / total * 100

    print(f"→ {grew:,} out of {total:,} dyad-years grew >10% ({pct:.2f}%)")

    # ───────────────────────────────
    # 1. Top 15 by GROWTH RATE (%)
    # ───────────────────────────────
    top_growth = (
        growth_cases.groupby(['ccode1', 'ccode2', 'country1', 'country2'])['growth_next']
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_growth['pair'] = top_growth['country1'] + " → " + top_growth['country2']
    top_growth['growth_pct'] = (top_growth['growth_next'] * 100).round(1)

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(range(len(top_growth)-1, -1, -1), top_growth['growth_pct'],
                   color="#d62728", edgecolor="black", alpha=0.9)

    ax.set_yticks(range(len(top_growth)))
    ax.set_yticklabels(top_growth['pair'][::-1], fontsize=12)
    ax.set_xlabel("Average Annual Trade Growth (%)", fontsize=14)
    ax.set_title(f"Top 15 Fastest-Growing Trade Relationships (2000–2014)\n"
                 f"{grew:,}/{total:,} dyads grew >10% ({pct:.1f}%)",
                 fontweight="bold", pad=20, fontsize=16)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height()/2,
                f"{w:.1f}%", va='center', fontweight='bold', fontsize=11)

    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    plt.tight_layout()
    save_plot(fig, "09_top_growing_trade_pairs", "d")
    print("→ Saved: 09_top_growing_trade_pairs.pdf")

    # ───────────────────────────────
    # 2. Top 15 by TRADE VOLUME in 2014
    # ───────────────────────────────
    print("\nTop 15 largest trade relationships by volume in 2014...")
    latest_year = test_df[test_df['year'] == test_df['year'].max()].copy()

    top_volume = (
        latest_year.groupby(['ccode1', 'ccode2', 'country1', 'country2'])['flow']
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_volume['pair'] = top_volume['country1'] + " ↔ " + top_volume['country2']
    top_volume['volume_billion'] = (top_volume['flow'] / 1e3).round(2)  # to billion USD

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(range(len(top_volume)-1, -1, -1), top_volume['volume_billion'],
                   color="#1f77b4", edgecolor="black", alpha=0.9)
    
    ax.set_yticks(range(len(top_volume)))
    ax.set_yticklabels(top_volume['pair'][::-1], fontsize=12)
    ax.set_xlabel("Average Annual Trade Volume (billion current USD)", fontsize=14)
    ax.set_title("Top 15 Largest Trade Relationships by Volume (2014)",
                 fontweight="bold", pad=20, fontsize=16)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax.text(w + 10, bar.get_y() + bar.get_height()/2,
                f"${w:.1f}B", va='center', fontweight='bold', fontsize=11)

    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    plt.tight_layout()
    save_plot(fig, "10_largest_trade_pairs_volume", "d")
    print("→ Saved: 10_largest_trade_pairs_volume.pdf")

    print("\nTop 5 largest trade relationships in 2014:")
    for _, row in top_volume.head(5).iterrows():
        print(f"   • {row['pair']}: ${row['volume_billion']:.2f} billion/year")

    breakpoint()
    print("All trade classification + real-world growth & volume analysis completed!")
    breakpoint()

# ======================================================================
breakpoint()
print(f" PART D ({args.dataset.upper()}) – ALL FIGURES GENERATED SUCCESSFULLY!")
print(f" • All plots saved to {fig_dir}/")
breakpoint()

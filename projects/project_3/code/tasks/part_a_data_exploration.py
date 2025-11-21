# tasks/part_a_exploration.py
"""
Part A – Data Exploration & Preprocessing
FYS-STK3155 / FYS4155 – Project 3

Generates exploratory plots and saves clean CSVs.
Run via: python code/project3.py --dataset power --part a
"""

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Import our beautiful divider + utilities
from utils import ensure_dir, save_plot, breakpoint
from datasets.power import PowerDataset
from datasets.trade import TradeDataset


# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Part A – Data Exploration")
parser.add_argument("--dataset", type=str, choices=["power", "trade"], required=True,
                    help="Dataset to explore: 'power' or 'trade'")
parser.add_argument("--seed", type=int, default=1993, help="Random seed")
parser.add_argument("--part", type=str, help="Internal flag (ignored)")

args, _ = parser.parse_known_args()

np.random.seed(args.seed)
sns.set(style="whitegrid", font_scale=1.35, palette="deep", rc={"grid.alpha": 0.3})

ensure_dir("figures/part_a")
ensure_dir("data")

breakpoint()
print(f" PART A – DATA EXPLORATION: {args.dataset.upper()} DATASET")
breakpoint()


# ======================================================================
# POWER DATASET – Electricity Consumption
# ======================================================================
if args.dataset == "power":
    print("Loading PowerDataset (household electricity consumption)...")
    dataset = PowerDataset(seed=args.seed)
    df = dataset.df_full

    print(f"Time range      : {df.index.min()} → {df.index.max()}")
    print(f"Observations    : {len(df):,}")
    print(f"Mean power      : {df['Global_active_power'].mean():.3f} kW")
    print(f"Std deviation   : {df['Global_active_power'].std():.3f} kW\n")

    print("Generating exploratory plots...")

    # 1. Full time series
    fig, ax = plt.subplots(figsize=(18, 6))
    df['Global_active_power'].plot(ax=ax, linewidth=0.9, color="#1f77b4")
    ax.set_title("Household Global Active Power – Full Hourly Series (2006–2010)",
                 fontweight="bold", pad=20, fontsize=16)
    ax.set_ylabel("Power (kW)", fontsize=14)
    ax.set_xlabel("Date", fontsize=14)
    save_plot(fig, "power_full_timeseries", "a")

    # 2. TYPICAL WEEK – clean, professional, matches Part D style
    typical_week = df.loc["2010-06-13":"2010-06-20"]  # Mon → Sun, normal winter week

    fig, ax = plt.subplots(figsize=(16, 6.5))
    typical_week['Global_active_power'].plot(
        ax=ax, linewidth=2.4, color="#ff7f0e", alpha=0.98
    )

    ax.set_title("One Week of Electricity Consumption – Typical Week (Jun 13–20, 2010)",
                 fontweight="bold", pad=20, fontsize=17)
    ax.set_ylabel("Power (kW)", fontsize=14)
    ax.set_xlabel("")  # Clean look – dates are self-explanatory

    # ONE CLEAN TICK PER DAY – centered at noon with day name + date
    day_starts = typical_week.index[::24]                    # midnight of each day
    day_centers = day_starts + pd.Timedelta(hours=12)        # noon for perfect centering

    ax.set_xticks(day_centers)
    ax.set_xticklabels([t.strftime("%a %b %d") for t in day_starts],
                       fontsize=12.5, fontweight="medium")

    # Subtle day separators at midnight
    for midnight in day_starts:
        ax.axvline(midnight, color="gray", linestyle="-", linewidth=0.7, alpha=0.35)

    ax.grid(True, alpha=0.35, linewidth=0.8)
    ax.margins(x=0.01)
    plt.tight_layout()

    save_plot(fig, "power_one_week_typical", "a")
    print("   → Saved: power_one_week_typical.pdf (clean daily ticks, perfect match with Part D)")

    # 3. Average daily pattern
    daily = df.groupby(df.index.hour)['Global_active_power'].mean()
    fig, ax = plt.subplots(figsize=(13, 7))
    daily.plot(ax=ax, marker="o", markersize=10, linewidth=4, color="#2ca02c")
    ax.set_title("Average Daily Electricity Consumption Pattern",
                 fontweight="bold", pad=20, fontsize=16)
    ax.set_xlabel("Hour of Day", fontsize=14)
    ax.set_ylabel("Average Power (kW)", fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.4)
    save_plot(fig, "power_daily_pattern", "a")

    # Save clean dataset
    clean_path = Path("data/power_hourly_clean.csv")
    df.to_csv(clean_path)
    print(f"Clean power data saved → data/power_hourly_clean.csv")


# ======================================================================
# TRADE DATASET – Bilateral Trade Flows (1870–2014) – 100% REAL COUNTRY NAMES
# ======================================================================
else:  # trade
    print("Loading TradeDataset (bilateral trade flows 1870–2014)...")
    dataset = TradeDataset(seed=args.seed)
    df = dataset.df_full.copy()

    # === ABSOLUTE PATH FIX: Force load country_name_mapping.py ===
    import sys
    from pathlib import Path

    # Add utils directory to path (critical!)
    utils_path = Path(__file__).parent.parent / "utils"
    sys.path.insert(0, str(utils_path))

    # Now import — this will work 100%
    try:
        from helpers.country_name_mapping import get_country_name
        print("SUCCESS: Country name mapping loaded – using real country names")
        def name(ccode):
            return get_country_name(ccode)
    except Exception as e:
        print(f"FAILED to load country_name_mapping.py: {e}")
        print("Falling back to ccodes... (but this should NOT happen)")
        def name(ccode):
            return f"Country {int(ccode)}"

    print(f"Dyad-years      : {len(df):,}")
    print(f"Unique pairs    : {df[['ccode1', 'ccode2']].drop_duplicates().shape[0]:,}")
    print(f"Years covered   : {df['year'].min()} – {df['year'].max()}")
    print(f">10% growth obs : {df['target'].sum():,} ({df['target'].mean():.2%})\n")

    print("Generating beautiful exploratory plots with REAL country names...")

    # 1. Top 10 trading nations over time
    print("   → Plot 1: Top 10 trading nations over time")
    total_trade = df.groupby(['year', 'ccode1'])['flow'].sum().reset_index()
    top10_per_year = (
        total_trade.sort_values(['year', 'flow'], ascending=[True, False])
                   .groupby('year')
                   .head(10)
                   .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = sns.color_palette("tab10", 10)
    plotted = set()

    for ccode, group in top10_per_year.groupby('ccode1'):
        if ccode in plotted:
            continue
        label = name(ccode)
        color = colors[len(plotted) % 10]
        ax.plot(group['year'], group['flow'], label=label, linewidth=2.8, color=color)
        plotted.add(ccode)
        if len(plotted) >= 10:
            break

    ax.set_title("Top 10 Trading Nations by Total Trade Volume (1870–2014)",
                 fontweight="bold", pad=20, fontsize=18)
    ax.set_ylabel("Total Trade Flow (million current USD)", fontsize=14)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc="upper left", ncol=2)
    plt.tight_layout()
    save_plot(fig, "trade_top10_nations_over_time", "a")

    # 2. USA–China
    print("   → Plot 2: USA–China bilateral trade")
    usa, china = 2, 710
    us_cn = df[((df['ccode1'] == usa) & (df['ccode2'] == china)) |
               ((df['ccode1'] == china) & (df['ccode2'] == usa))].copy()

    if not us_cn.empty:
        us_cn_yearly = us_cn.groupby('year')['flow'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(us_cn_yearly['year'], us_cn_yearly['flow'], linewidth=4, color="#d62728")
        ax.set_title(f"Bilateral Trade: {name(usa)} ↔ {name(china)} (1870–2014)",
                     fontweight="bold", pad=20, fontsize=18)
        ax.set_ylabel("Average Annual Trade Flow (million current USD)", fontsize=14)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        post_1979 = us_cn_yearly[us_cn_yearly['year'] >= 1979]
        if not post_1979.empty:
            first = post_1979.iloc[0]
            ax.annotate("Diplomatic relations\n& trade begins (1979)",
                        xy=(first['year'], first['flow']),
                        xytext=(1985, first['flow'] * 10),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                        fontsize=12, ha='center', bbox=dict(boxstyle="round", fc="0.95"))

        plt.tight_layout()
        save_plot(fig, "trade_usa_china", "a")

    # 3. Germany–France
    print("   → Plot 3: Germany–France bilateral trade")
    ger, fra = 255, 220
    ger_fra = df[((df['ccode1'] == ger) & (df['ccode2'] == fra)) |
                 ((df['ccode1'] == fra) & (df['ccode2'] == ger))].copy()

    if not ger_fra.empty:
        ger_fra_yearly = ger_fra.groupby('year')['flow'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(ger_fra_yearly['year'], ger_fra_yearly['flow'], linewidth=4, color="#2ca02c")
        ax.set_title(f"Bilateral Trade: {name(ger)} ↔ {name(fra)} (1870–2014)",
                     fontweight="bold", pad=20, fontsize=18)
        ax.set_ylabel("Average Annual Trade Flow (million current USD)", fontsize=14)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot(fig, "trade_germany_france", "a")

    # 4. Global trade network density
    print("   → Plot 4: Global trade network density (1950 vs 2010)")
    years = [1950, 2010]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    for i, year in enumerate(years):
        year_df = df[df['year'] == year]
        if year_df.empty:
            axes[i].text(0.5, 0.5, f"No data\nin {year}", transform=axes[i].transAxes,
                         ha='center', va='center', fontsize=16, color='gray')
            axes[i].set_title(f"Global Trade Network – {year}", fontweight="bold")
            continue

        dyads = set(zip(year_df['ccode1'], year_df['ccode2']))
        n_dyads = len(dyads)
        countries = set(year_df['ccode1']) | set(year_df['ccode2'])
        n_countries = len(countries)
        possible = n_countries * (n_countries - 1) if n_countries > 1 else 1
        density = n_dyads / possible

        ax = axes[i]
        bars = ax.bar(["Observed\nLinks", "Possible\nLinks"], [n_dyads, possible],
                      color=["#1f77b4", "#ff7f0e"], edgecolor="black", alpha=0.9)
        ax.set_title(f"Global Trade Network – {year}\n"
                     f"Density = {density:.1%}\n"
                     f"({n_dyads:,} / {possible:,} directed)",
                     fontweight="bold", fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + possible*0.01,
                    f'{int(h):,}', ha='center', va='bottom', fontsize=11)

    plt.suptitle("Evolution of the Global Trade Network: 1950 → 2010",
                 fontsize=20, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig, "trade_network_evolution", "a")

    # Save processed dataset
    clean_path = Path("data/trade_processed.csv")
    df.to_csv(clean_path, index=False)
    print(f"Processed trade data saved → data/trade_processed.csv")

# ======================================================================
breakpoint()
print(f" PART A ({args.dataset.upper()}) – COMPLETED SUCCESSFULLY!")
print(" • All plots saved → figures/part_a/")
print(" • Clean data saved → data/")
breakpoint()

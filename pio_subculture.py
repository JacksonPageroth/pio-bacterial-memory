#!/usr/bin/env python3
"""
Pioreactor Growth Curves Analysis

This script demonstrates how to:
1. Unzip & rename CSVs for experiments.
2. Merge experiment data and transform OD readings.
3. Identify & filter subculture artifacts.
4. Classify growth steps and plot data.
5. Perform regression (growth rate) calculations.
6. Optionally run external GrowthRates software (GRME).
7. Optionally predict time to reach specified OD thresholds.

"""

import os
import glob
import math
import zipfile

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from statsmodels.formula.api import ols

import warnings
import statsmodels.api as sm

from adjustText import adjust_text
from sklearn.linear_model import LinearRegression

sns.set_theme(context="poster", style="ticks")

###############################################################################
# Global Config & Dictionaries
###############################################################################
NAME_DICT = {
    # 'worker1': 'W1 - Replicate 3',
    'worker2': 'W2 - 35 ppt Control',
    'worker3': 'W3 - Replicate 1',
    'worker4': 'W4 - Replicate 2',
    'worker5': 'W5 - 55 ppt Control'
}

EQUATIONS = {
    # 'W1  - Replicate 3': (0, 0),  # Not defined
    "W2 - 35 ppt Control": (1.348628103, 0.077),
    "W3 - Replicate 1": (2.073660771, 0.077),
    "W4 - Replicate 2": (2.204291876, 0.077),
    "W5 - 55 ppt Control": (1.169484467, 0.077),
}

THRESHOLD_OD = 0.24
OPTIMAL_CONTROLS = ["W2 - 35 ppt Control"]
STRESSED_CONTROLS = ["W5 - 55 ppt Control"]
CONTROLS = ["W2 - 35 ppt Control", "W5 - 55 ppt Control"]
REPLICATES = ["W3 - Replicate 1", "W4 - Replicate 2"]

WEIRD_STEPS = {
    "W2 - 35 ppt Control": [15, 16, 18, 21],
    "W3 - Replicate 1": [9, 10, 16],
    "W4 - Replicate 2": [10, 11, 16],
    "W5 - 55 ppt Control": [3, 9, 10],
}

COL_ORDER = ["W3 - Replicate 1", "W4 - Replicate 2", "W2 - 35 ppt Control", "W5 - 55 ppt Control"]


###############################################################################
# Data I/O Functions
###############################################################################
def unzip_and_rename_csv(zip_file_path, extract_to_folder=None, renamed_csv="latest_od_data.csv"):
    """
    Unzips a file containing one CSV and renames it to a specified name.
    """
    if extract_to_folder is None:
        extract_to_folder = os.path.dirname(zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
        csv_file = None
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith('.csv'):
                csv_file = os.path.join(extract_to_folder, file_name)
                break

    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the ZIP archive.")

    renamed_csv_path = os.path.join(extract_to_folder, renamed_csv)
    os.rename(csv_file, renamed_csv_path)
    return renamed_csv_path


def load_data_for_exp1() -> str:
    """
    Unzip the exp1 data into 'exp1.csv'.

    Returns:
        str: Path to the CSV file for Exp1.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp1_zip_path = os.path.join(base_dir, 'od_logs', 'exp1.zip')
    return unzip_and_rename_csv(exp1_zip_path, renamed_csv="exp1.csv")


def find_latest_download_csv() -> str:
    """
    Find the most recent .csv in ~/Downloads, or default to './od_logs/latest_od_data.csv'.
    """

    csv_files = glob.glob(os.path.expanduser('~/Downloads/*.csv'))
    if len(csv_files) == 0:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, 'od_logs', 'latest_od_data.csv')
    else:
        return max(csv_files, key=os.path.getmtime)


###############################################################################
# Data Preparation
###############################################################################
def merge_experiments(exp1_csv: str, latest_csv: str) -> pl.LazyFrame:
    """
    Read and merge two experimental CSV files via Polars lazy scanning.
    """
    exp1_lf = (
        pl.scan_csv(exp1_csv)
        .select("od_reading", "timestamp_localtime", "pioreactor_unit")
        .rename({"od_reading": "OD600", "timestamp_localtime": "Time", "pioreactor_unit": "Unit"})
    )

    df_latest_lf = (
        pl.scan_csv(latest_csv)
        .select("od_reading", "timestamp_localtime", "pioreactor_unit")
        .rename({"od_reading": "OD600", "timestamp_localtime": "Time", "pioreactor_unit": "Unit"})
    )

    return pl.concat([exp1_lf, df_latest_lf])


def transform_data(df: pl.LazyFrame) -> pl.DataFrame:
    """
    - Rename Pioreactor units
    - Apply OD600 transformations
    - Convert time to datetime
    - Filter subculture artifacts
    - Compute 'Hours' since start
    - Remove a known artifact
    - Sort data by Unit, Hours

    Returns:
        pl.DataFrame: Eager DataFrame with transformations applied.
    """
    df = df.with_columns(pl.col("Unit").replace_strict(NAME_DICT))

    # Apply the equation
    a = pl.col("Unit").replace_strict(EQUATIONS).list.first()
    b = pl.col("Unit").replace_strict(EQUATIONS).list.last()
    df = df.with_columns((pl.col("OD600") * a + b))

    # Convert Time to datetime
    df = df.with_columns(pl.col("Time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))

    # Filter initial subculture artifacts (first 2 min)
    df = df.filter(pl.col("Time") >= pl.col("Time").min().dt.offset_by("2m"))

    # Hours since start
    df = df.with_columns(
        (pl.col("Time").sub(pl.col("Time").min()).dt.total_seconds() / 3600).alias("Hours")
    )

    # Remove known artifact for W5 - 55 ppt Control
    df = df.filter(
        (
                pl.col("Unit").eq("W5 - 55 ppt Control")
                & pl.col("Hours").ge(104.2)
                & pl.col("Hours").le(104.4)
        ).not_()
    )

    df = df.group_by("Unit", "Time", "Hours").agg(pl.col("OD600").mean().alias("OD600")).sort(["Unit", "Hours"])
    return df.collect()


###############################################################################
# Plotting
###############################################################################
def plot_10min_bins_whole_experiment(df: pl.DataFrame) -> None:
    """
    Take a rolling average in 10m bins and plot growth curves.
    """
    plot_df = (
        df.with_columns(pl.col("Time").dt.truncate("10m").alias("Time"))
        .group_by("Unit", "Time")
        .agg(pl.col("OD600").mean().alias("Mean OD600"))
        .with_columns(
            (pl.col("Time").sub(pl.col("Time").min()).dt.total_seconds() / 3600).alias("Hours")
        )
    )

    hue_order = ["W2 - 35 ppt Control", "W5 - 55 ppt Control", "W3 - Replicate 1", "W4 - Replicate 2"]
    f, ax = plt.subplots(figsize=(40, 7))
    sns.lineplot(
        data=plot_df.to_pandas(),
        x="Time",
        y="Mean OD600",
        hue="Unit",
        hue_order=hue_order,
        markersize=8,
        sizes=(1, 8),
        ax=ax,
        palette=["blue", "red", "green", "mediumseagreen"],
    )

    plt.axhline(y=THRESHOLD_OD, color='black', linestyle='--')
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.set_xlabel("Time")

    # Secondary date axis
    secax = ax.secondary_xaxis(-0.1)
    secax.xaxis.set_major_locator(mdates.DayLocator())
    secax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    secax.set_xlabel("Date")

    ax.set_title('Pioreactor OD Curves binned in 10m intervals')
    plt.grid(True)
    plt.savefig("plots/general/whole_experiment_binned_od_curves.pdf", bbox_inches='tight', dpi=300)


###############################################################################
# Step Classification
###############################################################################
def find_steps(class_df: pl.DataFrame, shift: int) -> pl.DataFrame:
    """
    Increment 'Step' each time there's a drop > 0.03 (indicating subculture).
    Then classify as 'Optimal' or 'Stressed' depending on replicate or control.
    """
    class_df = class_df.sort(["Unit", "Hours"])
    step_cond = (pl.col("OD600").shift(shift) - pl.col("OD600") > 0.03)

    class_df = class_df.with_columns(
        pl.when(step_cond)
        .then(1)
        .otherwise(0)
        .cum_sum()
        .add(1)
        .over(pl.col("Unit"))
        .alias("Step")
    )

    # Map odd/even steps to "Optimal"/"Stressed" for REPLICATES
    class_df = class_df.with_columns(
        pl.when(
            (pl.col("Step") % 2 == 1) & (pl.col("Unit").is_in(REPLICATES))
        )
        .then(pl.lit("Optimal"))
        .otherwise(
            pl.when(
                (pl.col("Step") % 2 == 0) & (pl.col("Unit").is_in(REPLICATES))
            )
            .then(pl.lit("Stressed"))
            .otherwise(
                pl.when(pl.col("Unit").is_in(STRESSED_CONTROLS))
                .then(pl.lit("Stressed"))
                .otherwise(
                    pl.when(pl.col("Unit").is_in(OPTIMAL_CONTROLS))
                    .then(pl.lit("Optimal"))
                )
            )
        )
        .alias("Condition")
    )
    return class_df


def classify_steps(df: pl.DataFrame) -> pl.DataFrame:
    """
    1) Classify steps with shift=5, discard first/last 20 points from each step.
    2) Re-classify with shift=1, rename 'Condition' -> 'Salinity'.
    """
    df = find_steps(df, shift=5)

    # Discard the first and last 20 points in every step
    filter_points = 20
    df = (
        df.with_columns(pl.cum_count("OD600").over(["Unit", "Step"]).alias("index"))
        .filter(pl.col("index") > filter_points)
        .filter(pl.col("index") < pl.col("index").max().over(["Unit", "Step"]) - filter_points)
        .drop("index")
    )

    # Re-classify after trimming
    df = df.drop(["Step", "Condition"])
    df = find_steps(df, shift=1)
    df = df.rename({"Condition": "Salinity"})
    return df


def plot_classified_steps(df: pl.DataFrame) -> None:
    """
    Plot the new 'OD600' vs Hours with steps labeled as Optimal / Stressed.
    """
    g = sns.relplot(
        x="Hours",
        y="OD600",
        hue="Salinity",
        style="Step",
        col="Unit",
        col_wrap=2,
        col_order=COL_ORDER,
        height=10,
        markersize=8,
        data=df.to_pandas(),
        kind="line",
        hue_order=["Optimal", "Stressed"],
        palette=["blue", "red"],
    )

    for ax in g.axes.flatten():
        ax.axhline(y=THRESHOLD_OD, color='black', linestyle='--')
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle('Pioreactor OD Curves')
    plt.savefig("plots/general/classified_steps_od_curves.pdf", bbox_inches='tight', dpi=300)


###############################################################################
# Filtering Weird Steps & Binning
###############################################################################
def filter_weird_steps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove steps that have been identified as outliers/weird from the analysis.
    """
    print(f"Removed steps: {WEIRD_STEPS}")
    return df.filter(
        *[
            ~((pl.col("Unit") == unit) & pl.col("Step").is_in(steps))
            for unit, steps in WEIRD_STEPS.items()
        ]
    )


def bin_data(df: pl.DataFrame, log: bool) -> pl.DataFrame:
    """
    Bin data in 10-minute increments, compute Mean OD600, then log-transform.
    """
    mean_df = (df
        .with_columns(pl.col("Time").dt.truncate("10m").alias("Time"))
        .group_by("Unit", "Time", "Salinity", "Step")
        .agg(pl.col("OD600").mean().alias("Mean OD600"))
        .with_columns(
            (pl.col("Time").sub(pl.col("Time").min()).dt.total_seconds() / 3600).alias("Hours")
        )
    )
    
    if log:
        mean_df = mean_df.with_columns(pl.col("Mean OD600").log().alias("ln(Mean OD600)"))
    return mean_df


def plot_binned_data(mean_df: pl.DataFrame) -> None:
    """
    Plot the binned Mean OD600 and ln(Mean OD600).
    """
    # Plot binned OD
    g = sns.relplot(
        x="Hours",
        y="Mean OD600",
        hue="Salinity",
        style="Step",
        col="Unit",
        col_wrap=2,
        col_order=COL_ORDER,
        height=10,
        markers=True,
        markersize=8,
        data=mean_df.to_pandas(),
        kind="line",
        hue_order=["Optimal", "Stressed"],
        palette=["blue", "red"],
    )
    for ax in g.axes.flatten():
        ax.axhline(y=THRESHOLD_OD, color='black', linestyle='--')
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle('Pioreactor binned OD Curves')
    plt.savefig("plots/general/binned_od_curves.pdf", bbox_inches='tight', dpi=300)

    # Plot binned ln(OD)
    g = sns.relplot(
        x="Hours",
        y="ln(Mean OD600)",
        hue="Salinity",
        style="Step",
        col="Unit",
        col_wrap=2,
        col_order=COL_ORDER,
        height=10,
        markers=True,
        markersize=8,
        data=mean_df.to_pandas(),
        kind="line",
        hue_order=["Optimal", "Stressed"],
        palette=["blue", "red"],
    )
    for ax in g.axes.flatten():
        ax.axhline(y=np.log(THRESHOLD_OD), color='black', linestyle='--')
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle('Pioreactor binned ln(OD) Curves')
    plt.savefig("plots/general/binned_ln_OD_curves.pdf", bbox_inches='tight', dpi=300)


###############################################################################
# Regression & Plots
###############################################################################
def plot_fit_with_selected_points(
        ldf: pd.DataFrame,
        unit: str,
        step: str,
        slope: float,
        intercept: float,
        selected_indices: list,
        significant: float,
        lag_point: tuple,
        filename: str
):
    """
    Plot original data with the line fit and highlight points used for slope calculation.
    Also show a horizontal line for the lag and intersection point.
    """
    group = ldf[(ldf['Unit'] == unit) & (ldf['Step'] == step)].sort_values('Hours')
    x = group['Hours'].values
    y = group['ln(Mean OD600)'].values

    intersection_x, intersection_y = lag_point
    intersection_x += x[0]  # shift to absolute hours

    # Horizontal lag line
    x_lag_fit = np.linspace(x.min(), x.max(), 100)
    y_lag_fit = np.full_like(x_lag_fit, intersection_y)

    # Exponential fit line
    x_fit = np.linspace(x.min(), x.max(), 100)
    x_fit = x_fit[x_fit >= intersection_x - 1]
    y_fit = slope * x_fit + intercept

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Original Data", color="gray", alpha=0.7)

    selected_points = group.iloc[selected_indices]
    plt.scatter(
        selected_points['Hours'],
        selected_points['ln(Mean OD600)'],
        label="Selected Points",
        color="orange",
        edgecolor="black",
        zorder=5
    )

    plt.plot(x_fit, y_fit, label=f"Main Fitted Line (Slope={slope:.2f})", color="blue", linewidth=2)
    plt.plot(x_lag_fit, y_lag_fit, label="Lag fit", color="green", linestyle="--", linewidth=2)

    # Mark intersection
    plt.scatter([intersection_x], [intersection_y], color="red", zorder=10,
                label=f"Intersection (x={intersection_x:.2f})")
    plt.axvline(x=intersection_x, color="red", linestyle=":", linewidth=1, alpha=0.7)

    plt.xlabel("Hours")
    plt.ylabel("ln(Mean OD600)")
    plt.title(f"Unit: {unit}, Step: {step}, R2 {significant:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{filename}_{unit}_{step}.pdf", bbox_inches='tight', dpi=300)


def calculate_step_slopes_and_lag(df: pl.DataFrame, window_length=6) -> pd.DataFrame:
    """
    Calculate slope & R^2 for each step in Polars DataFrame:
      - Identify subset of points where rolling slope >= 95% of max rolling slope.
      - Fit linear regression to that subset, compute slope & R^2.
      - Estimate lag time by looking at all points within 2x std of the first 30 minutes.

    Returns:
        pd.DataFrame: Columns 'Unit', 'Step', 'Slope', 'R2', 'Points', 'Salinity',
                      'Significant', 'Lag', 'SelectedIndices' (for optional plotting).
    """
    pdf = df.to_pandas()
    results = []

    grouped = pdf.groupby(['Unit', 'Step'])
    for (unit, step), group in grouped:
        group = group.sort_values('Hours')
        x = group['Hours'].values
        y = group['ln(Mean OD600)'].values

        # Calculate rolling slopes
        rolling_slopes = []
        for i in range(len(x) - (window_length - 1)):
            x_window = x[i:i + window_length].reshape(-1, 1)
            y_window = y[i:i + window_length]
            model = LinearRegression().fit(x_window, y_window)
            rolling_slopes.append(model.coef_[0])

        group['Rolling Slope'] = rolling_slopes + [np.nan] * (window_length - 1)
        group.reset_index(drop=True, inplace=True)

        max_slope = np.nanmax(rolling_slopes) if rolling_slopes else 0
        matching_indices = group.index[group['Rolling Slope'] >= 0.95 * max_slope].tolist()

        # Include up to window_length rows after each match
        expanded_indices = set()
        for idx in matching_indices:
            expanded_indices.update(range(idx, min(idx + window_length, len(x))))

        slope_points = group.loc[list(expanded_indices)].sort_values("Hours")
        if len(slope_points) > 1:
            x_filtered = slope_points['Hours'].values.reshape(-1, 1)
            y_filtered = slope_points['ln(Mean OD600)'].values
            model = LinearRegression().fit(x_filtered, y_filtered)
            slope = model.coef_[0]
            r2 = model.score(x_filtered, y_filtered)

            # Calculate lag
            first_point_y = group[group["Hours"] - group.iloc[0]["Hours"] <= 0.5]['ln(Mean OD600)'].mean()
            lag_time = ((first_point_y - model.intercept_) / slope) - group.iloc[0]['Hours']
            # Calculate lag, by getting every point who's Y is within the standard deviation of the first 30 minutes
            # subset_30 = group[group["Hours"] - group.iloc[0]["Hours"] <= 0.5]
            # assert not subset_30.empty, "No points in the first 30 minutes"
            # baseline_mean = subset_30["ln(Mean OD600)"].mean()
            # baseline_std = subset_30["ln(Mean OD600)"].std()
            # lag_time = group[group["ln(Mean OD600)"] <= baseline_mean + 2 * baseline_std]["Hours"].max() - \
            #            group.iloc[0]["Hours"]

            # Instead of plotting here, store the indices for optional plotting
            selected_indices = slope_points.index.tolist()

            results.append({
                'Unit': unit,
                'Step': step,
                'Slope': slope,
                'R2': r2,
                'Points': len(group),
                'Salinity': group['Salinity'].values[0],
                'Significant': round(r2, 2) > 0.95,
                'Lag': lag_time,
                'Intercept': model.intercept_,
                'SelectedIndices': selected_indices,
                'x_data': x_filtered.reshape(y_filtered.shape),
                'y_data': y_filtered
            })

    return pd.DataFrame(results)


def calculate_alt_step_slopes_and_lag(df: pl.DataFrame, window_length=6) -> pd.DataFrame:
    """
    Calculate slope & R^2 for each step in Polars DataFrame:
      - Identify subset of points where rolling slope >= 50% of max rolling slope.
      - Estimates lag as the time when ln(Mean OD600) <= baseline + 2*std in the first 30 minutes.
      - Estimate lag time by looking at all points within 2x std of the first 30 minutes.
      - Fit linear regression to that subset, compute slope & R^2.
\
    Returns:
        pd.DataFrame: Columns 'Unit', 'Step', 'Slope', 'R2', 'Points', 'Salinity',
                      'Significant', 'Lag', 'SelectedIndices' (for optional plotting).
    """
    pdf = df.to_pandas()
    results = []

    grouped = pdf.groupby(['Unit', 'Step'])
    for (unit, step), group in grouped:
        group = group.sort_values('Hours')
        x = group['Hours'].values
        y = group['ln(Mean OD600)'].values

        # Calculate rolling slopes
        rolling_slopes = []
        for i in range(len(x) - (window_length - 1)):
            x_window = x[i:i + window_length].reshape(-1, 1)
            y_window = y[i:i + window_length]
            model = LinearRegression().fit(x_window, y_window)
            rolling_slopes.append(model.coef_[0])

        group['Rolling Slope'] = rolling_slopes + [np.nan] * (window_length - 1)
        group.reset_index(drop=True, inplace=True)

        max_slope = np.nanmax(rolling_slopes) if rolling_slopes else 0
        matching_indices = group.index[group['Rolling Slope'] >= 0.50 * max_slope].tolist()

        # Include up to window_length rows after each match
        expanded_indices = set()
        for idx in matching_indices:
            expanded_indices.update(range(idx, min(idx + window_length, len(x))))

        slope_points = group.loc[list(expanded_indices)].sort_values("Hours")
        if len(slope_points) > 1:
            x_filtered = slope_points['Hours'].values.reshape(-1, 1)
            y_filtered = slope_points['ln(Mean OD600)'].values
            model = LinearRegression().fit(x_filtered, y_filtered)
            slope = model.coef_[0]
            r2 = model.score(x_filtered, y_filtered)

            # Calculate lag, by getting every point who's Y is within the standard deviation of the first 30 minutes
            subset_30 = group[group["Hours"] - group.iloc[0]["Hours"] <= 0.5]
            assert not subset_30.empty, "No points in the first 30 minutes"
            baseline_mean = subset_30["ln(Mean OD600)"].mean()
            baseline_std = subset_30["ln(Mean OD600)"].std()
            lag_time = group[group["ln(Mean OD600)"] <= baseline_mean + 2 * baseline_std]["Hours"].max() - \
                       group.iloc[0]["Hours"]

            # Instead of plotting here, store the indices for optional plotting
            selected_indices = slope_points.index.tolist()

            results.append({
                'Unit': unit,
                'Step': step,
                'Slope': slope,
                'R2': r2,
                'Points': len(group),
                'Salinity': group['Salinity'].values[0],
                'Significant': r2 > 0.95,
                'Lag': lag_time,
                'Intercept': model.intercept_,
                'SelectedIndices': selected_indices,
                'x_data': x_filtered,
                'y_data': y_filtered
            })

    return pd.DataFrame(results)


def compute_lines_are_different_test(slopes_and_r2: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    indices = list(slopes_and_r2.index)
    pairs = list(combinations(indices, 2))
    
    for idx1, idx2 in pairs:
        row1 = slopes_and_r2.loc[idx1]
        row2 = slopes_and_r2.loc[idx2]
        
        # Stack data from both regressions
        x_combined = np.concatenate([row1['x_data'], row2['x_data']])
        y_combined = np.concatenate([row1['y_data'], row2['y_data']])
        group = np.concatenate([np.zeros(len(row1['x_data'])), np.ones(len(row2['x_data']))])
        
        # Create dataframe with interaction term
        combined_df = pd.DataFrame({
            'x': x_combined,
            'y': y_combined,
            'group': group,
            'x_group_interaction': x_combined * group
        })
        
        # Fit model: y ~ x + group + x*group
        model = ols('y ~ x + group + x_group_interaction', data=combined_df).fit()
        
        # Extract interaction statistics
        interaction_coef = model.params['x_group_interaction']
        interaction_se = model.bse['x_group_interaction']
        interaction_pval = model.pvalues['x_group_interaction']
        
        reg1_id = f"Unit_{row1['Unit']}_Sal_{row1['Salinity']}"
        reg2_id = f"Unit_{row2['Unit']}_Sal_{row2['Salinity']}"
        
        comparisons.append({
            'Regression_1': reg1_id,
            'Regression_2': reg2_id,
            'Slope_1': row1['Slope'],
            'Slope_2': row2['Slope'],
            'Slope_Difference': row1['Slope'] - row2['Slope'],
            'Interaction_Coefficient': interaction_coef,
            'Interaction_SE': interaction_se,
            'Interaction_p_value': interaction_pval,
            'Significantly_Different': interaction_pval < 0.05
        })
    
    return pd.DataFrame(comparisons)


def plot_selected_fits(slopes_and_r2: pd.DataFrame, original_df: pd.DataFrame, filename: str) -> None:
    """
    Using the results from calculate_step_slopes_and_r2 (including 'SelectedIndices'),
    plot the original data with the fits for each row in slopes_and_r2.
    """
    for _, row in slopes_and_r2.iterrows():
        unit = row["Unit"]
        step = row["Step"]
        slope = row["Slope"]
        intercept = row["Intercept"]
        r2 = row["R2"]
        selected_indices = row["SelectedIndices"]
        lag_time = row["Lag"]
        first_point_y = original_df[
            (original_df["Unit"] == unit) & (original_df["Step"] == step)
            ].sort_values("Hours")["ln(Mean OD600)"].iloc[0]

        # Create a lag_point tuple for the intersection
        lag_point = (lag_time, first_point_y)

        plot_fit_with_selected_points(
            original_df,
            unit,
            step,
            slope,
            intercept,
            selected_indices,
            r2,
            lag_point,
            filename
        )


def plot_rolling_slopes(df: pd.DataFrame, window=6) -> None:
    """
    Calculate and plot rolling slopes for each Unit, Step.
    """
    rolling_slopes = []
    grouped = df.groupby(['Unit', 'Step'])

    for _, group in grouped:
        group = group.sort_values('Hours')
        x = group['Hours'].values
        y = group['ln(Mean OD600)'].values

        slopes = []
        for i in range(len(x) - window + 1):
            x_window = x[i:i + window].reshape(-1, 1)
            y_window = y[i:i + window]
            model = LinearRegression().fit(x_window, y_window)
            slopes.append(model.coef_[0])

        slopes = [np.nan] * (window - 1) + slopes
        group['Rolling Slope'] = slopes
        rolling_slopes.append(group)

    rolling_slopes_df = pd.concat(rolling_slopes)

    g = sns.relplot(
        x="Hours",
        y="Rolling Slope",
        col="Unit",
        col_wrap=2,
        hue="Salinity",
        data=rolling_slopes_df,
        col_order=COL_ORDER,
        height=10,
        markers=True,
        hue_order=("Optimal", "Stressed"),
        palette=("blue", "red"),
    )
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(f"Rolling Slopes with Window Size {window}")
    plt.savefig("plots/general/rolling_slopes.pdf", bbox_inches='tight', dpi=300)


def plot_robust_max_slopes_and_lag(
        slopes_and_r2,
        slope_title,
        lag_title,
        hue_order=("Optimal", "Stressed"),
        palette=("blue", "red"),
        filename="robust_slopes_and_lags"
):
    """
    Plots two lmplot figures:
      1) Slope vs. Step (Robust RLM)
      2) Lag vs. Step (Robust RLM)
    """
    # 1) PLOT SLOPE VS STEP
    slope_grid = sns.lmplot(
        x="Step",
        y="Slope",
        hue="Salinity",
        col="Unit",
        col_wrap=2,
        data=slopes_and_r2,
        height=10,
        col_order=COL_ORDER,
        hue_order=hue_order,
        palette=palette,
        robust=True,  # Seaborn's 'robust' is not the same as statsmodels RLM
        scatter_kws={"s": 50},
        line_kws={"lw": 2},
    )
    slope_grid.fig.subplots_adjust(top=0.95)
    slope_grid.fig.suptitle(slope_title)
    slope_grid.set_axis_labels("Step", "Slope (1/hour)")

    # 2) PLOT LAG VS STEP
    lag_grid = sns.lmplot(
        x="Step",
        y="Lag",
        hue="Salinity",
        col="Unit",
        col_wrap=2,
        data=slopes_and_r2,
        height=10,
        col_order=COL_ORDER,
        hue_order=hue_order,
        palette=palette,
        robust=True,  # again, just a hint to Seaborn
        scatter_kws={"s": 50},
        line_kws={"lw": 2},
    )
    lag_grid.fig.subplots_adjust(top=0.95)
    lag_grid.fig.suptitle(lag_title)
    lag_grid.set_axis_labels("Step", "Lag time (Hours)")

    robust_result = []
    
    # Annotate both slope and lag figures
    for fig, fig_suptitle in [(slope_grid, slope_title), (lag_grid, lag_title)]:
        # Each facet is an Axes object
        for ax in fig.axes.flat:
            title = ax.get_title()
            # e.g. "Unit = W3 - Replicate 1"
            unit_value = title.split(" = ")[-1].strip()

            # We'll collect all annotation text objects for this Axes here
            text_objs = []

            color_map = dict(zip(hue_order, palette))

            # We’ll place them near the median Step in data coordinates
            for sal in hue_order:
                subset = slopes_and_r2[(slopes_and_r2["Unit"] == unit_value) & (slopes_and_r2["Salinity"] == sal)]
                if subset.shape[0] < 2:
                    continue  # Not enough points for regression

                # Choose the formula depending on which figure we're annotating
                if fig == slope_grid:
                    model = smf.rlm("Slope ~ Step", data=subset).fit()
                else:
                    model = smf.rlm("Lag ~ Step", data=subset).fit()

                slope = model.params["Step"]
                intercept = model.params["Intercept"]
                p_value = model.pvalues["Step"]
                standard_error = model.bse["Step"]

                annot_text = (
                    f"{sal}: y = {slope:.2g}x + {intercept:.2g}\n"
                    f"p = {p_value:.2g}"
                )

                # We'll place the text near the median Step
                x_point = subset["Step"].median()
                # For consistency, compute the y_point from the regression line
                y_point = intercept + slope * x_point

                if fig == slope_grid:
                    robust_result.append({
                        'Unit': unit_value,
                        'Salinity': sal,
                        'Slope': slope,
                        'Intercept': intercept,
                        'p_value': p_value,
                        'x_data': subset['Step'].values,
                        'y_data': subset['Slope'].values if fig == slope_grid else subset['Lag'].values,
                        'standard_error': standard_error,
                    })

                # Create the annotation text object in data coords
                t = ax.annotate(
                    annot_text,
                    (x_point, y_point),
                    color=color_map[sal],
                )
                text_objs.append(t)

            # Now that all text objects are created for this Axes,
            # we call adjust_text to nudge them around if needed
            adjust_text(
                text_objs,
                ax=ax,
                force_text=0.8,
                force_points=0.2,
            )

        fig.fig.savefig(f"{filename}_{fig_suptitle}.pdf", bbox_inches='tight', dpi=300)
    
    return pd.DataFrame(robust_result)


def plot_robust_max_slopes_over_lag(
        slopes_and_r2,
        title,
        hue_order=("Optimal", "Stressed"),
        palette=("blue", "red"),
        filename="robust_max_slopes_over_lag"
):
    """
    Plots two lmplot figures:
      1) Slope vs. Step (Robust RLM)
      2) Lag vs. Step (Robust RLM)
    """
    # 1) PLOT SLOPE VS LAG
    slope_grid = sns.lmplot(
        x="Lag",
        y="Slope",
        hue="Salinity",
        col="Unit",
        col_wrap=2,
        data=slopes_and_r2,
        height=10,
        col_order=COL_ORDER,
        hue_order=hue_order,
        palette=palette,
        robust=True,  # Seaborn's 'robust' is not the same as statsmodels RLM
        scatter_kws={"s": 50},
        line_kws={"lw": 2},
    )
    slope_grid.fig.subplots_adjust(top=0.95)
    slope_grid.fig.suptitle(title)
    slope_grid.set_axis_labels("Lag (hour)", "Slope (1/hour)")

    # Each facet is an Axes object
    for ax in slope_grid.axes.flat:
        title = ax.get_title()
        # e.g. "Unit = W3 - Replicate 1"
        unit_value = title.split(" = ")[-1].strip()

        # We'll collect all annotation text objects for this Axes here
        text_objs = []

        color_map = dict(zip(hue_order, palette))

        # We’ll place them near the median Step in data coordinates
        for sal in hue_order:
            subset = slopes_and_r2[(slopes_and_r2["Unit"] == unit_value) & (slopes_and_r2["Salinity"] == sal)]
            if subset.shape[0] < 2:
                continue  # Not enough points for regression

            # Choose the formula depending on which figure we're annotating
            model = smf.rlm("Slope ~ Lag", data=subset).fit()

            slope = model.params["Lag"]
            intercept = model.params["Intercept"]
            p_value = model.pvalues["Lag"]

            annot_text = (
                f"{sal}: y = {slope:.2g}x + {intercept:.2g}\n"
                f"p = {p_value:.2g}"
            )

            # We'll place the text near the median
            x_point = subset["Lag"].median()
            # For consistency, compute the y_point from the regression line
            y_point = intercept + slope * x_point

            # Create the annotation text object in data coords
            t = ax.annotate(
                annot_text,
                (x_point, y_point),
                color=color_map[sal],
            )
            text_objs.append(t)

        # Now that all text objects are created for this Axes,
        # we call adjust_text to nudge them around if needed
        adjust_text(
            text_objs,
            ax=ax,
            force_text=0.8,
            force_points=0.2,
        )

    plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)


def plot_robust_generation_time_lag_ratio_over_step(
        slopes_and_r2,
        title,
        hue_order=("Optimal", "Stressed"),
        palette=("blue", "red"),
        filename="robust_generation_time_lag_ratio_over_step"
):
    """
    Plots two lmplot figures:
      1) Slope vs. Step (Robust RLM)
      2) Lag vs. Step (Robust RLM)
    """

    # Take the ratio of slope to lag
    slopes_and_r2["GT_Lag"] = (math.log(2) / slopes_and_r2["Slope"]) / slopes_and_r2["Lag"]

    # 1) PLOT SLOPE VS LAG
    slope_grid = sns.lmplot(
        x="Step",
        y="GT_Lag",
        hue="Salinity",
        col="Unit",
        col_wrap=2,
        data=slopes_and_r2,
        height=10,
        col_order=COL_ORDER,
        hue_order=hue_order,
        palette=palette,
        robust=True,  # Seaborn's 'robust' is not the same as statsmodels RLM
        scatter_kws={"s": 50},
        line_kws={"lw": 2},
    )
    slope_grid.fig.subplots_adjust(top=0.95)
    slope_grid.fig.suptitle(title)
    slope_grid.set_axis_labels("Step", "Generation Time/Lag")

    # Each facet is an Axes object
    for ax in slope_grid.axes.flat:
        title = ax.get_title()
        # e.g. "Unit = W3 - Replicate 1"
        unit_value = title.split(" = ")[-1].strip()

        # We'll collect all annotation text objects for this Axes here
        text_objs = []

        color_map = dict(zip(hue_order, palette))

        # We’ll place them near the median Step in data coordinates
        for sal in hue_order:
            subset = slopes_and_r2[(slopes_and_r2["Unit"] == unit_value) & (slopes_and_r2["Salinity"] == sal)]
            if subset.shape[0] < 2:
                continue  # Not enough points for regression

            # Choose the formula depending on which figure we're annotating
            model = smf.rlm("GT_Lag ~ Step", data=subset).fit()

            slope = model.params["Step"]
            intercept = model.params["Intercept"]
            p_value = model.pvalues["Step"]

            annot_text = (
                f"{sal}: y = {slope:.2g}x + {intercept:.2g}\n"
                f"p = {p_value:.2g}"
            )

            # We'll place the text near the median
            x_point = subset["Step"].median()
            # For consistency, compute the y_point from the regression line
            y_point = intercept + slope * x_point

            # Create the annotation text object in data coords
            t = ax.annotate(
                annot_text,
                (x_point, y_point),
                color=color_map[sal],
            )
            text_objs.append(t)

        # Now that all text objects are created for this Axes,
        # we call adjust_text to nudge them around if needed
        adjust_text(
            text_objs,
            ax=ax,
            force_text=0.8,
            force_points=0.2,
        )

    plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)


def plot_robust_last_step_od_over_lag(mean_df: pd.DataFrame, slopes_and_r2,
                                      title,
                                      hue_order=("Optimal", "Stressed"),
                                      palette=("blue", "red"),
                                      filename="robust_last_step_od_over_lag"):
    """
    Plots two lmplot figures:
      1) Slope vs. Step (Robust RLM)
      2) Lag vs. Step (Robust RLM)
    """

    # Get the OD of the last point in each the each step and the lag time of the next step
    slopes_and_r2["Last_OD_600"] = np.nan
    for (unit, step), group in mean_df.groupby(['Unit', 'Step']):
        group = group.sort_values('Hours')
        last_od = group['ln(Mean OD600)'].iloc[-1]
        slopes_and_r2.loc[(slopes_and_r2["Unit"] == unit) & (slopes_and_r2["Step"] == step+1), "Last_OD_600"] = last_od

    # Plot the last OD vs the lag time
    slope_grid = sns.lmplot(
        x="Lag",
        y="Last_OD_600",
        hue="Salinity",
        col="Unit",
        col_wrap=2,
        data=slopes_and_r2,
        height=10,
        col_order=COL_ORDER,
        hue_order=hue_order,
        palette=palette,
        robust=True,  # Seaborn's 'robust' is not the same as statsmodels RLM
        scatter_kws={"s": 50},
        line_kws={"lw": 2},
    )
    slope_grid.fig.subplots_adjust(top=0.95)
    slope_grid.fig.suptitle(title)
    slope_grid.set_axis_labels("Lag", "Previous step ln(OD600)")

    # Each facet is an Axes object
    for ax in slope_grid.axes.flat:
        title = ax.get_title()
        # e.g. "Unit = W3 - Replicate 1"
        unit_value = title.split(" = ")[-1].strip()

        # We'll collect all annotation text objects for this Axes here
        text_objs = []

        color_map = dict(zip(hue_order, palette))

        # We’ll place them near the median Step in data coordinates
        for sal in hue_order:
            subset = slopes_and_r2[(slopes_and_r2["Unit"] == unit_value) & (slopes_and_r2["Salinity"] == sal)]
            if subset.shape[0] < 2:
                continue  # Not enough points for regression

            # Choose the formula depending on which figure we're annotating
            model = smf.rlm("Last_OD_600 ~ Lag", data=subset).fit()

            slope = model.params["Lag"]
            intercept = model.params["Intercept"]
            p_value = model.pvalues["Lag"]

            annot_text = (
                f"{sal}: y = {slope:.2g}x + {intercept:.2g}\n"
                f"p = {p_value:.2g}"
            )

            # We'll place the text near the median
            x_point = subset["Lag"].median()
            # For consistency, compute the y_point from the regression line
            y_point = intercept + slope * x_point

            # Create the annotation text object in data coords
            t = ax.annotate(
                annot_text,
                (x_point, y_point),
                color=color_map[sal],
            )
            text_objs.append(t)

        # Now that all text objects are created for this Axes,
        # we call adjust_text to nudge them around if needed
        adjust_text(
            text_objs,
            ax=ax,
            force_text=0.8,
            force_points=0.2,
        )

    plt.savefig(f"{filename}.pdf", bbox_inches='tight', dpi=300)


def predict_threshold_time(equations: dict, df: pl.DataFrame, predict_od: float) -> None:
    """
    Predict the time at which OD will reach predict_od for each pioreactor,
    based on the time shift from a previous step.
    """
    pioreactors = equations.keys()
    for pioreactor in pioreactors:
        pio_df = df.filter(pl.col("Unit") == pioreactor).sort("Time", descending=False).collect()

        steps = pio_df.get_column("Step").sort().unique().to_list()
        if len(steps) < 3:
            continue  # not enough data

        current_od = pio_df.filter(pl.col("Step") == steps[-1]).get_column("OD600")[-1]
        last_step = pio_df.filter(pl.col("Step") == steps[-3])

        thresh_time = (
            last_step
            .with_columns((pl.col("OD600") - predict_od).abs().alias("distance"))
            .filter(pl.col("distance") == pl.col("distance").min())
            .get_column("Time")[0]
        )

        time_in_last_step = (
            last_step
            .with_columns((pl.col("OD600") - current_od).abs().alias("distance"))
            .filter(pl.col("distance") == pl.col("distance").min())
            .get_column("Time")[0]
        )

        predicted_time_at_threshold = (
                pio_df.filter(pl.col("Step") == steps[-1]).get_column("Time")[-1]
                + (thresh_time - time_in_last_step)
        )
        print(f"Predicted time to OD={predict_od:.2f} for {pioreactor}: {predicted_time_at_threshold}")


def do_grofit(df: pl.DataFrame) -> None:
    
    # Within each step, subtract hours from the first hour to get relative time
    df = df.with_columns((pl.col("Hours") - pl.col("Hours").min()).over("Unit", "Step").alias("Hours"))
    
    # Create the metadata columns:
    # - experiment_id: here taken as "Unit"
    # - additional_info: here taken as "Step"
    # - substrate_conc: here taken as "Salinity"
    df = df.with_columns([
        pl.col("Unit").alias("experiment_id"),
        pl.col("Step").alias("additional_info"),
        pl.col("Salinity").alias("substrate_conc")
    ])

    # Data: 3+n columns, containing three coulumns of additional information and n columns of growth data corresponding to time
    # Time: Numeric matrix, consisting of n columns for the time points and m rows for the different experiments.
    # Call grofit one row at a time because we dont have matching time points
    fit_results = []
    for _, data in df.group_by("experiment_id", "additional_info", "substrate_conc"):
        df_wide = data.sort("Hours", descending=False).pivot(
            index=["experiment_id", "additional_info", "substrate_conc"],
            on="Hours",
            values="Mean OD600",
        ).sort("experiment_id", "additional_info", "substrate_conc").to_numpy()

        # Here we group and collect the sorted Hours as a list for each experiment.
        time_series = data.with_columns(time_list = pl.col("Hours").sort())["time_list"].to_numpy()
                
        # Write to temporary file
        np.savetxt("data.csv", df_wide, delimiter=",", fmt='%s')
        np.savetxt("time.csv", [time_series], delimiter=",")

        # Execute grofit.R command
        os.system(f"/Users/GeorgesKanaan/micromamba/envs/jupyter/bin/Rscript grofit.R")
        
        # Read summary.csv into fit_results
        if os.path.exists("summary.csv"):
            fit_results.append(pd.read_csv("summary.csv"))
        
    # Delete the files
    os.remove("data.csv")
    os.remove("time.csv")
    os.remove("summary.csv")

    # Concatenate the fit_results
    fit_results = pd.concat(fit_results)
    fit_results.to_csv("grofit_results.csv", index=False)


def process_grofit_results():
    """
    Get the grofit results and shape into a DataFrame.
    """
    # Load the grofit results
    grofit_results = pd.read_csv("grofit_results.csv")
    
    # Select the columns TestId	AddId	concentration	mu.spline	lambda.spline	A.spline	integral.spline
    # Remove duplicates
    grofit_results = grofit_results[["TestId", "AddId", "concentration", "mu.spline", "lambda.spline", "A.spline", "integral.spline"]].drop_duplicates()
    
    # Rename to Unit, Step, Salinity, Slope, Lag, A, Integral
    grofit_results = grofit_results.rename(columns={
        "TestId": "Unit",
        "AddId": "Step",
        "concentration": "Salinity",
        "mu.spline": "Slope",
        "lambda.spline": "Lag",
        "A.spline": "A",
        "integral.spline": "Integral"
    })
    
    return grofit_results


def do_pyphe(df: pl.DataFrame) -> pd.DataFrame:
    df = df.with_columns((pl.col("Hours") - pl.col("Hours").min()).over("Unit", "Step").alias("Hours"))
    
    results = []
    
    for names, data in df.group_by("Unit", "Step", "Salinity"):
        # Get the data for pyphe then write to CSV
        pyphe_df = data.select("Hours", "Mean OD600").rename({"Mean OD600": names[0] + "_Step" + str(names[1]) + "_" + names[2]}).sort("Hours", descending=False)
        csv_name = f"pyphe_{names[0]}_Step{names[1]}_{names[2]}.csv".replace(" ", "_")
        pyphe_df.write_csv(csv_name)

        # Call pyphe then delete CSV
        os.system(f"/Users/GeorgesKanaan/micromamba/envs/jupyter/bin/python /Users/GeorgesKanaan/micromamba/envs/jupyter/bin/pyphe-growthcurves --input {csv_name} --fitrange 20 --lag-method rel --plots --out pyphe_result  --t0-fitrange 10")
        os.remove(csv_name)
        
        # Load the results, add the group information
        result_csv = pd.read_csv(f"pyphe_result/{csv_name[:-4]}_results.csv", header=1, index_col=0).T
        result_csv["Unit"] = names[0]
        result_csv["Step"] = names[1]
        result_csv["Salinity"] = names[2]
        results.append(result_csv)

    # Concatenate the results
    pyphe_results = pd.concat(results)
    pyphe_results = pyphe_results.rename(columns={"lag": "Lag", "max_slope": "Slope"})

    # Set the dtypes to float for anything not a string
    string_columns = ["Unit", "Step", "Salinity", "warning_bad_fit", "warning_negative_slope"]
    for col in pyphe_results.columns:
        if col not in string_columns:
            pyphe_results[col] = pyphe_results[col].astype(float)

    return pyphe_results


###############################################################################
# Main Orchestration
###############################################################################
def main():
    """
    1. Load and merge data from Exp1 and ~/Downloads.
    2. Transform & filter the data.
    3. Plot rolling average curves.
    4. Classify steps and plot them.
    5. Filter weird steps, bin data, and plot binned curves.
    6. Fit slopes; plot rolling slopes, slopes, and lag times.
    7. Optionally run GrowthRates or threshold predictions.
    """
    # Make all needed plot directories
    os.makedirs("plots/fits/alt", exist_ok=True)
    os.makedirs("plots/fits/bgme", exist_ok=True)
    os.makedirs("plots/general", exist_ok=True)

    # Step 1: Load Data
    exp1_csv = load_data_for_exp1()
    latest_csv = find_latest_download_csv()

    # Step 2: Merge & Transform
    combined_lf = merge_experiments(exp1_csv, latest_csv)
    df = transform_data(combined_lf)

    # Step 3: Rolling average plot
    plot_10min_bins_whole_experiment(df)

    # Step 4: Classify steps, filter and plot
    df = classify_steps(df)
    df = filter_weird_steps(df)
    plot_classified_steps(df)
    
    # Bin data
    mean_df = bin_data(df, log=True)
    plot_binned_data(mean_df)

    # # Grofit analysis
    # do_grofit(mean_df)
    # slopes_and_lag = process_grofit_results()
    # plot_robust_max_slopes_and_lag(slopes_and_lag, "Grofit mu", "Grofit lambda", filename="plots/grofit_max_slopes_and_lags")
    
    # # Pyphe
    # slopes_and_lag = do_pyphe(mean_df)
    # plot_robust_max_slopes_and_lag(slopes_and_lag, "Pyphe mu", "Pyphe lambda", filename="plots/pyphe_max_slopes_and_lags")
    
    # Step 6: Filter weird steps, calculate slopes, then plot them (rolling + robust)
    slopes_and_r2 = calculate_step_slopes_and_lag(mean_df, window_length=10)
    slopes_and_r2 = slopes_and_r2[slopes_and_r2["Significant"]]  # Only keep significant

    plot_rolling_slopes(mean_df.to_pandas(), window=10)
    plot_selected_fits(slopes_and_r2, mean_df.to_pandas(), filename="plots/fits/bgme/fit")
    robust_result = plot_robust_max_slopes_and_lag(slopes_and_r2, "Max slopes", "Lag (Intercept)", filename="plots/robust_max_slopes_and_lags")

    a = compute_lines_are_different_test(robust_result)
    a.to_csv("plots/robust_lines_are_different.csv", index=False)

    # plot_robust_max_slopes_over_lag(slopes_and_r2, "Max slopes over Lag", filename="plots/robust_max_slopes_over_lag")
    # plot_robust_last_step_od_over_lag(mean_df.to_pandas(), slopes_and_r2, "Last step ln(OD600) over lag", filename="plots/robust_last_step_od_over_lag")

    # Calculate steps and slopes alternatives
    # slopes_and_r2 = calculate_alt_step_slopes_and_lag(mean_df, window_length=10)
    # slopes_and_r2 = slopes_and_r2[slopes_and_r2["Significant"]]  # Only keep significant
    
    # plot_selected_fits(slopes_and_r2, mean_df.to_pandas(), filename="plots/fits/alt/fit")
    # plot_robust_max_slopes_and_lag(slopes_and_r2, "Avg Slopes (alt)", "Lag (alt)", filename="plots/robust_avg_slopes_and_lags_alt")
    # plot_robust_max_slopes_over_lag(slopes_and_r2, "Avg Slopes over Lag (alt)", filename="plots/robust_avg_slopes_over_lag_alt")
    # plot_robust_generation_time_lag_ratio_over_step(slopes_and_r2, "Generation Time/Lag over Step (alt)", filename="plots/robust_generation_time_lag_ratio_over_step_alt")
    # plot_robust_last_step_od_over_lag(mean_df.to_pandas(), slopes_and_r2, "Last step ln(OD600) over lag (alt)", filename="plots/robust_last_step_od_over_lag_alt")

    # Step 7 (Optional): Run GrowthRates or predict threshold
    # do_grme(mean_df)
    # predict_threshold_time(EQUATIONS, df, predict_od=0.30)


###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    main()

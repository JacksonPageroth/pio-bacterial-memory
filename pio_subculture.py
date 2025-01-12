import numpy as np
import polars as pl
import subprocess
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import zipfile
import pandas as pd
from sklearn.linear_model import LinearRegression


sns.set_theme(context="poster", style="white")


def unzip_and_rename_csv(zip_file_path, extract_to_folder=None, renamed_csv="latest_od_data.csv"):
    """
    Unzips a file containing one CSV and renames it to a specified name.

    Args:
        zip_file_path (str): Path to the zip file.
        extract_to_folder (str): Folder where files should be extracted.
                                 Defaults to the same directory as the zip file.
        renamed_csv (str): Desired name for the extracted CSV file.

    Returns:
        str: Path to the renamed CSV file.
    """
    if extract_to_folder is None:
        extract_to_folder = os.path.dirname(zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all files
        zip_ref.extractall(extract_to_folder)

        # Identify the extracted CSV file
        csv_file = None
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith('.csv'):
                csv_file = os.path.join(extract_to_folder, file_name)
                break

    # Check if a CSV file was found
    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the ZIP archive.")

    # Rename the CSV file
    renamed_csv_path = os.path.join(extract_to_folder, renamed_csv)
    os.rename(csv_file, renamed_csv_path)

    return renamed_csv_path


# Exp 1
exp1 = unzip_and_rename_csv('./od_logs/exp1.zip', renamed_csv="exp1.csv")


# Get the list of CSV files in the Downloads folder
csv_files = glob.glob(os.path.expanduser('~/Downloads/*.csv'))

if len(csv_files) == 0:
    latest_csv = "./od_logs/latest_od_data.csv" #unzip_and_rename_csv('./od_logs/latest_od_data.zip')
else:
    # Find the most recent file based on modification time
    latest_csv = max(csv_files, key=os.path.getmtime)


# Merge the two experiments
exp1 = pl.scan_csv(exp1).select("od_reading", "timestamp_localtime", "pioreactor_unit").rename({"od_reading": "OD600", "timestamp_localtime": "Time", "pioreactor_unit": "Unit"})
df = pl.scan_csv(latest_csv).select("od_reading", "timestamp_localtime", "pioreactor_unit").rename({"od_reading": "OD600", "timestamp_localtime": "Time", "pioreactor_unit": "Unit"})
df = pl.concat([df, exp1])

# Dictionary to map 'Pioreactor name' to new values
name_dict = {
    #'worker 1': 'W1  - Replicate 3',
    'worker2': 'W2 - 35 ppt Control',
    'worker3': 'W3 - Replicate 1',
    'worker4': 'W4 - Replicate 2',
    'worker5': 'W5 - 55 ppt Control'
}

optimal_controls = ["W2 - 35 ppt Control"]
stressed_controls = ["W5 - 55 ppt Control"]
controls = ["W5 - 55 ppt Control", "W2 - 35 ppt Control"]
replicates = ["W3 - Replicate 1", "W4 - Replicate 2"]

# Define the corresponding transformations for each condition
equations = {
    #'W1  - Replicate 3': (0, 0),  # Not defined
    "W2 - 35 ppt Control": (1.348628103, 0.077),
    "W3 - Replicate 1": (2.073660771, .077),
    "W4 - Replicate 2": (2.204291876, .077),
    "W5 - 55 ppt Control": (1.169484467, .077),
}

threshold_od = 0.24  # Point after which we subculture

# Replace values in 'Pioreactor name' column using the dictionary
df = df.with_columns(pl.col("Unit").replace_strict(name_dict))

# Apply the equation
a = pl.col("Unit").replace_strict(equations).list.first()
b = pl.col("Unit").replace_strict(equations).list.last()
df = df.with_columns((pl.col("OD600") * a + b))

# Convert 'timestamp_localtime' to datetime format
df = df.with_columns(pl.col("Time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))

# Filter subculture artifacts
df = df.filter(pl.col("Time") >= pl.col("Time").min().dt.offset_by("5m"))

# Filter out artifact
df= df.with_columns((pl.col("Time").sub(pl.col("Time").min()).dt.total_seconds()/3600).alias("Hours"))
df = df.filter((pl.col("Unit").eq("W5 - 55 ppt Control") & pl.col("Hours").ge(104.2) & pl.col("Hours").le(104.4)).not_())

df = df.sort("Unit", "Hours")


# Take a rolling average in each step
plot_df = df.with_columns(pl.col("Time").dt.truncate("10m").alias("Time"))
plot_df = plot_df.group_by("Unit", "Time").agg(pl.col("OD600").mean().alias("Mean OD600"))
plot_df = plot_df.with_columns((pl.col("Time").sub(pl.col("Time").min()).dt.total_seconds()/3600).alias("Hours"))

# Plotting the growth curves
hue_order = ["W2 - 35 ppt Control", "W5 - 55 ppt Control", "W3 - Replicate 1", "W4 - Replicate 2"]

f, ax = plt.subplots(figsize=(40, 7))
sns.lineplot(data=plot_df.collect(streaming=True).to_pandas(), x="Time", y="Mean OD600", hue="Unit", hue_order=hue_order, markersize=8,
             sizes=(1, 8), ax=ax, palette=["blue", "red", "green", "mediumseagreen"])

plt.axhline(y=threshold_od, color='black', linestyle='--')
# Set major ticks to every 6 hours
ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.set_xlabel("Time")

# Create a secondary x-axis for date display
secax = ax.secondary_xaxis(-0.1)
secax.xaxis.set_major_locator(mdates.DayLocator())
secax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
secax.set_xlabel("Date")
ax.set_title('Pioreactor Transformed OD Curves')
plt.grid(True)
plt.show()


def find_steps(class_df, shift):
    # Condition for incrementing 'step'
    class_df = class_df.sort("Unit", "Hours")
    step_cond = (pl.col("OD600").shift(shift) - pl.col("OD600") > 0.03)

    # Use cumulative sum over the 'step_condition' to increment each time it's True
    class_df = class_df.with_columns(pl.when(step_cond).then(1).otherwise(0).cum_sum().add(1).over(pl.col("Unit")).alias("Step"))

    # Add condition to the ldf
    class_df = class_df.with_columns(pl.when((pl.col("Step") % 2 == 1) & (pl.col("Unit").is_in(replicates)))
                         .then(pl.lit("Optimal"))
                         .otherwise(pl.when((pl.col("Step") % 2 == 0) & (pl.col("Unit").is_in(replicates)))
                                    .then(pl.lit("Stressed"))
                                    .otherwise(pl.when(pl.col("Unit").is_in(stressed_controls))
                                               .then(pl.lit("Stressed"))
                                               .otherwise(pl.when(pl.col("Unit").is_in(optimal_controls))
                                                          .then(pl.lit("Optimal")))))
                         .alias("Condition"))

    return class_df

# Find steps
df = find_steps(df, 5)

# Discard the first and last 5 points in every step (because of subculture spike)
filter_points = 20
df = df.with_columns(pl.cum_count("OD600").over("Unit", "Step").alias("index")).filter(pl.col("index") > filter_points)
df = df.filter(pl.col("index") < pl.col("index").max() - filter_points).drop("index")

# Redo step classification
df = df.drop("Step", "Condition")
df = find_steps(df, 1)

# Rename condition to salinity
df = df.rename({"Condition": "Salinity"})

# Plotting the growth curves using the new 'Transformed OD' column
col_order = ["W3 - Replicate 1", "W4 - Replicate 2", "W2 - 35 ppt Control", "W5 - 55 ppt Control"]
g = sns.relplot(x="Hours", y="OD600", hue="Salinity", style="Step", col="Unit", col_wrap=2, col_order=col_order, height=10,
             markersize=8, data=df.collect(streaming=True).to_pandas(), kind="line", hue_order=["Optimal", "Stressed"], palette=["blue", "red"])

for ax in g.axes.flatten():
    ax.axhline(y=threshold_od, color='black', linestyle='--')
g.fig.suptitle('Pioreactor OD Curves')
plt.show()


# Filter out weird steps
weird_steps = {"W2 - 35 ppt Control": [16, 18, 21],
               "W3 - Replicate 1": [10, 16],
               "W4 - Replicate 2": [11, 16],
               "W5 - 55 ppt Control": [3, 10]}
df = df.filter(*[~((pl.col("Unit") == unit) & pl.col("Step").is_in(steps)) for unit, steps in weird_steps.items()])

# Average data in 10 minute bins
mean_df = df.with_columns(pl.col("Time").dt.truncate("10m").alias("Time"))
mean_df = mean_df.group_by("Unit", "Time", "Salinity", "Step").agg(pl.col("OD600").mean().alias("Mean OD600"))
mean_df = mean_df.with_columns((pl.col("Time").sub(pl.col("Time").min()).dt.total_seconds()/3600).alias("Hours"))
mean_df = mean_df.with_columns(pl.col("Mean OD600").log().alias("ln(Mean OD600)")).collect()

# Plotting the growth curves using the new 'Transformed OD' column
col_order = ["W3 - Replicate 1", "W4 - Replicate 2", "W2 - 35 ppt Control", "W5 - 55 ppt Control"]
g = sns.relplot(x="Hours", y="Mean OD600", hue="Salinity", style="Step", col="Unit", col_wrap=2, col_order=col_order, height=10, markers=True,
             markersize=8, data=mean_df.to_pandas(), kind="line", hue_order=["Optimal", "Stressed"], palette=["blue", "red"])

for ax in g.axes.flatten():
    ax.axhline(y=threshold_od, color='black', linestyle='--')
g.fig.suptitle('Pioreactor binned OD Curves')
plt.show()

# Plotting the growth curves using the new 'Transformed OD' column
col_order = ["W3 - Replicate 1", "W4 - Replicate 2", "W2 - 35 ppt Control", "W5 - 55 ppt Control"]
g = sns.relplot(x="Hours", y="ln(Mean OD600)", hue="Salinity", style="Step", col="Unit", col_wrap=2, col_order=col_order, height=10, markers=True,
             markersize=8, data=mean_df.to_pandas(), kind="line", hue_order=["Optimal", "Stressed"], palette=["blue", "red"])

for ax in g.axes.flatten():
    ax.axhline(y=np.log(threshold_od), color='black', linestyle='--')
g.fig.suptitle('Pioreactor binned ln(OD) Curves')
plt.show()


def plot_fit_with_selected_points(ldf, unit, step, slope, intercept, selected_indices, significant, lag_point):
    """
    Plot the original data with the line fit and highlight the points used for slope calculation.
    Additionally, plot a line fitted to the first 10 points, extend it to the intersection with the main fitted line,
    and annotate the intersection point.

    Parameters:
        ldf (pd.DataFrame): Input DataFrame with columns 'Unit', 'Step', 'Hours', and 'ln(Mean OD600)'.
        unit (str): The unit to plot.
        step (str): The step to plot.
        slope (float): Slope of the fitted line.
        intercept (float): Intercept of the fitted line.
        selected_indices (list): Indices of points used for the slope calculation.
        significant (float): R^2 value of the fitted line.

    Returns:
        None: Displays the plot.
    """
    # Filter the data for the specified Unit and Step
    group = ldf[(ldf['Unit'] == unit) & (ldf['Step'] == step)].sort_values('Hours')
    x = group['Hours'].values
    y = group['ln(Mean OD600)'].values
    intersection_x, intersection_y = lag_point
    intersection_x += x[0]  # Lag was given in absolute amount, we want relative to start

    # Get the coordinates for point along the lag line
    x_lag_fit = np.linspace(x.min(), x.max(), 100)
    y_lag_fit = np.full_like(x_lag_fit, intersection_y)

    # Get the X for the exp fit line, and corresponding Ys from model
    x_fit = np.linspace(x.min(), x.max(), 100)
    x_fit = x_fit[x_fit >= intersection_x - 1]
    y_fit = slope * x_fit + intercept

    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Original Data", color="gray", alpha=0.7)

    # Highlight selected points
    selected_points = group.iloc[selected_indices]
    plt.scatter(selected_points['Hours'], selected_points['ln(Mean OD600)'],
                label="Selected Points", color="orange", edgecolor="black", zorder=5)

    # Plot the exp fit line and lag line
    plt.plot(x_fit, y_fit, label=f"Main Fitted Line (Slope={slope:.2f})", color="blue", linewidth=2)
    plt.plot(x_lag_fit, y_lag_fit, label="Lag fit", color="green", linestyle="--", linewidth=2)

    # Mark the intersection point
    plt.scatter([intersection_x], [intersection_y], color="red", zorder=10, label=f"Intersection (x={intersection_x:.2f})")
    plt.axvline(x=intersection_x, color="red", linestyle=":", linewidth=1, alpha=0.7)

    # Add labels, legend, and title
    plt.xlabel("Hours")
    plt.ylabel("ln(Mean OD600)")

    plt.title(f"Unit: {unit}, Step: {step}, R2 {significant:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()



def calculate_step_slopes_and_r2(df, window_length=6):
    """
    Calculate the slope and R^2 for each step in the given Polars DataFrame, considering only points where the slope
    over a rolling window of 6 is within 95% of the maximum slope obtained by that rolling window.

    Parameters:
        df (pl.DataFrame): Input Polars DataFrame with columns 'Unit', 'Step', 'Hours', and 'ln(Mean OD600)'.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns 'Unit', 'Step', 'Slope', and 'R2'.
    """
    # Convert Polars DataFrame to Pandas for processing
    df = df.to_pandas()

    results = []

    # Group by Unit and Step
    grouped = df.groupby(['Unit', 'Step'])

    for (unit, step), group in grouped:
        group = group.sort_values('Hours')
        x = group['Hours'].values
        y = group['ln(Mean OD600)'].values

        # Calculate rolling slopes over a window
        rolling_slopes = []
        for i in range(len(x) - (window_length - 1)):
            x_window = x[i:i + window_length].reshape(-1, 1)
            y_window = y[i:i + window_length]
            model = LinearRegression().fit(x_window, y_window)
            rolling_slopes.append(model.coef_[0])

        # Pad the rolling slopes with NaN to match the original length
        group['Rolling Slope'] = rolling_slopes + [np.nan] * (window_length - 1)
        group.reset_index(drop=True, inplace=True)

        # Filter points where the rolling slope is within 95% of the max rolling slope
        max_slope = np.nanmax(rolling_slopes)
        matching_indices = group.index[group['Rolling Slope'] >= 0.95 * max_slope].tolist()

        # Include the 5 next rows for each matching point
        expanded_indices = set()
        for idx in matching_indices:
            expanded_indices.update(range(idx, min(idx + window_length, len(x))))

        slope_points = group.loc[list(expanded_indices)].sort_values("Hours")

        if len(slope_points) > 1:  # Ensure there are enough points for regression
            x_filtered = slope_points['Hours'].values.reshape(-1, 1)
            y_filtered = slope_points['ln(Mean OD600)'].values
            model = LinearRegression().fit(x_filtered, y_filtered)
            slope = model.coef_[0]
            r2 = model.score(x_filtered, y_filtered)

            # Get the lag time
            first_point_y = group.iloc[0]['ln(Mean OD600)']
            lag_time = ((group.iloc[0]['ln(Mean OD600)'] - model.intercept_) / slope) - group.iloc[0]['Hours']
            lag_point = (lag_time, first_point_y)

            # Plot the data with fit
            plot_fit_with_selected_points(df, unit, step, slope, model.intercept_, slope_points.index.tolist(), r2, lag_point)

            results.append({'Unit': unit, 'Step': step, 'Slope': slope, 'R2': r2, 'Points': len(group), 'Salinity': group['Salinity'].values[0], 'Significant': r2 > 0.95, "Lag": lag_time})

    # Convert results to DataFrame
    return pd.DataFrame(results)


def plot_rolling_slopes(df, window=6):
    """
    Plot rolling slopes for the given DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns 'Unit', 'Step', 'Hours', and 'ln(Mean OD600)'.
        window (int): Rolling window size for slope calculation.
    """
    # Calculate rolling slopes for each Unit and Step
    rolling_slopes = []
    grouped = df.groupby(['Unit', 'Step'])

    for (unit, step), group in grouped:
        group = group.sort_values('Hours')
        x = group['Hours'].values
        y = group['ln(Mean OD600)'].values

        slopes = []
        for i in range(len(x) - window + 1):
            x_window = x[i:i + window].reshape(-1, 1)
            y_window = y[i:i + window]
            model = LinearRegression().fit(x_window, y_window)
            slopes.append(model.coef_[0])

        slopes = [np.nan] * (window - 1) + slopes  # Pad to match the length
        group['Rolling Slope'] = slopes
        rolling_slopes.append(group)

    rolling_slopes_df = pd.concat(rolling_slopes)

    # Plot
    g = sns.relplot(x="Hours", y="Rolling Slope", col="Unit", col_wrap=2, hue="Salinity", data=rolling_slopes_df, col_order=col_order, height=10, markers=True)
    g.fig.suptitle("Rolling Slopes")
    plt.show()


# Individual fits
window = 10
slopes_and_r2 = calculate_step_slopes_and_r2(mean_df, window_length=window)

# Rolling slope
plot_rolling_slopes(mean_df.to_pandas(), window=window)

# Plot
g = sns.relplot(x="Step", y="Slope", hue="Salinity", col="Unit", col_wrap=2, style="Significant", col_order=col_order, height=10, markers=True,
            data=slopes_and_r2, kind="line", hue_order=["Optimal", "Stressed"], palette=["blue", "red"])
g.fig.suptitle("Slopes")
plt.show()

g = sns.relplot(x="Step", y="Lag", hue="Salinity", col="Unit", col_wrap=2, style="Significant", col_order=col_order, height=10, markers=True,
            data=slopes_and_r2, kind="line", hue_order=["Optimal", "Stressed"], palette=["blue", "red"])
plt.ylabel("Lag time (hours)")
g.fig.suptitle("Lag times")
plt.show()


def do_grme():
    # Paths and configurations
    GROWTH_RATES_EXEC_PATH = '/Applications/GrowthRates_6.2.1/GrowthRates'
    OUTPUT_DIR = './grme/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate worker-specific data and process GrowthRates
    for worker in mean_df.get_column("Unit").unique().to_list():
        worker_file = os.path.join(OUTPUT_DIR, f"{worker}.txt")

        # Get the minute within each step
        worker_df = mean_df.filter(pl.col("Unit").eq(worker)).with_columns((pl.col("Hours") * 6).round().mul(10).cast(pl.Int64).alias("Min"))
        worker_df = worker_df.with_columns(pl.col("Min").sub(pl.col("Min").min()).over("Step"))

        # Pivot data for this step
        worker_df = worker_df.pivot(index="Min", on="Step", values="ln(Mean OD600)").sort("Min")
        worker_df = worker_df.with_columns(pl.col("*").exclude("Min").exp())

        # Aggregate the data for each minute, and remove nulls
        worker_df = worker_df.group_by("Min").agg(pl.col("*").exclude("Min").max())
        worker_df = worker_df.with_columns(pl.col("*").exclude("Min").forward_fill())

        # Rename the column to append the worker name as a suffix
        worker_df = worker_df.rename({col: f"{worker}_{col}" for col in worker_df.columns if col != "Min"})
        worker_df.write_csv(worker_file, separator='\t')

        # Run GrowthRates
        try:
            command = [GROWTH_RATES_EXEC_PATH, "-i", os.path.relpath(worker_file), "-b", "0.077"]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input="1\nY\n")
            if process.returncode == 0:
                print(f"Successfully processed {worker}:\n{stdout}")
            else:
                print(f"Error processing {worker}:\n{stderr}")
        except Exception as e:
            print(f"Error running GrowthRates for {worker}: {e}")

    # Visualization
    fig, ax = plt.subplots(figsize=(20, 12))

    palette = {'W2 - 35 ppt Control': "blue", 'W5 - 55 ppt Control': "red", 'W3 - Replicate 1': "green", 'W4 - Replicate 2': "mediumseagreen"}
    replicates = ['W3 - Replicate 1', 'W4 - Replicate 2']
    controls = ['W2 - 35 ppt Control', 'W5 - 55 ppt Control']

    for filepath in glob.glob(os.path.expanduser('grme/*.summary')):
        temp_df = pd.read_csv(filepath, sep='\t', header=2, usecols=['Well', 'min', 'hours', 'R', 'Max OD', 'lag time (minutes)'])
        temp_df.rename(columns={'hours': 'Growth Rate (1/hours)'}, inplace=True)
        unit_name = os.path.basename(filepath).split('.')[0]

        sns.lineplot(data=temp_df, x=temp_df.index, y='Growth Rate (1/hours)', ax=ax, color=palette[unit_name], label=unit_name)

        # Add best-fit lines
        x = temp_df.index
        y = temp_df['Growth Rate (1/hours)']
        if unit_name in replicates:
            if len(x[::2]) > 1:
                m, b = np.polyfit(x[::2], y[::2], 1)
                ax.plot(x, m * x + b, color=palette[unit_name], linestyle='dashed', label=f'{unit_name} 35 fit')

            if len(x[1::2]) > 1:
                m, b = np.polyfit(x[1::2], y[1::2], 1)
                ax.plot(x, m * x + b, color=palette[unit_name], linestyle='dashed', label=f'{unit_name} 55 fit')

        elif unit_name in controls and len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m * x + b, color=palette[unit_name], linestyle='dashed', label=f'{unit_name} fit')

    plt.xlabel('Step Number')
    plt.ylabel('Growth Rate (1/hours)')
    plt.title('Growth rates')
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()


def predict_threshold_time(equations: dict, df: pl.DataFrame, predict_od: int):
    pioreactors = equations.keys()
    for pioreactor in pioreactors:
        pio_df = df.filter(pl.col("Unit").eq(pioreactor)).sort("Time", descending=False).collect()

        steps = pio_df.get_column("Step").sort().unique().to_list()
        current_od = pio_df.filter(pl.col("Step").eq(steps[-1])).get_column("OD600")[-1]

        last_step = pio_df.filter(pl.col("Step").eq(steps[-3]))
        thresh_time = (last_step.with_columns((pl.col("OD600") - predict_od).abs().alias("distance"))
                            .filter(pl.col("distance") == pl.col("distance").min())
                            .get_column("Time")[0])

        time_in_last_step = (last_step.with_columns((pl.col("OD600") - current_od).abs().alias("distance"))
                            .filter(pl.col("distance") == pl.col("distance").min())
                            .get_column("Time")[0])

        #Add time to threshold to latest timestamp of current step
        predicted_time_at_threshold = pio_df.filter(pl.col("Step").eq(steps[-1])).get_column("Time")[-1] + (thresh_time - time_in_last_step)

        print(f"Predicted time to {predict_od:.2f} for {pioreactor}: {predicted_time_at_threshold}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Get the list of CSV files in the Downloads folder
csv_files = glob.glob(os.path.expanduser('~/Downloads/*.csv'))

if len(csv_files) == 0:
    csv_files = glob.glob('./od_logs/*.csv')

# Find the most recent file based on modification time
latest_csv = max(csv_files, key=os.path.getmtime)

# Load the CSV file
df = (
    pd.read_csv(latest_csv)
    .rename(columns={"od_reading": "OD600", "timestamp_localtime": "Time", "pioreactor_unit": "Unit"})
    .drop(columns=["experiment", "timestamp", "angle", "channel"])
    .assign(Time=lambda d: pd.to_datetime(d["Time"]))  # Convert 'Time' to datetime
)

# Filter the DataFrame to only include data after 100 measurements and OD600 values less than 1
df = df[df.index >= 100]
df = df[df['OD600'] < 1]

# Pivot the DataFrame to have Units as columns and Min as the index
df = df.pivot(index='Time', columns='Unit', values='OD600').resample('min').mean()

# Define the corresponding transformations for each condition
equations = {
    "worker2": (1.348628103, 0.077),
    "worker3": (2.073660771, 0.077),
    "worker4": (2.204291876, 0.077),
    "worker5": (1.169484467, 0.077)
}

# Apply the transformations
for unit, (slope, intercept) in equations.items():
    df[unit] = df[unit] * slope + intercept

# Plot the DataFrame
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, dashes=False)
plt.xlabel('Time')
plt.ylabel('OD600')
plt.title('OD600 Readings for Each Unit')
plt.legend(title='Unit')
plt.grid(True)
plt.tight_layout()
plt.show()
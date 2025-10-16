import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read and parse the CSV file
data = []

with open('cellpose_results.csv', 'r') as f:
    lines = f.readlines()
    
# Skip the header
i = 1
while i < len(lines):
    parts = lines[i].strip().split(',')
    
    # Check if this looks like a folder_name line
    if len(parts) >= 2 and 'S' in parts[0] and '_' in parts[0]:
        folder_name = parts[0]
        image_name = parts[1]
        
        # Next line should have cell count
        if i + 1 < len(lines):
            cell_count_line = lines[i + 1].strip().split(',')
            cell_count = cell_count_line[0]
            
            # Next line should have median cell area
            if i + 2 < len(lines):
                median_area_line = lines[i + 2].strip()
                median_cell_area = median_area_line
                
                data.append({
                    'folder_name': folder_name,
                    'image_name': image_name,
                    'cell_count': cell_count,
                    'median_cell_area': median_cell_area
                })
                
                i += 3
            else:
                i += 1
        else:
            i += 1
    else:
        i += 1

# Create DataFrame
df = pd.DataFrame(data)

# Convert to numeric
df['cell_count'] = pd.to_numeric(df['cell_count'], errors='coerce')
df['median_cell_area'] = pd.to_numeric(df['median_cell_area'], errors='coerce')

# Parse condition and time from folder_name
def parse_folder_name(folder_name):
    match = re.match(r'([^S]+)S(\d+)_?(\d+)?', folder_name)
    if match:
        condition = match.group(1)
        time = int(match.group(2))
        volume = int(match.group(3)) if match.group(3) else None
        return condition, time, volume
    return None, None, None

df['condition'] = df['folder_name'].apply(lambda x: parse_folder_name(x)[0])
df['time'] = df['folder_name'].apply(lambda x: parse_folder_name(x)[1])
df['volume'] = df['folder_name'].apply(lambda x: parse_folder_name(x)[2])

# Merge R1- with R1 and 32 with 33
df['condition'] = df['condition'].replace('R1-', 'R1')
df['condition'] = df['condition'].replace('32', '33')

# Normalize count by volume
df['cell_count'] = df['cell_count'] / df['volume']

# Remove rows with invalid parsing or zero values
df = df.dropna(subset=['condition', 'time', 'volume'])

# Get unique conditions
conditions = sorted(df['condition'].unique())

# Create plots for cell count and cell area (separate plots)
fig1, axes1 = plt.subplots(len(conditions), 1, figsize=(12, 4*len(conditions)))
if len(conditions) == 1:
    axes1 = [axes1]

for idx, condition in enumerate(conditions):
    df_cond = df[df['condition'] == condition]
    times = sorted(df_cond['time'].unique())
    
    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    for time in times:
        df_time = df_cond[df_cond['time'] == time]
        data_to_plot.append(df_time['cell_count'].values)
        labels.append(f'T{time}')
    
    # Create boxplot
    bp = axes1[idx].boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    axes1[idx].set_xlabel('Time', fontsize=12)
    axes1[idx].set_ylabel('Normalized cell count (uncorrected)', fontsize=12)
    axes1[idx].set_title(f'Condition {condition} - Cell Count', fontsize=14, fontweight='bold')
    axes1[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boxplot_cell_count.png', dpi=300, bbox_inches='tight')
plt.show()

# Create plots for cell area
fig2, axes2 = plt.subplots(len(conditions), 1, figsize=(12, 4*len(conditions)))
if len(conditions) == 1:
    axes2 = [axes2]

for idx, condition in enumerate(conditions):
    df_cond = df[df['condition'] == condition]
    times = sorted(df_cond['time'].unique())
    
    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    for time in times:
        df_time = df_cond[df_cond['time'] == time]
        data_to_plot.append(df_time['median_cell_area'].values)
        labels.append(f'T{time}')
    
    # Create boxplot
    bp = axes2[idx].boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    
    axes2[idx].set_xlabel('Time', fontsize=12)
    axes2[idx].set_ylabel('Median Cell Area', fontsize=12)
    axes2[idx].set_title(f'Condition {condition} - Median Cell Area', fontsize=14, fontweight='bold')
    axes2[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boxplot_cell_area.png', dpi=300, bbox_inches='tight')
plt.show()

# Create plots for cell area / cell count
df['area_per_count'] = df['median_cell_area'] / df['cell_count']

fig3, axes3 = plt.subplots(len(conditions), 1, figsize=(12, 4*len(conditions)))
if len(conditions) == 1:
    axes3 = [axes3]

for idx, condition in enumerate(conditions):
    df_cond = df[df['condition'] == condition]
    times = sorted(df_cond['time'].unique())
    
    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    for time in times:
        df_time = df_cond[df_cond['time'] == time]
        # Remove inf and nan values
        values = df_time['area_per_count'].replace([np.inf, -np.inf], np.nan).dropna().values
        data_to_plot.append(values)
        labels.append(f'T{time}')
    
    # Create boxplot
    bp = axes3[idx].boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    
    axes3[idx].set_yscale('log')
    axes3[idx].set_xlabel('Time', fontsize=12)
    axes3[idx].set_ylabel('Cell Area / Normalized cell count (uncorrected)', fontsize=12)
    axes3[idx].set_title(f'Condition {condition} - Area per Count', fontsize=14, fontweight='bold')
    axes3[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boxplot_area_per_count.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as:")
print("- boxplot_cell_count.png")
print("- boxplot_cell_area.png")
print("- boxplot_area_per_count.png")

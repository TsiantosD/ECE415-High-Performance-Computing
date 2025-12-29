import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

# --- CONFIGURATION ---
INPUT_DIR = "./reports/"  # Directory containing the CSV files
VALUE_COL = "GPU GInt/s"  # Metric to plot (Change to "CPU GInt/s" if needed)
FILENAME_COL = "Filename"

# Regex to parse filename: version-input-timestamp.csv
# Assumes structure: {Version}-{Input}-{Timestamp}.csv
# The timestamp is typically YYYYMMDD_HHMMSS_mmm (digits and underscores)
FILENAME_PATTERN = re.compile(r"^(.*?)-(.*)-(\d{8}_\d{6}_\d{3})\.csv$")

def parse_filename(filename):
    """
    Extracts Version and Input from the filename.
    Example: sample-galaxy_data-20251229_014952_860.csv
    Returns: ('sample', 'galaxy_data')
    """
    match = FILENAME_PATTERN.match(filename)
    if match:
        return match.group(1), match.group(2)
    
    # Fallback: simple split if regex fails (assumes no extra hyphens in version)
    parts = filename.replace(".csv", "").split("-")
    if len(parts) >= 3:
        return parts[0], "-".join(parts[1:-1])
    return "Unknown", "Unknown"

# --- MAIN EXECUTION ---

csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

if not csv_files:
    print(f"No CSV files found in '{INPUT_DIR}'.")
    exit()

all_data = []
print(f"Found {len(csv_files)} files. Processing...")

for file in csv_files:
    basename = os.path.basename(file)
    
    # Parse metadata from filename
    version_name, input_name = parse_filename(basename)
    
    # Read CSV
    try:
        df = pd.read_csv(file, dtype=str)
    except Exception as e:
        print(f"Skipping {basename}: {e}")
        continue

    # Ensure necessary columns exist
    if FILENAME_COL not in df.columns or VALUE_COL not in df.columns:
        print(f"Skipping {basename}: Missing required columns.")
        continue

    # Filter: Keep only actual run rows (exclude "Average", "Std Dev" statistics rows)
    # The logs from create_csv.py are named "run_1.log", "run_2.log", etc.
    df = df[df[FILENAME_COL].str.contains("run_", na=False)]

    # Convert metric to numeric
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    
    # Drop rows where the specific metric is NaN (e.g., CPU-only runs might have empty GPU cols)
    df = df.dropna(subset=[VALUE_COL])

    if df.empty:
        continue

    # Assign metadata
    df["Version"] = version_name
    df["Input"] = input_name
    
    all_data.append(df)

if not all_data:
    print("No valid data found after processing.")
    exit()

# Combine all dataframes
master_df = pd.concat(all_data, ignore_index=True)

# --- PLOTTING PER INPUT ---

# Get unique inputs to create one plot per input
unique_inputs = master_df["Input"].unique()

sns.set_theme(style="whitegrid")

for input_type in unique_inputs:
    subset = master_df[master_df["Input"] == input_type]
    
    if subset.empty:
        continue

    # Sort versions alphabetically for consistent x-axis
    sorted_versions = sorted(subset["Version"].unique())
    
    plt.figure(figsize=(10, 6))

    # Create Boxplot
    ax = sns.boxplot(
        data=subset,
        x="Version",
        y=VALUE_COL,
        order=sorted_versions,
        palette="viridis",
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 10}
    )

    # Calculate and display max mean for annotation
    means = subset.groupby("Version")[VALUE_COL].mean()
    max_ver = means.idxmax()
    max_val = means.max()
    
    plt.title(f"Performance Comparison: {input_type}", fontsize=16)
    plt.ylabel("Throughput (GInteractions/s)", fontsize=12)
    plt.xlabel("Kernel Version", fontsize=12)
    plt.xticks(rotation=45)
    
    # Optional: Add a subtle note about the metric used
    plt.figtext(0.99, 0.01, f"Metric: {VALUE_COL}", ha="right", fontsize=8, color="gray")

    plt.tight_layout()
    
    output_filename = f"plot_{input_type}.png"
    plt.savefig(output_filename)
    print(f"Saved plot: {output_filename}")
    plt.close() # Close figure to free memory

print("Done.")

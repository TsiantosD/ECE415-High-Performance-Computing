import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

# --- CONFIGURATION ---
INPUT_DIR = "./reports"
FILENAME_COL = "Filename"

# 1. METRIC CONFIGURATION
# We have two potential columns to read
GPU_COL = "GPU GInt/s"
CPU_COL = "CPU GInt/s"

# 2. DEFINING THE CPU BASLINES
# If a filename's version matches these, we plot the CPU column.
# Otherwise, we plot the GPU column.
CPU_VERSIONS = ["seq", "omp", "sequential", "openmp"]

# 3. SORTING PRIORITY
# These versions will always appear first on the X-axis, in this order.
PRIORITY_ORDER = ["seq", "omp", "sequential", "openmp"]

# Regex to parse filename: version-input-timestamp.csv
FILENAME_PATTERN = re.compile(r"^(.*?)-(.*)-(\d{8}_\d{6}_\d{3})\.csv$")

def parse_filename(filename):
    """
    Extracts Version and Input from the filename.
    Returns: ('version', 'input_name')
    """
    match = FILENAME_PATTERN.match(filename)
    if match:
        return match.group(1), match.group(2)
    
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
    version_name, input_name = parse_filename(basename)
    
    try:
        df = pd.read_csv(file, dtype=str)
    except Exception as e:
        print(f"Skipping {basename}: {e}")
        continue

    # Filter for valid run rows
    if FILENAME_COL not in df.columns:
        continue
    df = df[df[FILENAME_COL].str.contains("run_", na=False)]

    # --- HYBRID COLUMN SELECTION ---
    # Determine which column to plot based on the version name
    if version_name in CPU_VERSIONS:
        target_col = CPU_COL
        # Rename it to a common "Throughput" column for plotting consistency
        metric_source = "CPU"
    else:
        target_col = GPU_COL
        metric_source = "GPU"

    # Check if column exists
    if target_col not in df.columns:
        print(f"Warning: {basename} (Version: {version_name}) missing column '{target_col}'. Skipping.")
        continue

    # Convert to numeric
    df["Throughput"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["Throughput"])

    if df.empty:
        continue

    # Assign metadata
    df["Version"] = version_name
    df["Input"] = input_name
    df["Metric Type"] = metric_source # Optional: Useful if you want different colors for CPU vs GPU
    
    all_data.append(df)

if not all_data:
    print("No valid data found.")
    exit()

master_df = pd.concat(all_data, ignore_index=True)

# --- PLOTTING PER INPUT ---

unique_inputs = master_df["Input"].unique()

sns.set_theme(style="whitegrid")

for input_type in unique_inputs:
    subset = master_df[master_df["Input"] == input_type]
    if subset.empty: continue

    # --- CUSTOM SORTING ---
    # 1. Identify versions present in this subset
    present_versions = subset["Version"].unique().tolist()
    
    # 2. Extract priority items that actually exist in data
    ordered_versions = [v for v in PRIORITY_ORDER if v in present_versions]
    
    # 3. Append the rest, sorted alphabetically
    rest = sorted([v for v in present_versions if v not in ordered_versions])
    final_sort_order = ordered_versions + rest

    plt.figure(figsize=(10, 6))

    # Create Boxplot
    ax = sns.boxplot(
        data=subset,
        x="Version",
        y="Throughput",
        order=final_sort_order,
        palette="viridis", # Or use 'hue="Metric Type"' to color CPU/GPU differently
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 10}
    )

    plt.title(f"Performance Comparison: {input_type}", fontsize=16)
    plt.ylabel("Throughput (GInteractions/s)", fontsize=12)
    plt.xlabel("Version", fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"plot_{input_type}.png")
    print(f"Saved plot: plot_{input_type}.png")
    plt.close()

print("Done.")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# --- CONFIGURATION ---
value_col = "GPU Throughput (MPixels/s)"
filename_col = "Filename"

csv_files = glob.glob("csv/*.csv")

if not csv_files:
    print("No CSV files found in the current directory.")
    exit()

all_data = []

print(f"Found {len(csv_files)} files. Processing...")

for file in csv_files:
    # Read CSV, initially as strings
    df = pd.read_csv(file, dtype=str)

    # Keep only rows that correspond to actual runs
    df = df[df[filename_col].str.contains("run_", na=False)]

    # Convert throughput to numeric
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    # Keep only the first two characters of the filename (no extension)
    version_name = os.path.splitext(os.path.basename(file))[0]
    short_label = version_name[:2]

    df["Version"] = short_label
    all_data.append(df)

# Combine everything
master_df = pd.concat(all_data, ignore_index=True)

# Sort x labels alphabetically
sorted_labels = sorted(master_df["Version"].unique())
master_df["Version"] = pd.Categorical(master_df["Version"],
                                      categories=sorted_labels,
                                      ordered=True)

# --- PLOTTING ---

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=master_df,
    x="Version",
    y=value_col,
    palette="viridis",
    showmeans=True,
    meanprops={"marker": "^", "markerfacecolor": "white", "markeredgecolor": "black"}
)

plt.title("GPU Throughput Comparison by Version", fontsize=16)
plt.ylabel("Throughput (MPixels/s)", fontsize=12)
plt.xlabel("Version", fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("throughput.png")
plt.show()

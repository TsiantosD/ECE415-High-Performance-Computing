import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# --- CONFIGURATION ---
# Define the column name exactly as it appears in your CSV
value_col = "GPU Throughput (MPixels/s)"
filename_col = "Filename"

# Locate all CSV files in the current directory
csv_files = glob.glob("csv/*.csv")

if not csv_files:
    print("No CSV files found in the current directory.")
    exit()

all_data = []

print(f"Found {len(csv_files)} files. Processing...")

for file in csv_files:
    # 1. Read the CSV
    # We read everything as strings initially to avoid type errors with the footer text
    df = pd.read_csv(file, dtype=str)
    
    # 2. Data Cleaning
    # The CSV contains a footer with "Average", "Standard Deviation", etc.
    # We only want rows where the 'Filename' column actually looks like a log file (contains "run_")
    df = df[df[filename_col].str.contains("run_", na=False)]
    
    # Convert the Throughput column to numeric, forcing errors to NaN (just in case)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Drop any rows that failed conversion
    df = df.dropna(subset=[value_col])
    
    # 3. Labeling
    # Use the filename (without extension) as the "Version" label for the plot
    version_name = os.path.splitext(os.path.basename(file))[0]
    df['Version'] = version_name
    
    all_data.append(df)

# Combine all loaded files into one big table
master_df = pd.concat(all_data, ignore_index=True)

# --- PLOTTING ---

# Set a visual theme
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Create a Box Plot
# This shows the distribution, median, and outliers for each version
sns.boxplot(
    data=master_df, 
    x='Version', 
    y=value_col, 
    palette="viridis",
    showmeans=True, # Adds a small triangle for the mean
    meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black"}
)

# Optional: Add a Strip Plot on top to see individual data points
# (Uncomment the line below if you want to see every single dot)
# sns.stripplot(data=master_df, x='Version', y=value_col, color='black', alpha=0.3, jitter=True)

plt.title('GPU Throughput Comparison by Version', fontsize=16)
plt.ylabel('Throughput (MPixels/s)', fontsize=12)
plt.xlabel('File Version', fontsize=12)

# Rotate x-labels if you have many files
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
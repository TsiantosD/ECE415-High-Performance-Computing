import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Configuration ---
DEFAULT_OUTPUT_BASE_DIR = 'metrics'
PLOTS_DIR = 'plots'
CSV_DIR = os.path.join(PLOTS_DIR, 'csv')
LOGFILE_REGEX = re.compile(r'.*-(?:run_\d+)\.log$')
MAX_DIFF_REGEX = re.compile(r'Max difference:\s*([0-9.eE+-]+)')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def parse_log_file(path):
    """Extract max difference from a log file."""
    try:
        with open(path, 'r') as f:
            content = f.read()
        m = MAX_DIFF_REGEX.search(content)
        if not m:
            return None
        return float(m.group(1))
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def count_zeros(val):
    """
    Calculates number of zeros after decimal point before the first non-zero digit.
    Example: 0.00123 -> 2 zeros.
    """
    if val <= 0:
        # If error is 0.0, it's a perfect match. 
        # Return a high cap (e.g., 16 for standard double precision) or handle as needed.
        return 7 
    if val >= 1.0:
        return 0
    
    # Formula: -floor(log10(x)) - 1
    # 0.1   (1e-1) -> -(-1) - 1 = 0 zeros
    # 0.01  (1e-2) -> -(-2) - 1 = 1 zero
    # 0.005 (5e-3) -> -(-3) - 1 = 2 zeros
    return int(-np.floor(np.log10(val))) - 1

def collect_data(base_dir):
    records = []

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return pd.DataFrame()

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        print(f"No subdirectories found in '{base_dir}'. Exiting.")
        return pd.DataFrame()
    
    if len(subdirs) == 1:
        selected_subdir = subdirs[0]
        print(f"Only one subdirectory found: **{selected_subdir}**. Selecting it automatically.")
    else:
        print("\nMultiple metrics subdirectories found. Select one to plot:")
        for idx, d in enumerate(subdirs, start=1):
            print(f"  {idx}) {d}")
        
        selected_subdir = None
        while selected_subdir is None:
            try:
                choice = input("Enter number: ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(subdirs):
                    selected_subdir = subdirs[choice_idx]
                    break
            except ValueError:
                pass
            print("Invalid selection.")

    metrics_path = os.path.join(base_dir, selected_subdir)
    src_folder_placeholder = selected_subdir 

    for config_dir in os.listdir(metrics_path):
        config_path = os.path.join(metrics_path, config_dir)
        
        if not os.path.isdir(config_path):
            continue

        match = re.match(r'(\d+)_(\d+)(-nofmad)?(-dbl)?', config_dir)
        if not match:
            continue
            
        filter_radius = int(match.group(1))
        image_size = int(match.group(2))
        nofmad_flag = bool(match.group(3))
        dbl_flag = bool(match.group(4))

        for logfile in os.listdir(config_path):
            if not LOGFILE_REGEX.match(logfile):
                continue
            
            full_path = os.path.join(config_path, logfile)
            max_diff = parse_log_file(full_path)
            if max_diff is None:
                continue

            records.append({
                "src_folder": src_folder_placeholder, 
                "filter_radius": filter_radius,
                "image_size": image_size,
                "nofmad": nofmad_flag,
                "dbl": dbl_flag,
                "logfile": logfile,
                "max_diff": max_diff
            })

    return pd.DataFrame(records)


def plot_max_diff(df):
    if df.empty:
        print("No data to plot.")
        return

    # 1. APPLY TRANSFORMATION HERE
    # Calculate the zero count for every row
    df['zero_count'] = df['max_diff'].apply(count_zeros)

    # Create label
    df['label'] = df.apply(
        lambda r: f"{r['src_folder']}{'-nofmad' if r['nofmad'] else ''}{'-dbl' if r['dbl'] else ''}", axis=1
    )

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.5,
        "lines.markersize": 10
    })

    for label, group in df.groupby('label'):
        # 2. AGGREGATE ON THE NEW 'zero_count' COLUMN
        agg = group.groupby('filter_radius').agg(
            zeros_mean=('zero_count', 'mean'),
            zeros_std=('zero_count', 'std'),
            count=('zero_count', 'count')
        ).reset_index()

        agg['zeros_std'] = agg['zeros_std'].fillna(0)

        # Save CSV (renamed file to reflect new metric)
        csv_file = os.path.join(CSV_DIR, f"{label}_zerocount_vs_radius.csv")
        agg.to_csv(csv_file, index=False)
        print(f"Saved CSV data: {csv_file}")

        # Plot
        plt.figure(figsize=(9, 6))
        
        # Plotting Mean Zeros vs Radius
        plt.errorbar(
            agg['filter_radius'], agg['zeros_mean'], yerr=agg['zeros_std'],
            fmt='o-', capsize=5, linewidth=2
        )

        plt.title(f"Accuracy (Leading Zeros)", fontsize=16)
        plt.xlabel("Filter Radius", fontsize=14)
        
        # Update Y-Label
        plt.ylabel("Count of zero digits", fontsize=14)
        
        plt.grid(True, alpha=0.3)

        # Update filename
        out_path = os.path.join(PLOTS_DIR, f"{label}_zerocount_vs_radius.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == '__main__':
    output_base_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_BASE_DIR
    print(f"Using metrics directory: {output_base_dir}")

    df_all = collect_data(output_base_dir)
    plot_max_diff(df_all)
    print("Done.")
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

# Optional: limit plotting to these src folders or leave empty to include all
TARGET_SRC_FOLDERS = []  # e.g., ['step5a', 'step6a']

# Regex for log files: timestamp-run_X.log
LOGFILE_REGEX = re.compile(r'.*-(?:run_\d+)\.log$')

# Regex to extract max difference from log content
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


def collect_data(base_dir):
    """
    Collect max difference data from all runs in metrics.
    Adjusted for structure: base_dir / selected_subdir / config_dir / logfiles
    """
    records = []

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return pd.DataFrame()

    # --- Directory Selection Logic (Kept as is) ---
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
            print("Invalid selection. Please enter a valid number.")

    metrics_path = os.path.join(base_dir, selected_subdir)
    print(f"Selected metrics path: **{metrics_path}**")
    # --- End of Directory Selection Logic ---
    
    # --- REVISED DATA COLLECTION LOGIC ---
    # We treat the directories like '128_1024' as the config_dir (Level 2).
    # We use the selected_subdir (e.g., 'precision') as the 'src_folder' label.
    
    # Use the selected subdirectory name as the placeholder for the 'src_folder' column
    src_folder_placeholder = selected_subdir 

    # Iterate over directories like '128_1024', '16_1024', etc. (These are config_dir)
    for config_dir in os.listdir(metrics_path):
        config_path = os.path.join(metrics_path, config_dir)
        
        if not os.path.isdir(config_path):
            continue

        # Extract filter_radius and image_size
        match = re.match(r'(\d+)_(\d+)(-nofmad)?(-dbl)?', config_dir)
        if not match:
            continue
            
        filter_radius = int(match.group(1))
        image_size = int(match.group(2))
        nofmad_flag = bool(match.group(3))
        dbl_flag = bool(match.group(4))

        # Iterate over all timestamped log files *directly* inside the config directory
        for logfile in os.listdir(config_path):
            if not LOGFILE_REGEX.match(logfile):
                continue
            
            full_path = os.path.join(config_path, logfile)
            max_diff = parse_log_file(full_path)
            if max_diff is None:
                continue

            records.append({
                # Use the placeholder value here:
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

    # Create a label for each combination of src_folder and flags
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
        # Compute mean and std of max_diff per filter_radius
        agg = group.groupby('filter_radius').agg(
            max_diff_mean=('max_diff', 'mean'),
            max_diff_std=('max_diff', 'std'),
            count=('max_diff', 'count')
        ).reset_index()

        # Fill NaN std for single-run groups
        agg['max_diff_std'] = agg['max_diff_std'].fillna(0)

        # Save CSV
        csv_file = os.path.join(CSV_DIR, f"{label}_maxdiff_vs_radius.csv")
        agg.to_csv(csv_file, index=False)
        print(f"Saved CSV data: {csv_file}")

        # Plot
        plt.figure(figsize=(9, 6))
        plt.errorbar(
            agg['filter_radius'], agg['max_diff_mean'], yerr=agg['max_diff_std'],
            fmt='o-', capsize=5, linewidth=2
        )

        plt.title(f"{'Doubles' if 'dbl' in label else 'Floats'} - Max Difference vs Filter Radius", fontsize=16)
        plt.xlabel("Filter Radius", fontsize=14)
        plt.ylabel("Max Difference (error)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)

        out_path = os.path.join(PLOTS_DIR, f"{label}_maxdiff_vs_radius.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == '__main__':
    output_base_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_BASE_DIR
    print(f"Using metrics directory: {output_base_dir}")

    df_all = collect_data(output_base_dir)
    plot_max_diff(df_all)
    print("Done.")

import os
import re
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Default Configuration ---
DEFAULT_OUTPUT_BASE_DIR = 'metrics'
PLOTS_DIR = 'plots'
CSV_DIR = os.path.join(PLOTS_DIR, 'csv')
TARGET_KERNEL_LENGTH = 33  # actual kernel length, e.g., 2*filter_radius+1

# Regex for parsing logs: timestamp-run files
FILENAME_REGEX = re.compile(r'.*-(?:run_\d+)\.log$')

# Regex for parsing times from log files
TIME_GPU_REGEX = re.compile(r'Time in GPU: ([\d\.]+)')
TIME_CPU_REGEX = re.compile(r'Time in CPU: ([\d\.]+)')


def parse_log_file(filepath):
    """Extracts GPU and CPU times from a log file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        gpu_time_match = TIME_GPU_REGEX.search(content)
        cpu_time_match = TIME_CPU_REGEX.search(content)
        gpu_time = float(gpu_time_match.group(1)) if gpu_time_match else None
        cpu_time = float(cpu_time_match.group(1)) if cpu_time_match else None
        return gpu_time, cpu_time
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None, None


def collect_data(output_base_dir):
    """
    Collects GPU/CPU times from all metrics runs in the folder structure.
    Adjusted to handle structure: base_dir / config_dir / logfiles
    """
    data = []

    if not os.path.isdir(output_base_dir):
        print(f"Error: Output base directory '{output_base_dir}' does not exist.")
        return data

    # Prompt user to select a subdirectory under metrics if multiple exist
    subdirs = [d for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))]
    
    # --- Start of user selection logic (Kept for completeness) ---
    if not subdirs:
        # If running on 'metrics/runtimes' directly, this might be skipped
        selected_dir = "" 
        print(f"No subdirectories found in '{output_base_dir}'. Proceeding with base directory.")
    elif len(subdirs) == 1:
        selected_dir = subdirs[0]
    else:
        print("Multiple metrics subdirectories found. Select one to plot:")
        for idx, d in enumerate(subdirs, start=1):
            print(f"{idx}) {d}")
        while True:
            try:
                choice = int(input("Enter number: "))
                if 1 <= choice <= len(subdirs):
                    selected_dir = subdirs[choice - 1]
                    break
            except ValueError:
                pass
            print("Invalid selection. Try again.")
            
    if selected_dir:
        output_base_dir = os.path.join(output_base_dir, selected_dir)
        
    print(f"Selected metrics folder: {output_base_dir}")
    # --- End of user selection logic ---

    # --- Start of Revised Data Collection Logic ---
    # In the original script: 
    #   Level 1: selected_dir (handled above)
    #   Level 2: src_folder (e.g., 'default_run') - NO LONGER PRESENT
    #   Level 3: config_dir (e.g., '16_1024') - THIS IS NOW LEVEL 2
    
    # Iterate over the config directories (e.g., '16_1024', '16_1024-dbl', etc.)
    for config_dir in os.listdir(output_base_dir):
        config_path = os.path.join(output_base_dir, config_dir)
        
        # Check if the item is a directory
        if not os.path.isdir(config_path):
            continue

        # Check if the directory name matches the expected configuration regex
        match = re.match(r'(\d+)_(\d+)(-nofmad)?(-dbl)?', config_dir)
        if not match:
            continue

        filter_radius = int(match.group(1))
        image_size = int(match.group(2))
        nofmad_flag = bool(match.group(3))
        dbl_flag = bool(match.group(4))
        
        # We need a placeholder for the original 'src_folder' since it's used 
        # later for plotting labels. We use the directory name of the runtimes folder.
        # Example: 'runtimes' if the input was 'metrics/runtimes'
        src_folder_placeholder = os.path.basename(output_base_dir)
        
        # Now, iterate through the log files directly inside the config directory
        for logfile in os.listdir(config_path):
            if not FILENAME_REGEX.match(logfile):
                continue
            
            filepath = os.path.join(config_path, logfile)
            gpu_time, cpu_time = parse_log_file(filepath)
            
            if gpu_time is not None and cpu_time is not None:
                data.append({
                    'src_folder': src_folder_placeholder, # Using the base folder name
                    'filter_radius': filter_radius,
                    'image_size': image_size,
                    'gpu_time': gpu_time,
                    'cpu_time': cpu_time,
                    'nofmad': nofmad_flag,
                    'dbl': dbl_flag,
                    'logfile': logfile
                })

    return data
# --- End of Revised Data Collection Logic ---


def generate_plots(df):
    """Generates plots for GPU/CPU mean runtimes and saves CSV data with speedup and std dev."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

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

    # Label generation logic remains the same, using the placeholder 'src_folder'
    df['label'] = df.apply(
        lambda r: f"{r['src_folder']}{'-nofmad' if r['nofmad'] else ''}{'-dbl' if r['dbl'] else ''}", axis=1
    )

    for label, group in df.groupby('label'):
        agg = (
            group.groupby(['filter_radius', 'image_size'])
                 .agg(
                     gpu_time_mean=('gpu_time', 'mean'),
                     cpu_time_mean=('cpu_time', 'mean'),
                     gpu_time_std=('gpu_time', 'std'),
                     cpu_time_std=('cpu_time', 'std'),
                     count=('gpu_time', 'count')
                 )
                 .reset_index()
        )

        agg[['gpu_time_std', 'cpu_time_std']] = agg[['gpu_time_std', 'cpu_time_std']].fillna(0)
        
        # Calculate speedup and standard deviation of speedup
        agg['speedup'] = agg['cpu_time_mean'] / agg['gpu_time_mean']
        # Error propagation formula for division (R = A/B): (sigma_R/R)^2 = (sigma_A/A)^2 + (sigma_B/B)^2
        agg['speedup_std'] = agg['speedup'] * np.sqrt(
            (agg['cpu_time_std'] / agg['cpu_time_mean'])**2 +
            (agg['gpu_time_std'] / agg['gpu_time_mean'])**2
        )

        csv_file = os.path.join(CSV_DIR, f"{label}_runtime_k{TARGET_KERNEL_LENGTH}.csv")
        agg.to_csv(csv_file, index=False)
        print(f"Saved CSV data: {csv_file}")

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(agg['image_size'], agg['gpu_time_mean'], marker='o', linestyle='-', color='tab:blue', label='GPU Time')
        ax.plot(agg['image_size'], agg['cpu_time_mean'], marker='x', linestyle='--', color='tab:orange', label='CPU Time')

        for _, row in agg.iterrows():
            ax.text(row['image_size'],
                    row['gpu_time_mean'] - 4,
                    f"{row['gpu_time_mean']:.4f}±{row['gpu_time_std']:.4f}",
                    ha='center', va='bottom', color='tab:blue', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.4"))
            ax.text(row['image_size'],
                    row['cpu_time_mean'] + (8 if row['cpu_time_mean'] < 25 else 4), # add padding
                    f"{row['cpu_time_mean']:.4f}±{row['cpu_time_std']:.4f}",
                    ha='center', va='top', color='tab:orange', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.4"))

        filter_radius_val = agg['filter_radius'].iloc[0] if not agg.empty else '?'
        title_prefix = "Doubles" if '-dbl' in label else "Floats"
        ax.set_title(f"{title_prefix} – CPU/GPU Runtime (Filter radius: {filter_radius_val})")
        ax.set_xlabel('Image Size (N)')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_xscale('log', base=2)
        ax.grid(True, linestyle='--', axis='x')

        ticks = sorted(agg['image_size'].unique())
        if ticks:
            ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))

        ax.legend(loc='upper left')
        plt.tight_layout()
        plot_file = os.path.join(PLOTS_DIR, f"{label}_runtime_k{TARGET_KERNEL_LENGTH}.png")
        plt.savefig(plot_file)
        plt.close(fig)
        print(f"Saved plot for {label} to {plot_file}")


if __name__ == '__main__':
    base_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_BASE_DIR
    print(f"Base metrics directory: {base_dir}")

    all_data = collect_data(base_dir)
    if not all_data:
        print("No data found in the selected metrics folder.")
        sys.exit(1)

    df = pd.DataFrame(all_data)
    generate_plots(df)
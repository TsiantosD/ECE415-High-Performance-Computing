import os
import re
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd

# --- Default Configuration ---
DEFAULT_OUTPUT_BASE_DIR = 'output'
PLOTS_DIR = 'plots'
TARGET_STEPS = ['step6', 'step6_dbl']
TARGET_KERNEL_LENGTH = 33

# Regex for filenames
FILENAME_REGEX = re.compile(r'out_(\d+)_(\d+)(?:_rep\d+)?$')

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
        print(f"Error parsing file {filepath}: {e}")
        return None, None


def collect_data(output_dir, step_name):
    """Collects GPU/CPU measurements from all runs inside step folders."""
    data = []

    step_path = os.path.join(output_dir, step_name)
    if not os.path.isdir(step_path):
        print(f"Warning: Directory '{step_path}' not found. Skipping.")
        return data

    for first_level_dir in os.listdir(step_path):
        first_level_path = os.path.join(step_path, first_level_dir)
        if not os.path.isdir(first_level_path):
            continue

        timestamp_dirs_to_check = []
        exec_name = first_level_dir

        # First level is a timestamp
        if re.match(r'^\d{8}_\d{6}$', first_level_dir):
            timestamp_dirs_to_check.append(first_level_path)
            exec_name = step_name
        else:
            # First level is executable suffix, second has timestamps
            for sub_dir in os.listdir(first_level_path):
                if re.match(r'^\d{8}_\d{6}$', sub_dir):
                    timestamp_dirs_to_check.append(os.path.join(first_level_path, sub_dir))

        # Iterate through timestamp dirs
        for timestamp_path in timestamp_dirs_to_check:
            for filename in os.listdir(timestamp_path):
                if not filename.startswith('out_'):
                    continue

                match = FILENAME_REGEX.match(filename)
                if not match:
                    continue

                kernel_length = int(match.group(1))
                image_size = int(match.group(2))

                if kernel_length != TARGET_KERNEL_LENGTH:
                    continue

                filepath = os.path.join(timestamp_path, filename)

                gpu_time, cpu_time = parse_log_file(filepath)
                if gpu_time is not None and cpu_time is not None:
                    data.append({
                        'step': step_name,
                        'kernel_length': kernel_length,
                        'image_size': image_size,
                        'gpu_time': gpu_time,
                        'cpu_time': cpu_time,
                        'executable': exec_name
                    })

    return data


def generate_separate_plots(df):
    """Plots separate GPU/CPU mean times with runtime annotations."""

    os.makedirs(PLOTS_DIR, exist_ok=True)
    df['image_size'] = df['image_size'].astype(int)

    plt.rcParams.update({
        "font.size": 16,            # base font size
        "axes.titlesize": 20,       # title size
        "axes.labelsize": 18,       # axis label size
        "xtick.labelsize": 16,      # tick label size
        "ytick.labelsize": 16,
        "legend.fontsize": 16,      # legend text
        "lines.linewidth": 2.5,     # thicker lines
        "lines.markersize": 10      # bigger points
    })

    # Group by step (step6, step6_dbl)
    for step, group in df.groupby('step'):
        group_sorted = group.sort_values(by='image_size')

        fig, ax1 = plt.subplots(figsize=(12, 7))

        # GPU mean time
        line_gpu, = ax1.plot(
            group_sorted['image_size'],
            group_sorted['gpu_time_mean'],
            marker='o',
            linestyle='-',
            color='tab:blue',
            label='GPU Time (mean)'
        )

        # CPU mean time
        line_cpu, = ax1.plot(
            group_sorted['image_size'],
            group_sorted['cpu_time_mean'],
            marker='x',
            linestyle='--',
            color='tab:orange',
            label='CPU Time (mean)'
        )

        # Annotate GPU & CPU runtimes on each point
        for _, row in group_sorted.iterrows():

            gpu_label = f"{row['gpu_time_mean']:.4f}±{row['gpu_time_std']:.4f}s"
            cpu_label = f"{row['cpu_time_mean']:.4f}±{row['cpu_time_std']:.4f}s"

            # GPU annotation
            y_offset_gpu = -2.5
            ax1.text(
                row['image_size'],
                row['gpu_time_mean'] + y_offset_gpu,
                gpu_label,
                ha='center',
                va='bottom',
                fontsize=12,
                color='tab:blue',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.4")
            )

            # CPU annotation
            y_offset_cpu = 2.5
            ax1.text(
                row['image_size'],
                row['cpu_time_mean'] + y_offset_cpu,
                cpu_label,
                ha='center',
                va='top',
                fontsize=12,
                color='tab:orange',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.4")
            )

        title_prefix = "Doubles" if "_dbl" in step else "Floats"
        ax1.set_title(f'{title_prefix} – CPU/GPU Runtime (Filter radius: {math.floor(TARGET_KERNEL_LENGTH / 2)})')
        ax1.set_xlabel('Image Size (N)')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.grid(True, linestyle='--', axis='x')
        plt.xscale('log', base=2)

        if not group_sorted['image_size'].empty:
            ticks = sorted(group_sorted['image_size'].unique())
            ax1.set_xticks(ticks)
            ax1.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))

        ax1.legend(loc='upper left')
        plt.tight_layout()

        plot_path = os.path.join(PLOTS_DIR, f'{step}_runtime_k{TARGET_KERNEL_LENGTH}.png')
        plt.savefig(plot_path)
        plt.show()

        print(f"Saved plot for {step} to {plot_path}")


if __name__ == '__main__':
    # Command line argument for output dir
    if len(sys.argv) > 1:
        OUTPUT_BASE_DIR = sys.argv[1]
        print(f"Using custom output directory: {OUTPUT_BASE_DIR}")
    else:
        OUTPUT_BASE_DIR = DEFAULT_OUTPUT_BASE_DIR
        print(f"Using default output directory: {OUTPUT_BASE_DIR}")

    # Ensure output directory exists
    if not os.path.isdir(OUTPUT_BASE_DIR):
        print(f"Error: Provided output directory '{OUTPUT_BASE_DIR}' does not exist.")
        sys.exit(1)

    all_data = []

    # Collect data from all target steps
    for step in TARGET_STEPS:
        print(f"Collecting data for Kernel Length {TARGET_KERNEL_LENGTH} from {OUTPUT_BASE_DIR}/{step}...")
        all_data.extend(collect_data(OUTPUT_BASE_DIR, step))

    if not all_data:
        print("Error: No data found. Check your directory structure.")
    else:
        df = pd.DataFrame(all_data)

        # Compute mean & std dev per (step, kernel_length, image_size)
        df_agg = (
            df.groupby(['step', 'kernel_length', 'image_size'])
              .agg(
                  gpu_time_mean=('gpu_time', 'mean'),
                  cpu_time_mean=('cpu_time', 'mean'),
                  gpu_time_std=('gpu_time', 'std'),
                  cpu_time_std=('cpu_time', 'std'),
                  count=('gpu_time', 'count')
              )
              .reset_index()
        )

        # Fix NaN std dev for groups with only 1 run
        df_agg[['gpu_time_std', 'cpu_time_std']] = df_agg[['gpu_time_std', 'cpu_time_std']].fillna(0)

        generate_separate_plots(df_agg)

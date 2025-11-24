import os
import re
import math
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
OUTPUT_BASE_DIR = 'output'
PLOTS_DIR = 'plots'
TARGET_STEPS = ['step5', 'step5_dbl']
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
    """Plots separate GPU/CPU mean times and speedup with std dev annotations."""
    
    # Speedup mean
    df['speedup'] = df['cpu_time_mean'] / df['gpu_time_mean']

    # Speedup std dev (error propagation)
    df['speedup_std'] = (
        df['speedup'] *
        ((df['cpu_time_std'] / df['cpu_time_mean'])**2 +
         (df['gpu_time_std'] / df['gpu_time_mean'])**2)**0.5
    )

    # Fix NaN speedup_std
    df['speedup_std'] = df['speedup_std'].fillna(0)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df['image_size'] = df['image_size'].astype(int)

    # Group by step (step5, step5_dbl)
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

        # Annotate speedup on each point
        for _, row in group_sorted.iterrows():
            text = f"{row['speedup']:.2f}±{row['speedup_std']:.2f}x"
            ax1.text(
                row['image_size'],
                row['gpu_time_mean'] + 1.5,
                text,
                ha='center',
                va='bottom',
                fontsize=12,
                color='tab:blue',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.5")
            )

        ax1.set_title(f'Runtime and Speedup (Kernel Radius: {math.floor(TARGET_KERNEL_LENGTH / 2)})')
        ax1.set_xlabel('Image Size (N)')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.grid(True, linestyle='--', axis='x')

        if not group_sorted['image_size'].empty:
            ax1.set_xticks(sorted(group_sorted['image_size'].unique()))

        ax1.legend(loc='upper left')

        plt.tight_layout()

        plot_path = os.path.join(PLOTS_DIR, f'{step}_speedup_k{TARGET_KERNEL_LENGTH}.png')
        plt.savefig(plot_path)
        plt.show()

        print(f"Saved plot for {step} to {plot_path}")


if __name__ == '__main__':
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

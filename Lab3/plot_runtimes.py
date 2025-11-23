import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
OUTPUT_BASE_DIR = 'output'
PLOTS_DIR = 'plots' # New directory for plots
TARGET_STEPS = ['step5', 'step6']

# Regex to extract data from the log filename
# e.g., out_11_32, where 11 is kernel_length and 32 is image_size
FILENAME_REGEX = re.compile(r'out_(\d+)_(\d+)(?:_rep\d+)?$')

# Regexes to extract time and size from the file content
TIME_GPU_REGEX = re.compile(r'Time in GPU: ([\d\.]+)')
TIME_CPU_REGEX = re.compile(r'Time in CPU: ([\d\.]+)')

def parse_log_file(filepath):
    """Parses a single log file to extract GPU and CPU times."""
    with open(filepath, 'r') as f:
        content = f.read()

    gpu_time_match = TIME_GPU_REGEX.search(content)
    cpu_time_match = TIME_CPU_REGEX.search(content)

    gpu_time = float(gpu_time_match.group(1)) if gpu_time_match else None
    cpu_time = float(cpu_time_match.group(1)) if cpu_time_match else None
    
    return gpu_time, cpu_time

def collect_data(output_dir, step_name):
    """Scans step directories and collects execution data."""
    data = []
    
    step_path = os.path.join(output_dir, step_name)
    if not os.path.isdir(step_path):
        print(f"Warning: Directory '{step_path}' not found. Skipping.")
        return data

    # Find ALL subdirectories under step_path (e.g., '20251123_234518' or 'step5_dbl')
    for first_level_dir in os.listdir(step_path):
        first_level_path = os.path.join(step_path, first_level_dir)
        if not os.path.isdir(first_level_path):
            continue
            
        # Determine the paths to check for log files
        timestamp_dirs_to_check = []
        
        # Case 1: The subdirectory IS the timestamp directory (Your structure)
        # We can check if the directory name looks like a timestamp (e.g., starts with 20 and has 8 digits)
        if re.match(r'^\d{8}_\d{6}$', first_level_dir):
            timestamp_dirs_to_check.append(first_level_path)
            # Use the step name itself as the executable/suffix identifier for this plot
            exec_name = step_name
        
        # Case 2: The subdirectory is the executable/suffix directory (Script's original assumption)
        else:
            # We assume first_level_dir is the executable name (e.g., step5_dbl)
            exec_name = first_level_dir
            # Check for timestamp directories inside this path
            for sub_dir in os.listdir(first_level_path):
                if re.match(r'^\d{8}_\d{6}$', sub_dir):
                    timestamp_dirs_to_check.append(os.path.join(first_level_path, sub_dir))

        # 3. Iterate through all identified timestamp directories
        for timestamp_path in timestamp_dirs_to_check:
            # Check if this path actually holds log files
            for filename in os.listdir(timestamp_path):
                if not filename.startswith('out_'):
                    continue
                
                match = FILENAME_REGEX.match(filename)
                if not match:
                    continue

                kernel_length = int(match.group(1))
                image_size = int(match.group(2))
                
                filepath = os.path.join(timestamp_path, filename)
                
                # Parse times from the log content
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

def generate_plots(df, step_names):
    """Generates two plots: one for GPU time and one for CPU time."""
    
    # Create the plots directory if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Ensure image_size is treated as a numerical category
    df['image_size'] = df['image_size'].astype(int)
    
    # Determine the title prefix
    title_prefix = f"{', '.join(step_names)} Results"

    # --- Plot 1: GPU Time ---
    plt.figure(figsize=(12, 6))
    
    # Group by kernel_length for plotting separate lines
    for kernel, group in df.groupby('kernel_length'):
        # We need to sort by image_size for a proper line plot
        group_sorted = group.sort_values(by='image_size')
        
        # Plot Time in MS (Multiply by 1000)
        plt.plot(
            group_sorted['image_size'], 
            group_sorted['gpu_time'] * 1000, # <-- CHANGED TO MS
            marker='o', 
            linestyle='-',
            label=f'Kernel Length: {kernel}'
        )

    plt.title(f'{title_prefix}: GPU Execution Time vs. Image Size')
    plt.xlabel('Image Size (N)')
    plt.ylabel('Time in GPU (milliseconds)') # <-- CHANGED LABEL
    plt.xticks(sorted(df['image_size'].unique())) # Ensure all discrete sizes are shown
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Filter Configuration')
    plt.tight_layout()
    gpu_plot_path = os.path.join(PLOTS_DIR, 'gpu_time_plot.png')
    plt.savefig(gpu_plot_path)
    plt.show()
    print(f"Saved GPU plot to {gpu_plot_path}")

    # --- Plot 2: CPU Time ---
    plt.figure(figsize=(12, 6))
    
    for kernel, group in df.groupby('kernel_length'):
        group_sorted = group.sort_values(by='image_size')
        
        # Plot Time in MS (Multiply by 1000)
        plt.plot(
            group_sorted['image_size'], 
            group_sorted['cpu_time'] * 1000, # <-- CHANGED TO MS
            marker='x', 
            linestyle='--',
            label=f'Kernel Length: {kernel}'
        )

    plt.title(f'{title_prefix}: CPU Execution Time vs. Image Size')
    plt.xlabel('Image Size (N)')
    plt.ylabel('Time in CPU (milliseconds)') # <-- CHANGED LABEL
    plt.xticks(sorted(df['image_size'].unique()))
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Filter Configuration')
    plt.tight_layout()
    cpu_plot_path = os.path.join(PLOTS_DIR, 'cpu_time_plot.png')
    plt.savefig(cpu_plot_path)
    plt.show()
    print(f"Saved CPU plot to {cpu_plot_path}")


if __name__ == '__main__':
    all_data = []
    
    # Collect data from all specified steps
    for step in TARGET_STEPS:
        print(f"Collecting data from {OUTPUT_BASE_DIR}/{step}...")
        all_data.extend(collect_data(OUTPUT_BASE_DIR, step))

    if not all_data:
        print("Error: No data found matching the required format in the target directories.")
    else:
        # Convert list of dictionaries to a Pandas DataFrame
        df = pd.DataFrame(all_data)
        
        # Aggregate data if multiple identical runs exist (e.g., from --repeat mode or multiple runs)
        # We use the mean time for simplicity across identical (kernel, size) pairs.
        df_agg = df.groupby(['kernel_length', 'image_size']).mean(numeric_only=True).reset_index()
        
        generate_plots(df_agg, TARGET_STEPS)

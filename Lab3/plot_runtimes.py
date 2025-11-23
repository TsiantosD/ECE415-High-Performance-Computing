import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
OUTPUT_BASE_DIR = 'output'
PLOTS_DIR = 'plots'
TARGET_STEPS = ['step5', 'step6']
TARGET_KERNEL_LENGTH = 16

# Regex to extract data from the log filename
# The kernel length is the first number in the filename
FILENAME_REGEX = re.compile(r'out_(\d+)_(\d+)(?:_rep\d+)?$')

# Regexes to extract time from the file content
TIME_GPU_REGEX = re.compile(r'Time in GPU: ([\d\.]+)')
TIME_CPU_REGEX = re.compile(r'Time in CPU: ([\d\.]+)')

def parse_log_file(filepath):
    """Parses a single log file to extract GPU and CPU times."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        gpu_time_match = TIME_GPU_REGEX.search(content)
        cpu_time_match = TIME_CPU_REGEX.search(content)

        gpu_time = float(gpu_time_match.group(1)) if gpu_time_match else None
        cpu_time = float(cpu_time_match.group(1)) if cpu_time_match else None
        
        return gpu_time, cpu_time
    except Exception as e:
        # Handle cases where the file is unreadable or time is missing
        print(f"Error parsing file {filepath}: {e}")
        return None, None

def collect_data(output_dir, step_name):
    """Scans step directories and collects execution data."""
    data = []
    
    step_path = os.path.join(output_dir, step_name)
    if not os.path.isdir(step_path):
        print(f"Warning: Directory '{step_path}' not found. Skipping.")
        return data

    # Iterate through all first-level subdirectories (timestamp OR executable suffix)
    for first_level_dir in os.listdir(step_path):
        first_level_path = os.path.join(step_path, first_level_dir)
        if not os.path.isdir(first_level_path):
            continue
            
        timestamp_dirs_to_check = []
        exec_name = first_level_dir
        
        # Check if the directory itself is a timestamp (e.g., '20251123_234518')
        if re.match(r'^\d{8}_\d{6}$', first_level_dir):
            timestamp_dirs_to_check.append(first_level_path)
            exec_name = step_name # Use step name as identifier
        
        # Otherwise, assume it's an executable suffix (e.g., 'step5_dbl')
        else:
            for sub_dir in os.listdir(first_level_path):
                if re.match(r'^\d{8}_\d{6}$', sub_dir):
                    timestamp_dirs_to_check.append(os.path.join(first_level_path, sub_dir))

        # Iterate through all identified timestamp directories
        for timestamp_path in timestamp_dirs_to_check:
            for filename in os.listdir(timestamp_path):
                if not filename.startswith('out_'):
                    continue
                
                match = FILENAME_REGEX.match(filename)
                if not match:
                    continue

                kernel_length = int(match.group(1))
                
                # Check for the target kernel length
                if kernel_length != TARGET_KERNEL_LENGTH:
                    continue

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

def generate_combined_plot(df, step_names):
    """Generates a single plot comparing CPU and GPU runtimes."""
    
    # Create the plots directory if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Ensure image_size is treated as a numerical category
    df['image_size'] = df['image_size'].astype(int)
    
    # Determine the title prefix
    title_prefix = f"Steps {', '.join(step_names)} Runtime Comparison (Kernel Length: {TARGET_KERNEL_LENGTH})"

    plt.figure(figsize=(12, 7))
    
    # Group by step (step5, step6) to plot data from different steps separately if they exist
    # If the mean time for a given size is identical across steps, the lines will overlap.
    for step, group in df.groupby('step'):
        # Sort by image_size for a proper line plot
        group_sorted = group.sort_values(by='image_size')
        
        # --- GPU Time Line (Converted to MS) ---
        plt.plot(
            group_sorted['image_size'], 
            group_sorted['gpu_time'] * 1000,
            marker='o', 
            linestyle='-',
            label=f'GPU Time ({step})'
        )

        # --- CPU Time Line (Converted to MS) ---
        plt.plot(
            group_sorted['image_size'], 
            group_sorted['cpu_time'] * 1000,
            marker='x', 
            linestyle='--',
            label=f'CPU Time ({step})'
        )

    plt.title(title_prefix)
    plt.xlabel('Image Size (N)')
    plt.ylabel('Runtime (milliseconds)')
    plt.xticks(sorted(df['image_size'].unique()))
    plt.grid(True, which='both', linestyle='--')
    plt.legend(title='Execution Mode')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, f'combined_runtime_k{TARGET_KERNEL_LENGTH}.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved combined runtime plot to {plot_path}")


if __name__ == '__main__':
    all_data = []
    
    # Collect data from all specified steps
    for step in TARGET_STEPS:
        print(f"Collecting data for Kernel Length {TARGET_KERNEL_LENGTH} from {OUTPUT_BASE_DIR}/{step}...")
        all_data.extend(collect_data(OUTPUT_BASE_DIR, step))

    if not all_data:
        print(f"Error: No data found for Kernel Length {TARGET_KERNEL_LENGTH} in the target directories.")
    else:
        # Convert list of dictionaries to a Pandas DataFrame
        df = pd.DataFrame(all_data)
        
        # Aggregate data by grouping. Use numeric_only=True to handle string columns safely.
        # Group by step as well, in case we want to differentiate step5 vs step6 times.
        df_agg = df.groupby(['step', 'kernel_length', 'image_size']).mean(numeric_only=True).reset_index()
        
        generate_combined_plot(df_agg, TARGET_STEPS)
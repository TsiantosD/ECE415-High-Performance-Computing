import os
import math
import matplotlib.pyplot as plt
from datetime import datetime

# ================= CONFIGURATION =================
# Folders
# OUTPUT_DIR removed, we will scan dynamically
PLOTS_DIR = "plots"     # Where to save the new graph images
# =================================================

def parse_results(current_output_dir):
    """Reads the output files and extracts data for plotting."""
    data_points = [] # Stores dicts with parsed data

    if not os.path.exists(current_output_dir):
        print(f"Error: Directory '{current_output_dir}' not found. Please ensure data exists.")
        return [], [], [], None

    print(f"Scanning '{current_output_dir}' for files...")
    
    try:
        files = os.listdir(current_output_dir)
    except OSError as e:
        print(f"Error accessing directory: {e}")
        return [], [], [], None

    for filename in files:
        # Expected format: out_<first_input>_<second_input>
        if not filename.startswith("out_"):
            continue
        
        parts = filename.split('_')
        # We expect at least 3 parts: "out", "0", "32"
        if len(parts) < 3:
            continue
            
        try:
            first_input = int(parts[1])
            second_input = int(parts[2]) # Assumes filename ends there or just parses the number
        except ValueError:
            # Skip files that don't match the integer pattern
            continue

        filepath = os.path.join(current_output_dir, filename)

        with open(filepath, "r") as f:
            lines = f.readlines()
            
            if not lines:
                continue

            last_line = lines[-1].strip()
            prefix = "Max difference:"
            
            if prefix in last_line:
                try:
                    value_str = last_line.split(prefix)[1].strip()
                    value = float(value_str)
                    
                    # Calculate Decimal Digits
                    if value > 0:
                        digits = math.ceil(-math.log10(value)) - 1
                    else:
                        digits = 7 if "dbl" not in current_output_dir else 15 # Cap for 0 difference (infinite precision)
                        
                    data_points.append({
                        'x': first_input,
                        'y_input': second_input,
                        'diff': value,
                        'digits': digits
                    })
                    
                except ValueError:
                    print(f"Could not parse float from file {filename}")
    
    if not data_points:
        return [], [], [], None

    # Sort by first input (x-axis) so the line graph connects points correctly
    data_points.sort(key=lambda k: k['x'])

    inputs_x = [d['x'] for d in data_points]
    max_diffs_y = [d['diff'] for d in data_points]
    decimal_digits_y = [d['digits'] for d in data_points]
    
    # Determine label for second input (check if it was constant or variable)
    unique_second_inputs = list(set(d['y_input'] for d in data_points))
    if len(unique_second_inputs) == 1:
        second_input_label = str(unique_second_inputs[0])
    else:
        second_input_label = "Various"

    print(f"Found {len(inputs_x)} valid data files.")
    return inputs_x, max_diffs_y, decimal_digits_y, second_input_label

def plot_data(x, y_diff, y_digits, suffix_name, batch_dir_name):
    """Generates two separate plot files with timestamps."""
    
    if not x:
        print("No data found to plot.")
        return

    # Create directory if needed
    final_dir = os.path.join(PLOTS_DIR, batch_dir_name)
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        print(f"Created directory: {final_dir}")
    
    suffix_str = f"_{suffix_name}" if suffix_name else ""

    # --- Graph 1: Max Difference ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_diff, 'b-o', markersize=4, linewidth=1)
    plt.title(f'Max Difference vs Filter Length ({suffix_name})')
    plt.xlabel('Filter Length')
    plt.ylabel('Max Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename1 = f"max_diff{suffix_str}.png"
    filepath1 = os.path.join(final_dir, filename1)
    plt.savefig(filepath1)
    print(f"Saved plot 1: {filepath1}")
    plt.close() # Clear memory

    # --- Graph 2: Decimal Digits ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_digits, 'r-s', markersize=4, linewidth=1)
    plt.title(f'Precision vs Filter Length ({suffix_name})')
    plt.xlabel('Filter Length')
    plt.ylabel('Decimal Digits')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    filename2 = f"decimal_digits{suffix_str}.png"
    filepath2 = os.path.join(final_dir, filename2)
    plt.savefig(filepath2)
    print(f"Saved plot 2: {filepath2}")
    plt.close() # Clear memory

if __name__ == "__main__":
    # Find all directories starting with "output"
    dirs = [d for d in os.listdir('output')] if 'output' in os.listdir('.') else []
    
    if not dirs:
        print("No 'output' directories found.")
    else:
        print(f"Found directories: {dirs}")
    
    for d in dirs:
        current_suffix = d
        full_path = os.path.join('output', d)

        try:
            subdirs = [s for s in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, s))]
        except OSError:
            print(f"Could not access {d}")
            continue

        if not subdirs:
            print(f"No timestamp subdirectories found in {d}")
            continue
        
        # Sort to handle them in chronological order if named by timestamp
        subdirs.sort()

        for batch_dir_name in subdirs:
            full_batch_path = os.path.join(full_path, batch_dir_name)
            print(f"\nProcessing batch: {batch_dir_name} inside {full_path}")
            
            x_data, diff_data, digits_data, sec_label = parse_results(full_batch_path)
            if x_data:
                # Pass the folder name (batch_dir_name) to be used in the filename
                plot_data(x_data, diff_data, digits_data, current_suffix, os.path.join(d, batch_dir_name))

import os
import re
import csv
import statistics
import sys

def extract_metrics(file_path):
    """
    Parses a log file to find CPU and GPU Throughput values.
    Returns a dict with 'cpu', 'gpu', and 'note'.
    """
    metrics = {'cpu': None, 'gpu': None, 'note': ''}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Extract CPU Throughput
        # Matches: "Average CPU Throughput: 10.257"
        cpu_pattern = r"Average CPU Throughput:\s+([\d\.]+)"
        cpu_match = re.search(cpu_pattern, content)
        if cpu_match:
            metrics['cpu'] = float(cpu_match.group(1))

        # 2. Extract GPU Throughput
        # Matches: "Average GPU Throughput: 16.962"
        gpu_pattern = r"Average GPU Throughput:\s+([\d\.]+)"
        gpu_match = re.search(gpu_pattern, content)
        if gpu_match:
            metrics['gpu'] = float(gpu_match.group(1))

        # 3. Check for Correctness / Errors
        if "GPU data is not correct!" in content:
            metrics['note'] = "INCORRECT DATA"
        elif "CUDA Error" in content:
            metrics['note'] = "CUDA ERROR"

        return metrics
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return metrics

def calculate_stats(values):
    """Helper to calculate avg and std_dev with outlier removal (min/max)."""
    if not values:
        return 0.0, 0.0, "No data"

    sorted_vals = sorted(values)
    
    # Filter outliers if we have enough data (>2 points)
    if len(sorted_vals) > 2:
        filtered_vals = sorted_vals[1:-1]
        note = f"(n={len(values)}, removed min/max)"
    else:
        filtered_vals = sorted_vals
        note = f"(n={len(values)})"

    avg = statistics.mean(filtered_vals)
    std_dev = statistics.stdev(filtered_vals) if len(filtered_vals) > 1 else 0.0
    
    return avg, std_dev, note

def process_logs(input_folder, output_csv):
    """
    Iterates through folder, extracts N-Body stats, saves to CSV.
    """
    data_points = []

    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist.")
        return

    print(f"Scanning folder: {input_folder}...")
    
    # Get all files
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Sort files to make CSV look orderly (optional)
    files.sort()

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        metrics = extract_metrics(file_path)
        
        # Only add if we found at least one metric
        if metrics['cpu'] is not None or metrics['gpu'] is not None:
            data_points.append({
                'filename': filename, 
                'cpu': metrics['cpu'], 
                'gpu': metrics['gpu'],
                'note': metrics['note']
            })

    if not data_points:
        print("No valid N-Body data found in logs.")
        return

    # --- Calculate Statistics ---
    
    # Extract valid CPU values (ignore Nones)
    cpu_values = [d['cpu'] for d in data_points if d['cpu'] is not None]
    cpu_avg, cpu_std, cpu_note = calculate_stats(cpu_values)

    # Extract valid GPU values
    gpu_values = [d['gpu'] for d in data_points if d['gpu'] is not None]
    gpu_avg, gpu_std, gpu_note = calculate_stats(gpu_values)

    # --- Write CSV ---
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Filename', 'CPU GInt/s', 'GPU GInt/s', 'Note']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for item in data_points:
                writer.writerow({
                    'Filename': item['filename'], 
                    'CPU GInt/s': item['cpu'] if item['cpu'] else "", 
                    'GPU GInt/s': item['gpu'] if item['gpu'] else "",
                    'Note': item['note']
                })

            # Separation row
            writer.writerow({})
            writer.writerow({'Filename': '--- STATISTICS ---'})
            
            # CPU Stats Row
            writer.writerow({
                'Filename': 'CPU Average', 
                'CPU GInt/s': f"{cpu_avg:.4f}",
                'Note': cpu_note
            })
            writer.writerow({
                'Filename': 'CPU Std Dev', 
                'CPU GInt/s': f"{cpu_std:.4f}"
            })

            # GPU Stats Row
            writer.writerow({
                'Filename': 'GPU Average', 
                'GPU GInt/s': f"{gpu_avg:.4f}",
                'Note': gpu_note
            })
            writer.writerow({
                'Filename': 'GPU Std Dev', 
                'GPU GInt/s': f"{gpu_std:.4f}"
            })

        print(f"Successfully processed {len(data_points)} logs.")
        print(f"CPU Stats: Avg={cpu_avg:.2f}, StdDev={cpu_std:.2f}")
        print(f"GPU Stats: Avg={gpu_avg:.2f}, StdDev={gpu_std:.2f}")
        print(f"Output saved to: {output_csv}")

    except IOError as e:
        print(f"Error writing to CSV: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_csv.py <input_folder_path> [output_csv_path]")
    else:
        in_folder = sys.argv[1]
        out_csv = sys.argv[2] if len(sys.argv) > 2 else "nbody_stats.csv"
        process_logs(in_folder, out_csv)

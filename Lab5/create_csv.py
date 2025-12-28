import os
import re
import csv
import statistics
import argparse
import sys

def extract_gpu_throughput(file_path):
    """
    Parses a log file to find the GPU Throughput value.
    Returns the throughput as a float, or None if not found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Logic: Find "Running GPU..." and capture the Throughput value that follows it.
        # We use a non-greedy match for the content between the header and the throughput line.
        pattern = r"Running GPU CLAHE reference\.\.\..*?Throughput:\s+([\d\.]+)\s+MPixels/s"
        
        # re.DOTALL allows the dot (.) to match newlines, handling multi-line search
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return float(match.group(1))
        return None
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_logs(input_folder, output_csv):
    """
    Iterates through the folder, extracts data, calculates stats, and saves to CSV.
    """
    data_points = []

    # 1. Iterate through files
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist.")
        return

    print(f"Scanning folder: {input_folder}...")
    
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        throughput = extract_gpu_throughput(file_path)
        
        if throughput is not None:
            data_points.append({'filename': filename, 'throughput': throughput})

    if not data_points:
        print("No valid GPU throughput data found in logs.")
        return

    # 2. Extract values for statistics
    values = [d['throughput'] for d in data_points]
    
    # 3. Filter outliers (Remove Min and Max)
    # We sort the values first
    sorted_values = sorted(values)
    
    if len(sorted_values) > 2:
        # Remove the smallest and largest
        filtered_values = sorted_values[1:-1]
        removed_info = f"(Removed min: {sorted_values[0]} and max: {sorted_values[-1]})"
    else:
        # If we have 2 or fewer items, we cannot remove min AND max and still have data
        filtered_values = sorted_values
        removed_info = "(Not enough data to remove outliers)"

    # 4. Calculate Statistics on filtered data
    if filtered_values:
        avg_throughput = statistics.mean(filtered_values)
        # stdev requires at least two data points
        std_dev_throughput = statistics.stdev(filtered_values) if len(filtered_values) > 1 else 0.0
    else:
        avg_throughput = 0.0
        std_dev_throughput = 0.0

    # 5. Write to CSV
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Filename', 'GPU Throughput (MPixels/s)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for item in data_points:
                writer.writerow({
                    'Filename': item['filename'], 
                    'GPU Throughput (MPixels/s)': item['throughput']
                })

            # Add an empty row for separation
            writer.writerow({})
            writer.writerow({'Filename': '--- STATISTICS (Excluding Min/Max) ---', 'GPU Throughput (MPixels/s)': ''})
            
            writer.writerow({
                'Filename': 'Average', 
                'GPU Throughput (MPixels/s)': f"{avg_throughput:.4f}"
            })
            writer.writerow({
                'Filename': 'Standard Deviation', 
                'GPU Throughput (MPixels/s)': f"{std_dev_throughput:.4f}"
            })
            writer.writerow({
                'Filename': 'Note', 
                'GPU Throughput (MPixels/s)': removed_info
            })

        print(f"Successfully processed {len(data_points)} logs.")
        print(f"Stats (Filtered): Avg={avg_throughput:.2f}, StdDev={std_dev_throughput:.2f}")
        print(f"Output saved to: {output_csv}")

    except IOError as e:
        print(f"Error writing to CSV: {e}")

if __name__ == "__main__":
    # You can hardcode paths here or pass them as arguments
    # Example usage: python script.py ./logs results.csv
    
    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <input_folder_path> [output_csv_path]")
        print("Example: python log_parser.py ./my_logs output.csv")
    else:
        in_folder = sys.argv[1]
        out_csv = sys.argv[2] if len(sys.argv) > 2 else "gpu_stats.csv"
        process_logs(in_folder, out_csv)
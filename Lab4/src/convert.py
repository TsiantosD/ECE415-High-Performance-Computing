import sys
import os
from PIL import Image

def convert_pgm_to_png(input_path):
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    try:
        # 1. Define Output Directory (One level up, then into Outputs)
        # using os.path.join ensures it works on Windows/Linux/Mac
        output_dir = os.path.join(os.getcwd(), "../Output")
        
        # 2. Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📂 Created directory: {output_dir}")

        # 3. Construct new filename
        # Extract filename (e.g. "output.pgm") and change extension to ".png"
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 4. Save
        with Image.open(input_path) as img:
            img.save(output_path)
            print(f"✔ Success: Saved image to {output_path}")
            
    except Exception as e:
        print(f"✘ Error converting {input_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <file.pgm>")
    else:
        convert_pgm_to_png(sys.argv[1])
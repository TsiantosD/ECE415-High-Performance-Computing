import sys
import os
from PIL import Image

def convert_pgm_to_png(input_path):
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    try:
        # 1. Determine Output Location
        # Since run.sh now puts the PGM inside ../Output/, 
        # we simply save the PNG in the SAME directory as the input PGM.
        output_dir = os.path.dirname(input_path)

        # 2. Construct Filename
        # Input: "../Output/fort_out.pgm" 
        # Base:  "fort_out"
        filename_only = os.path.basename(input_path)
        base_name = os.path.splitext(filename_only)[0]
        
        # Output: "../Output/fort_out.png"
        output_filename = f"{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 3. Save
        with Image.open(input_path) as img:
            img.save(output_path)
            print(f"✔ Success: PNG saved to {output_path}")
            
    except Exception as e:
        print(f"✘ Error converting {input_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <file.pgm>")
    else:
        convert_pgm_to_png(sys.argv[1])
#!/bin/bash

# --- Configuration ---
INPUT_FILE="$1"
# IMPORTANT: This must match the 'TARGET' name inside your Makefile
C_EXEC="./main" 
VENV_DIR="./venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
OUTPUT_DIR="../Output"

# --- 1. Validation ---
if [ -z "$INPUT_FILE" ]; then
    echo "Usage: ./run.sh <path_to_input.pgm>"
    exit 1
fi

# --- 2. Dynamic Filename Logic ---
# Extract filename: "../Images/fort.pgm" -> "fort.pgm"
FULL_FILENAME=$(basename -- "$INPUT_FILE")
# Remove extension: "fort.pgm" -> "fort"
BASENAME="${FULL_FILENAME%.*}"

# Prepare the Output Directory first
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "📂 Creating Output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Point the Output PGM directly to ../Output/
# Name: "../Output/fort_out.pgm"
OUTPUT_PGM="$OUTPUT_DIR/${BASENAME}_out.pgm"

# --- 3. Virtual Environment Setup ---
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating isolated Python virtual environment..."
    python3 -m venv "$VENV_DIR" 
fi

# --- 5. Install Libraries ---
$VENV_PYTHON -c "import PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "🔍 Installing Pillow..."
    $VENV_PYTHON -m pip install pillow > /dev/null
fi

# --- 6. C Compilation ---
echo "⚙️  Compiling C code..."
make

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed."
    exit 1
fi

# --- 7. Execution Pipeline ---
echo "🚀 Running CLAHE on $INPUT_FILE..."

# Run C binary. 
# C Program will write DIRECTLY to ../Output/fort_out.pgm
$C_EXEC "$INPUT_FILE" "$OUTPUT_PGM"

if [ -f "$OUTPUT_PGM" ]; then
    echo "🎨 Converting result to PNG..."
    
    # Pass "../Output/fort_out.pgm" to Python
    $VENV_PYTHON convert.py "$OUTPUT_PGM"
    
    # --- Cleanup ---
    # Optional: Delete the PGM (if you only want the PNG)
    # rm "$OUTPUT_PGM" 
    
    echo "✨ Done. Result located in $OUTPUT_DIR"
else
    echo "❌ C Program failed to generate output."
    exit 1
fi

#!/bin/bash

# --- Configuration ---
INPUT_FILE="$1"
OUTPUT_PGM="output.pgm"
# IMPORTANT: This must match the 'TARGET' name inside your Makefile
C_EXEC="main" 
VENV_DIR="./venv"
VENV_PYTHON="$VENV_DIR/bin/python3"

# --- 1. Validation ---
if [ -z "$INPUT_FILE" ]; then
    echo "Usage: ./run.sh <input_image.pgm>"
    exit 1
fi

# --- 2. Virtual Environment Setup ---
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating isolated Python virtual environment..."
    python3 -m venv "$VENV_DIR" --without-pip
fi

# --- 3. Pip Self-Repair ---
$VENV_PYTHON -m pip --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "🔧 Pip is missing. Bootstrapping pip into venv..."
    python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
    $VENV_PYTHON get-pip.py
    rm get-pip.py
    echo "✅ Pip installed successfully."
fi

# --- 4. Install Libraries ---
echo "🔍 Checking for Pillow library..."
$VENV_PYTHON -m pip install pillow > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install Pillow. Check internet connection."
    exit 1
fi

# --- 5. C Compilation (Using Makefile) ---
echo "⚙️  Compiling C code using Make..."
# We run 'make' (which runs the 'all' rule by default)
make

# Check if make succeeded
if [ $? -ne 0 ]; then
    echo "❌ Compilation failed. Check your Makefile."
    exit 1
fi

# --- 6. Execution Pipeline ---
echo "🚀 Running CLAHE C implementation..."
./$C_EXEC "$INPUT_FILE" "$OUTPUT_PGM"

if [ -f "$OUTPUT_PGM" ]; then
    echo "🎨 Converting result to PNG..."
    $VENV_PYTHON convert.py "$OUTPUT_PGM"
    
    # # --- Cleanup ---
    # echo "🧹 Cleaning up..."
    # rm "$OUTPUT_PGM"     # Remove intermediate image
    # make clean           # Remove executable and object files (.o)
    echo "✨ Done."
else
    echo "❌ C Program failed to generate output."
    exit 1
fi
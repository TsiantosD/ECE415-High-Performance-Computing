#!/bin/bash

# Usage message
if [ $# -lt 3 ]; then
    echo "Usage: $0 <input.grey> <width> <height> [output.png]"
    echo "Example: $0 output/image.grey 2267 2267"
    exit 1
fi

INPUT_FILE="$1"
WIDTH="$2"
HEIGHT="$3"
OUTPUT_FILE="$4"

# Default output file if not provided
if [ -z "$OUTPUT_FILE" ]; then
    BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
    OUTPUT_FILE="${BASENAME}.png"
fi

# Check file existence
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found!"
    exit 1
fi

# Convert .grey → .png
convert -depth 8 -size "${WIDTH}x${HEIGHT}" "GRAY:${INPUT_FILE}" "$OUTPUT_FILE"

# Verify success
if [ $? -eq 0 ]; then
    echo "✅ Successfully created $OUTPUT_FILE"
else
    echo "❌ Conversion failed"
    exit 1
fi

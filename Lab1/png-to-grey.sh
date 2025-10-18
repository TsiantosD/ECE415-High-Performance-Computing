#!/bin/bash

# Check if a filename was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <image.png|image.jpg>"
    exit 1
fi

INPUT_FILE="$1"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found!"
    exit 1
fi

# Get the filename without extension
BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')

# Output filename
OUTPUT_FILE="${BASENAME}.grey"

# Convert to grayscale raw format
convert "$INPUT_FILE" -colorspace Gray -depth 8 "gray:${OUTPUT_FILE}"

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo "✅ Successfully converted to ${OUTPUT_FILE}"
else
    echo "❌ Conversion failed"
    exit 1
fi

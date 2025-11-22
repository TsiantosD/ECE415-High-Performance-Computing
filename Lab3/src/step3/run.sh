#!/bin/bash

# ================= CONFIGURATION =================
# Command to run your program. 
# Change this to your actual executable path (e.g., "./my_solver")
PROGRAM="./convolution2d"

# Ask for suffix
read -p "Enter output directory suffix (optional): " SUFFIX

if [ -n "$SUFFIX" ]; then
    OUTPUT_DIR="output_${SUFFIX}"
else
    OUTPUT_DIR="output"
fi

START_INPUT=0
UPPER_LIMIT=50
FIXED_SECOND_INPUT=32
# =================================================

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created directory: $OUTPUT_DIR"
fi

echo "Starting execution loop (0 to $UPPER_LIMIT)..."

# Loop from START to UPPER_LIMIT - 1
for (( i=START_INPUT; i<UPPER_LIMIT; i++ ))
do
    # Generate timestamp (YearMonthDay_HourMinuteSecond)
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    
    # Construct filename: out_<first>_<second>_<timestamp>
    FILENAME="out_${i}_${FIXED_SECOND_INPUT}_${TIMESTAMP}"
    FILEPATH="${OUTPUT_DIR}/${FILENAME}"
    
    # Run the program with inputs from stdin and redirect both stdout and stderr to the file
    echo "$i $FIXED_SECOND_INPUT" | $PROGRAM > "$FILEPATH" 2>&1
    
    echo "Processed input $i..."
done

echo "Execution complete. Files saved in '$OUTPUT_DIR'."

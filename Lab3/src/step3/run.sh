#!/bin/bash

# ================= CONFIGURATION =================
# Command to run your program. 
# Change this to your actual executable path (e.g., "./my_solver")
PROGRAM="python3 mock_program.py"

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
    # Calculate the actual input value: 2 * i + 1
    INPUT_VAL=$((2 * i + 1))

    # Generate timestamp (YearMonthDay_HourMinuteSecond)
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    
    # Construct filename using the calculated input value
    FILENAME="out_${INPUT_VAL}_${FIXED_SECOND_INPUT}_${TIMESTAMP}"
    FILEPATH="${OUTPUT_DIR}/${FILENAME}"
    
    # Run the program with the calculated input and redirect
    echo "$INPUT_VAL $FIXED_SECOND_INPUT" | $PROGRAM > "$FILEPATH" 2>&1
    
    # Optional: Print status every 10 runs to avoid clutter
    if (( i % 10 == 0 )); then
        echo "Processed iteration $i (Input: $INPUT_VAL)..."
    fi
done

echo "Execution complete. Files saved in '$OUTPUT_DIR'."
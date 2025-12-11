#!/bin/bash

# ==========================================
# Hardcoded Configuration
# ==========================================
IMAGE="senator.pgm"       # Default Input Image
ITERATIONS=25          # Default Iterations
CHECK_OUTPUT=0         # 0 = Speed (No checks), 1 = Verify correctness

# ==========================================
# 1. Find Kernels (Needed to define range)
# ==========================================
# We must find the files to know how many there are (END)
KERNELS=($(find src -maxdepth 1 -name "*.cu" | sort))
NUM_KERNELS=${#KERNELS[@]}

if [ $NUM_KERNELS -eq 0 ]; then
    echo "Error: No .cu files found in src/."
    exit 1
fi

# ==========================================
# Parse Range Flag (-r)
# ==========================================
START=1
END=$NUM_KERNELS

while getopts "r:" opt; do
    case $opt in
        r) 
            RANGE_ARG=$OPTARG
            if [[ "$RANGE_ARG" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                START=${BASH_REMATCH[1]}
                END=${BASH_REMATCH[2]}
            else
                echo "Error: Invalid range format. Use '1-3'."
                exit 1
            fi
            ;;
        *) echo "Usage: $0 [-r start-end]"; exit 1 ;;
    esac
done

# Ensure run.sh is executable
chmod +x run.sh

# ==========================================
# Execution Loop
# ==========================================
for (( i=START; i<=END; i++ )); do
    # Calls run.sh with:
    # -n 10 (Iterations)
    # -c 0  (Disable output check)
    # -i test.pgm (Image)
    # -f $i (Index of the file to use)
    ./run.sh -n "$ITERATIONS" -c "$CHECK_OUTPUT" -i "$IMAGE" -f "$i"
done
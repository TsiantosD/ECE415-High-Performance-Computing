#!/bin/bash

# ==========================================
# Hardcoded Configuration
# ==========================================
IMAGE="senator.pgm"       # Default Input Image
ITERATIONS=25          # Default Iterations
CHECK_OUTPUT=0         # 0 = Speed (No checks), 1 = Verify correctness

# ==========================================
# Parse Range Flag (-r)
# ==========================================
RANGE_ARG=""

while getopts "r:" opt; do
    case $opt in
        r) RANGE_ARG=$OPTARG ;;
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
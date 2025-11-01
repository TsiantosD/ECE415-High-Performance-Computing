#!/usr/bin/env bash
# testbench.sh — automated correctness test for sequential vs parallel K-Means

# Default settings
INPUT_FILE="./src/Image_data/texture17695.bin"
NUM_MEANS=2000
THREADS_LIST=(1 4 8 14 28 56)
RUNS_PER_THREAD=12
EXEC_SEQ="./src/seq_main"
EXEC_PAR="./src/par_main"
WORK_DIR="./testbench_work"
IMAGE_DATA_DIR="./src/Image_data"

# --- Step 0: Build the executables ---
echo "=== Building executables ==="
cd ./src || exit 1
make
cd .. || exit 1
echo "=== Build complete ==="
echo

# Create working directory to store test outputs
mkdir -p "$WORK_DIR"

INPUT_BASENAME=$(basename "$INPUT_FILE")
CLUSTER_FILE="$IMAGE_DATA_DIR/${INPUT_BASENAME}.cluster_centres"
MEMBERSHIP_FILE="$IMAGE_DATA_DIR/${INPUT_BASENAME}.membership"

# --- Step 1: Run sequential algorithm ---
echo "=== Running sequential algorithm on $INPUT_FILE ==="
$EXEC_SEQ -i "$INPUT_FILE" -n "$NUM_MEANS" -o -b

# Move outputs to golden reference in working directory
GOLD_CLUSTER="$WORK_DIR/golden.${INPUT_BASENAME}.cluster_centres"
GOLD_MEMBERSHIP="$WORK_DIR/golden.${INPUT_BASENAME}.membership"

mv "$CLUSTER_FILE" "$GOLD_CLUSTER"
mv "$MEMBERSHIP_FILE" "$GOLD_MEMBERSHIP"
echo "Golden files saved: $GOLD_CLUSTER, $GOLD_MEMBERSHIP"
echo

# --- Step 2: Run parallel algorithm with various threads ---
for THREADS in "${THREADS_LIST[@]}"; do
    echo "=== Running parallel algorithm with $THREADS threads ==="

    export OMP_NUM_THREADS=$THREADS

    for RUN_NUM in $(seq 1 $RUNS_PER_THREAD); do
        echo "--- Run $RUN_NUM / $RUNS_PER_THREAD ---"
        $EXEC_PAR -i "$INPUT_FILE" -n "$NUM_MEANS" -o -b

        # Generated output files are in IMAGE_DATA_DIR
        PAR_CLUSTER="$CLUSTER_FILE"
        PAR_MEMBERSHIP="$MEMBERSHIP_FILE"

        # Compare outputs with golden files
        CLUSTER_DIFF=$(diff -q "$PAR_CLUSTER" "$GOLD_CLUSTER")
        MEMBERSHIP_DIFF=$(diff -q "$PAR_MEMBERSHIP" "$GOLD_MEMBERSHIP")

        if [[ -z "$CLUSTER_DIFF" && -z "$MEMBERSHIP_DIFF" ]]; then
            echo "✅ Run $RUN_NUM: Outputs match the golden reference."
        else
            echo "❌ Run $RUN_NUM: Outputs differ from golden reference!"
        fi

        # Move the outputs to a subdirectory to keep them organized
        RUN_DIR="$WORK_DIR/par_${THREADS}threads_run${RUN_NUM}"
        mkdir -p "$RUN_DIR"
        mv "$PAR_CLUSTER" "$PAR_MEMBERSHIP" "$RUN_DIR/"
    done

    echo
done

echo "=== Testbench completed ==="

#!/usr/bin/env bash
# testbench.sh — automated correctness test using run.sh
# Compares outputs numerically and parses runtime metrics

# --- Configuration ---
INPUT_FILE="./src/Image_data/texture17695.bin"
NUM_MEANS=2000
THREADS_LIST=(1 4 8 14 28 56)
RUNS_PER_THREAD=12
WORK_DIR="./testbench_work"
TOL=1e-5  # numeric tolerance for comparison
RUN_SCRIPT="./run.sh"

IMAGE_DATA_DIR="./src/Image_data"

# --- Helper function: numerical comparison with tolerance ---
compare_files() {
    local file1=$1
    local file2=$2
    local tol=$3

    awk -v eps="$tol" '
    NR==FNR {for(i=1;i<=NF;i++) a[NR,i]=$i; next}
    {
        for(i=1;i<=NF;i++) {
            diff = $i - a[FNR,i]
            if(diff<0) diff=-diff
            if(diff > eps) exit 1
        }
    }' "$file1" "$file2"

    return $?  # 0 = match within tolerance, 1 = differs
}

# --- Create working directory ---
mkdir -p "$WORK_DIR"

INPUT_BASENAME=$(basename "$INPUT_FILE")
CLUSTER_FILE="$IMAGE_DATA_DIR/${INPUT_BASENAME}.cluster_centres"
MEMBERSHIP_FILE="$IMAGE_DATA_DIR/${INPUT_BASENAME}.membership"

# --- Step 1: Run sequential algorithm via run.sh ---
echo "=== Running sequential algorithm ==="
SEQ_LOG="$WORK_DIR/seq_run.log"

bash "$RUN_SCRIPT" -s -i "$INPUT_FILE" -n "$NUM_MEANS" | tee "$SEQ_LOG"

# Move outputs to golden reference
GOLD_CLUSTER="$WORK_DIR/golden.${INPUT_BASENAME}.cluster_centres"
GOLD_MEMBERSHIP="$WORK_DIR/golden.${INPUT_BASENAME}.membership"

mv "$CLUSTER_FILE" "$GOLD_CLUSTER"
mv "$MEMBERSHIP_FILE" "$GOLD_MEMBERSHIP"
echo "Golden files saved: $GOLD_CLUSTER, $GOLD_MEMBERSHIP"

# Parse runtime from sequential log (last "Execution time" line)
SEQ_RUNTIME=$(grep -oP 'Execution time: \K[0-9]+' "$SEQ_LOG")
echo "Sequential runtime: ${SEQ_RUNTIME}s"
echo

# --- Step 2: Run parallel algorithm via run.sh ---
for THREADS in "${THREADS_LIST[@]}"; do
    echo "=== Running parallel algorithm with $THREADS threads ==="
    for RUN_NUM in $(seq 1 $RUNS_PER_THREAD); do
        echo "--- Run $RUN_NUM / $RUNS_PER_THREAD ---"
        PAR_LOG="$WORK_DIR/par_${THREADS}threads_run${RUN_NUM}.log"

        # Call run.sh in parallel mode with threads
        bash "$RUN_SCRIPT" -p -i "$INPUT_FILE" -n "$NUM_MEANS" -t "$THREADS" | tee "$PAR_LOG"

        # Files generated in IMAGE_DATA_DIR
        PAR_CLUSTER="$CLUSTER_FILE"
        PAR_MEMBERSHIP="$MEMBERSHIP_FILE"

        # Compare outputs numerically
        compare_files "$GOLD_CLUSTER" "$PAR_CLUSTER" "$TOL"
        CLUSTER_OK=$?
        compare_files "$GOLD_MEMBERSHIP" "$PAR_MEMBERSHIP" "$TOL"
        MEMBERSHIP_OK=$?

        if [[ $CLUSTER_OK -eq 0 && $MEMBERSHIP_OK -eq 0 ]]; then
            echo "✅ Run $RUN_NUM: Outputs match golden reference (tolerance $TOL)"
        else
            echo "❌ Run $RUN_NUM: Outputs differ beyond tolerance $TOL!"
        fi

        # Parse runtime from run.sh log
        PAR_RUNTIME=$(grep -oP 'Execution time: \K[0-9]+' "$PAR_LOG")
        echo "Runtime: ${PAR_RUNTIME}s"

        # Move outputs to organized directory
        RUN_DIR="$WORK_DIR/par_${THREADS}threads_run${RUN_NUM}"
        mkdir -p "$RUN_DIR"
        mv "$PAR_CLUSTER" "$PAR_MEMBERSHIP" "$RUN_DIR/"
    done
    echo
done

echo "=== Testbench completed ==="

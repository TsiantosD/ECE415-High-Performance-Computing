#!/usr/bin/env bash
# run.sh — Wrapper to run sequential or parallel K-Means

# Default values
EXEC_TYPE=""
INPUT_FILE="./src/Image_data/texture17695.bin"
NUM_MEANS=2000
NUM_THREADS=""
BASE_DIR="$(dirname "$0")"
METRICS_DIR="$BASE_DIR/metrics"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--seq)
            EXEC_TYPE="seq"
            shift ;;
        -p|--par)
            EXEC_TYPE="par"
            shift ;;
        -i)
            INPUT_FILE="$2"
            shift 2 ;;
        -n)
            NUM_MEANS="$2"
            shift 2 ;;
        -t)
            NUM_THREADS="$2"
            shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-s|--seq | -p|--par] [-i input_file] [-n num_means] [-t num_threads]"
            echo "  -s, --seq        Run sequential version"
            echo "  -p, --par        Run parallel version"
            echo "  -i <file>        Input file (default: texture17695.bin)"
            echo "  -n <num>         Number of means (default: 2000)"
            echo "  -t <num>         Number of threads (parallel only)"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Try '$0 --help' for usage."
            exit 1 ;;
    esac
done

# --- Run make ---
cd ./src
make
cd ..

# --- Determine execution type ---
if [[ -z "$EXEC_TYPE" ]]; then
    echo "Choose which version to run:"
    select opt in "Sequential" "Parallel"; do
        case $opt in
            "Sequential") EXEC_TYPE="seq"; break ;;
            "Parallel") EXEC_TYPE="par"; break ;;
            *) echo "Invalid choice";;
        esac
    done
fi

# --- Ask for input file if not given ---
if [[ ! -f "$INPUT_FILE" ]]; then
    echo -e "\nAvailable input files:"
    ls ./src/Image_data
    read -p "Enter input filename [default: texture17695.bin]: " USER_FILE
    if [[ -n "$USER_FILE" ]]; then
        INPUT_FILE="./src/Image_data/$USER_FILE"
    else
        INPUT_FILE="./src/Image_data/texture17695.bin"
    fi
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

# --- Ask for means if not specified ---
if [[ -z "$NUM_MEANS" ]]; then
    read -p "Enter number of means [default: 2000]: " USER_MEANS
    NUM_MEANS=${USER_MEANS:-2000}
fi

# --- Binary flag ---
BINARY_FLAG=""
if [[ "$INPUT_FILE" == *.bin ]]; then
    BINARY_FLAG="-b"
fi

# --- Executable selection ---
if [[ "$EXEC_TYPE" == "seq" ]]; then
    EXEC="./src/seq_main"
elif [[ "$EXEC_TYPE" == "par" ]]; then
    EXEC="./src/par_main"
else
    echo "❌ Invalid execution type."
    exit 1
fi

if [[ ! -x "$EXEC" ]]; then
    echo "❌ Error: Executable '$EXEC' not found or not executable."
    exit 1
fi

# --- Handle threads (parallel only) ---
THREAD_LABEL="seq"
if [[ "$EXEC_TYPE" == "par" ]]; then
    if [[ -z "$NUM_THREADS" ]]; then
        read -p "Enter number of threads [default: all available]: " USER_THREADS
        NUM_THREADS="$USER_THREADS"
    fi

    if [[ -n "$NUM_THREADS" ]]; then
        export OMP_NUM_THREADS=$NUM_THREADS
        THREAD_LABEL=par/"${NUM_THREADS}-threads"
    else
        THREAD_LABEL="par/all-threads"
    fi
else
    THREAD_LABEL="seq"
fi

# --- Create log directory ---
OUT_DIR="$METRICS_DIR/$THREAD_LABEL"
mkdir -p "$OUT_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOGFILE="$OUT_DIR/${TIMESTAMP}.log"

# --- Run summary ---
echo -e "\n🚀 Running $EXEC_TYPE version..."
echo "Input file:   $INPUT_FILE"
echo "Num means:    $NUM_MEANS"
echo "Binary flag:  $BINARY_FLAG"
if [[ "$EXEC_TYPE" == "par" ]]; then
    echo "OMP threads:  ${NUM_THREADS:-all available}"
fi
echo "Saving log:   $LOGFILE"
echo "---------------------------------------"

# --- Run and time execution ---
START_TIME=$(date +%s)
"$EXEC" -i "$INPUT_FILE" -n "$NUM_MEANS" -o $BINARY_FLAG | tee "$LOGFILE"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "✅ Done. Log saved to: $LOGFILE"

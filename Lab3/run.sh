#!/bin/bash

# ================= CONFIGURATION =================
# Name of the executable file *inside* the step folder
PROGRAM_NAME="convolution2d" 

# --- Input Configuration ---
# Default image size (second input, must be power of two)
DEFAULT_SIZE=32 
# Default range for the first input (filter radius base)
DEFAULT_START=0
DEFAULT_LIMIT=15 
# =================================================

# Initialize variables
STEP_DIR=""
USE_DOUBLES=false
DISABLE_FMAD=false

# Variables for input range
START_INPUT="" # Start unset to detect if option was provided
LIMIT_INPUT="" # Limit unset to detect if option was provided
SIZE_INPUT=""  # Size unset to detect if option was provided

# Repeat Mode variables
REPEAT_COUNT=0
TARGET_INPUT=""

# --- Directory Constants ---
SOURCE_DIR="src"
OUTPUT_BASE_DIR="output"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message and exit"
    echo "  --step <dir_name>     Specify the step directory (e.g., step1). If omitted, scans '$SOURCE_DIR/' and prompts."
    echo "  --use-doubles         Compile with USE_DOUBLES=1 (Adds '_dbl' to output folder)"
    echo "  --disable-fmad        Compile with FMAD=false (Adds '_nofmad' to output folder)"
    echo "  --start <N>           Standard Mode: Start of filter radius base range (inclusive). Default: $DEFAULT_START."
    echo "  --limit <N>           Standard Mode: Limit/End of filter radius base range (exclusive). Default: $DEFAULT_LIMIT."
    echo "  --size <N>            Standard Mode: The fixed image size (second input). Default: $DEFAULT_SIZE."
    echo "  --repeat <N>          Repeat Mode: Run a specific input N times."
    echo "  --target <input>      Repeat Mode: The specific filter radius base to run."
    exit 0
}

# Function to check if a number is a power of two
is_power_of_two() {
    local n=$1
    if [ "$n" -le 0 ]; then
        return 1
    fi
    # Check if n AND (n-1) is 0
    # Bash arithmetic context uses $ notation for variables
    if [ $((n & (n - 1))) -eq 0 ]; then
        return 0 # Is power of two
    else
        return 1 # Not power of two
    fi
}

# 1. PARSE ARGUMENTS
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            usage
            ;;
        --step)
            STEP_DIR="$2"
            shift 2
            ;;
        --use-doubles)
            USE_DOUBLES=true
            shift
            ;;
        --disable-fmad)
            DISABLE_FMAD=true
            shift
            ;;
        --start)
            START_INPUT="$2"
            shift 2
            ;;
        --limit)
            LIMIT_INPUT="$2"
            shift 2
            ;;
        --size)
            SIZE_INPUT="$2"
            shift 2
            ;;
        --repeat)
            REPEAT_COUNT="$2"
            shift 2
            ;;
        --target)
            TARGET_INPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ "$REPEAT_COUNT" -gt 0 ]; then
    # --- Repeat Mode Check ---
    if [ -z "$TARGET_INPUT" ]; then
        echo "Error: You must specify --target <input> when using --repeat."
        exit 1
    fi
    # In repeat mode, size must still be set, use default if not specified
    if [ -z "$SIZE_INPUT" ]; then
        SIZE_INPUT=$DEFAULT_SIZE
        echo "Note: Image size (--size) not specified in Repeat Mode. Using default: $SIZE_INPUT."
    fi
fi

# 2. HANDLE STEP SELECTION (Prompt if missing)
if [ -z "$STEP_DIR" ]; then
    # Find ALL directories inside the SOURCE_DIR ('src/') excluding the source directory itself
    OPTIONS=($(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort))
    
    if [ ${#OPTIONS[@]} -eq 0 ]; then
        echo "Error: No step directories found in '$SOURCE_DIR/'."
        exit 1
    fi
    
    if [ ${#OPTIONS[@]} -eq 1 ]; then
        STEP_DIR=${OPTIONS[0]}
        echo "Auto-selected only available step: $STEP_DIR"
    else
        echo "Multiple directories found in '$SOURCE_DIR/'. Please select one:"
        PS3="Enter number: "
        select opt in "${OPTIONS[@]}"; do
            if [ -n "$opt" ]; then
                STEP_DIR=$opt
                break
            else
                echo "Invalid selection."
            fi
        done
    fi
fi

# The full path to the step directory (e.g., src/step6)
FULL_STEP_DIR="$SOURCE_DIR/$STEP_DIR"
TARGET_EXEC="./$FULL_STEP_DIR/$PROGRAM_NAME"


# 3. INTERACTIVE PROMPTING FOR STANDARD MODE INPUTS
if [ "$REPEAT_COUNT" -eq 0 ]; then
    echo "--- Standard Mode Input Configuration ---"

    # Prompt for SIZE
    if [ -z "$SIZE_INPUT" ]; then
        while true; do
            read -r -p "Enter image size (must be a power of two, default: $DEFAULT_SIZE): " INPUT
            INPUT=${INPUT:-$DEFAULT_SIZE} # Use default if empty
            if ! [[ "$INPUT" =~ ^[0-9]+$ ]]; then
                echo "Invalid input. Please enter a number."
            elif ! is_power_of_two "$INPUT"; then
                echo "Invalid input. Size must be a power of two."
            else
                SIZE_INPUT=$INPUT
                break
            fi
        done
    fi
    
    # Prompt for START
    if [ -z "$START_INPUT" ]; then
        while true; do
            read -r -p "Enter start of filter radius base (inclusive, default: $DEFAULT_START): " INPUT
            INPUT=${INPUT:-$DEFAULT_START}
            if ! [[ "$INPUT" =~ ^[0-9]+$ ]]; then
                echo "Invalid input. Please enter a number."
            else
                START_INPUT=$INPUT
                break
            fi
        done
    fi

    # Prompt for LIMIT
    if [ -z "$LIMIT_INPUT" ]; then
        while true; do
            read -r -p "Enter limit of filter radius base (exclusive, default: $DEFAULT_LIMIT): " INPUT
            INPUT=${INPUT:-$DEFAULT_LIMIT}
            if ! [[ "$INPUT" =~ ^[0-9]+$ ]]; then
                echo "Invalid input. Please enter a number."
            elif [ "$INPUT" -le "$START_INPUT" ]; then
                echo "Limit must be greater than Start ($START_INPUT)."
            else
                LIMIT_INPUT=$INPUT
                break
            fi
        done
    fi
fi

# 4. CONSTRUCT SUFFIX BASED ON OPTIONS
SUFFIX="${STEP_DIR}"

if [ "$USE_DOUBLES" = true ]; then
    SUFFIX="${SUFFIX}_dbl"
fi

if [ "$DISABLE_FMAD" = true ]; then
    SUFFIX="${SUFFIX}_nofmad"
fi

# Output directory path (e.g., output/step6_dbl)
OUTPUT_DIR="$OUTPUT_BASE_DIR/${SUFFIX}"

# Generate Batch Timestamp ONCE for the folder
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${BATCH_TIMESTAMP}"

# 5. COMPILE (RUN MAKE)
echo "----------------------- COMPILATION -----------------------"
echo "Preparing '$FULL_STEP_DIR'..."
printf "Configuration: Doubles=$USE_DOUBLES, FMAD_Disabled=$DISABLE_FMAD\n"

# Construct Make command. Compiling from the step directory (src/stepX)
MAKE_CMD="make -C $FULL_STEP_DIR"

if [ "$USE_DOUBLES" = true ]; then
    MAKE_CMD="$MAKE_CMD USE_DOUBLES=1"
fi

if [ "$DISABLE_FMAD" = true ]; then
    MAKE_CMD="$MAKE_CMD FMAD=false"
fi

printf "\nRunning: make -C $FULL_STEP_DIR clean\n"
make -C "$FULL_STEP_DIR" clean
printf "\nRunning: $MAKE_CMD\n"
$MAKE_CMD

# Check if make was successful
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi

if [ ! -f "$TARGET_EXEC" ]; then
    echo "Error: Executable '$TARGET_EXEC' not found after make."
    exit 1
fi

# 6. EXECUTION LOOP

# Create output directory
if [ ! -d "$RUN_OUTPUT_DIR" ]; then
    mkdir -p "$RUN_OUTPUT_DIR"
fi

echo "-----------------------------------------------------------"
printf "\n\n------------------------ EXECUTION ------------------------\n"
echo "Starting execution. Output folder: $RUN_OUTPUT_DIR"


# BRANCH: REPEAT MODE vs STANDARD MODE
if [ "$REPEAT_COUNT" -gt 0 ]; then

    # Calculate actual input: 2 * TARGET_INPUT + 1 (Filter Length)
    INPUT_VAL=$((2 * TARGET_INPUT + 1))
    
    echo "Mode: Repeating filter radius base $TARGET_INPUT (Actual Length: $INPUT_VAL) with size $SIZE_INPUT for $REPEAT_COUNT times."

    for (( j=1; j<=REPEAT_COUNT; j++ )); do
        # Filename: out_(filter_length)_(image_size)_rep(j)
        FILENAME="out_${INPUT_VAL}_${SIZE_INPUT}_rep${j}"
        FILEPATH="${RUN_OUTPUT_DIR}/${FILENAME}"
        
        # Pipe inputs: (filter_radius_length) (image_size)
        echo "$INPUT_VAL $SIZE_INPUT" | "$TARGET_EXEC" > "$FILEPATH" 2>&1
        
        echo -ne "Run $j/$REPEAT_COUNT -> $FILENAME\r"
    done

else
    # --- Standard Mode ---
    
    # Generate array of inputs based on --start (inclusive) and --limit (exclusive)
    # The validity of the range is already checked during prompting if not provided as option.
    INPUTS_ARRAY=($(seq "$START_INPUT" $((LIMIT_INPUT - 1))))

    echo "Mode: Standard. Filter Radius Base range: $START_INPUT to $((LIMIT_INPUT - 1)). Fixed Size: $SIZE_INPUT."
    
    # Check if the range is valid (necessary if options were provided but were invalid)
    if [ "$START_INPUT" -ge "$LIMIT_INPUT" ]; then
        echo "Error: Invalid range specified (Start: $START_INPUT, Limit: $LIMIT_INPUT). Exiting."
        exit 1
    fi

    for i in "${INPUTS_ARRAY[@]}"; do
        # Calculate actual input: 2 * i + 1 (Filter Length)
        INPUT_VAL=$((2 * i + 1))
        
        # Filename: out_(filter_length)_(image_size)
        FILENAME="out_${INPUT_VAL}_${SIZE_INPUT}"
        FILEPATH="${RUN_OUTPUT_DIR}/${FILENAME}"
        
        # Run program with inputs piped to stdin: (filter_radius_length) (image_size)
        echo "$INPUT_VAL $SIZE_INPUT" | "$TARGET_EXEC" > "$FILEPATH" 2>&1
        
        # Simple progress indicator
        echo -ne "Processed filter radius base $i (Actual Length: $INPUT_VAL) -> $FILENAME\r"
    done
fi

echo -e "\nExecution complete. All output saved to $RUN_OUTPUT_DIR"
echo "-----------------------------------------------------------"
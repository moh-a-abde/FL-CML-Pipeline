#!/bin/bash

# Script to run Ray Tune for XGBoost hyperparameter optimization
# with consistent preprocessing to fix disconnection between tuning and FL phases

# Default values
DATA_FILE="data/received/final_dataset.csv"
NUM_SAMPLES=150
CPUS_PER_TRIAL=2
OUTPUT_DIR="./tune_results"
TRAIN_FILE=""
TEST_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --train-file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --cpus-per-trial)
            CPUS_PER_TRIAL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

echo "Starting Ray Tune XGBoost hyperparameter optimization with consistent preprocessing..."

# Step 1: Create global feature processor for consistent preprocessing
echo "Step 1: Creating global feature processor..."

# Determine which data file to use for the global processor
PROCESSOR_DATA_FILE="$DATA_FILE"
if [[ -n "$TRAIN_FILE" && -f "$TRAIN_FILE" ]]; then
    PROCESSOR_DATA_FILE="$TRAIN_FILE"
fi

python create_global_processor.py \
    --data-file "$PROCESSOR_DATA_FILE" \
    --output-dir "outputs" \
    --force

if [ $? -ne 0 ]; then
    echo "Failed to create global feature processor. Exiting."
    exit 1
fi

echo "Global feature processor created successfully."

# Step 2: Run Ray Tune with the updated script
echo "Step 2: Running Ray Tune optimization..."

# Build the Python command based on available files
PYTHON_CMD="python ray_tune_xgboost_updated.py"
PYTHON_CMD="$PYTHON_CMD --num-samples $NUM_SAMPLES"
PYTHON_CMD="$PYTHON_CMD --cpus-per-trial $CPUS_PER_TRIAL"
PYTHON_CMD="$PYTHON_CMD --output-dir \"$OUTPUT_DIR\""

if [[ -n "$TRAIN_FILE" && -n "$TEST_FILE" && -f "$TRAIN_FILE" && -f "$TEST_FILE" ]]; then
    echo "Using separate train and test files"
    PYTHON_CMD="$PYTHON_CMD --train-file \"$TRAIN_FILE\" --test-file \"$TEST_FILE\""
else
    echo "Using single data file: $DATA_FILE"
    PYTHON_CMD="$PYTHON_CMD --data-file \"$DATA_FILE\""
fi

echo "Executing: $PYTHON_CMD"
eval $PYTHON_CMD

if [ $? -ne 0 ]; then
    echo "Ray Tune optimization failed. Exiting."
    exit 1
fi

echo "Ray Tune optimization completed successfully."

# Step 3: Generate updated parameters file
echo "Step 3: Generating tuned parameters file..."
python use_tuned_params.py

if [ $? -ne 0 ]; then
    echo "Failed to generate tuned parameters. Continuing anyway."
fi

echo "Ray Tune with consistent preprocessing completed!"
echo "Global feature processor is available at: outputs/global_feature_processor.pkl"
echo "Tuned parameters are available at: $OUTPUT_DIR/best_params.json" 
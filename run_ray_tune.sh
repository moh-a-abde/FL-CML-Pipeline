#!/bin/bash

# This script runs XGBoost hyperparameter tuning using Ray Tune

# Default values
TRAIN_FILE=""
TEST_FILE=""
NUM_SAMPLES=10
CPUS_PER_TRIAL=1
GPU_FRACTION=""
OUTPUT_DIR="./tune_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
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
    --gpu-fraction)
      GPU_FRACTION="--gpu-fraction $2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if required files are provided
if [ -z "$TRAIN_FILE" ]; then
  echo "Error: --train-file is required"
  echo "Usage: $0 --train-file <path_to_train_csv> --test-file <path_to_test_csv> [--num-samples <number>] [--cpus-per-trial <number>] [--gpu-fraction <float>] [--output-dir <path>]"
  exit 1
fi

if [ -z "$TEST_FILE" ]; then
  echo "Error: --test-file is required"
  echo "Usage: $0 --train-file <path_to_train_csv> --test-file <path_to_test_csv> [--num-samples <number>] [--cpus-per-trial <number>] [--gpu-fraction <float>] [--output-dir <path>]"
  exit 1
fi

# Print hyperparameter tuning settings
echo "===== XGBoost Hyperparameter Tuning with Ray Tune ====="
echo "Training data file: $TRAIN_FILE"
echo "Test data file: $TEST_FILE"
echo "Number of hyperparameter samples: $NUM_SAMPLES"
echo "CPUs per trial: $CPUS_PER_TRIAL"
if [ -n "$GPU_FRACTION" ]; then
  echo "GPU fraction: $GPU_FRACTION"
else
  echo "GPU: Not used"
fi
echo "Output directory: $OUTPUT_DIR"
echo "======================================================="

# Run the Ray Tune script
python ray_tune_xgboost.py \
  --train-file "$TRAIN_FILE" \
  --test-file "$TEST_FILE" \
  --num-samples "$NUM_SAMPLES" \
  --cpus-per-trial "$CPUS_PER_TRIAL" \
  $GPU_FRACTION \
  --output-dir "$OUTPUT_DIR"

# Check if the tuning completed successfully
if [ $? -eq 0 ]; then
  echo "===== Hyperparameter tuning completed successfully ====="
  echo "Best parameters saved to: $OUTPUT_DIR/best_params.json"
  echo "Best model saved to: $OUTPUT_DIR/best_model.json"
  echo "======================================================="
else
  echo "===== Hyperparameter tuning failed ====="
fi 
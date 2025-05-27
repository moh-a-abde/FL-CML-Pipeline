#!/bin/bash

# Script to run Ray Tune for XGBoost hyperparameter optimization
# with consistent preprocessing to fix disconnection between tuning and FL phases

echo "Starting Ray Tune XGBoost hyperparameter optimization with consistent preprocessing..."

# Step 1: Create global feature processor for consistent preprocessing
echo "Step 1: Creating global feature processor..."
python create_global_processor.py \
    --data-file "data/received/final_dataset.csv" \
    --output-dir "outputs" \
    --force

if [ $? -ne 0 ]; then
    echo "Failed to create global feature processor. Exiting."
    exit 1
fi

echo "Global feature processor created successfully."

# Step 2: Run Ray Tune with the updated script
echo "Step 2: Running Ray Tune optimization..."
python ray_tune_xgboost_updated.py \
    --data-file "data/received/final_dataset.csv" \
    --num-samples 50 \
    --cpus-per-trial 2 \
    --output-dir "./tune_results"

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
echo "Tuned parameters are available at: tune_results/best_params.json" 
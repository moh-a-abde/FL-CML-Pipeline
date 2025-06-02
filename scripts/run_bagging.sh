#!/bin/bash
set -e

# Change to the project root directory (parent of scripts directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/.."

echo "Starting Federated Learning with Bagging Strategy"
echo "=================================================="

# Ensure the global feature processor is created before starting federated learning
echo "Step 1: Ensuring global feature processor is created..."
python src/core/create_global_processor.py \
    --data-file "data/received/final_dataset.csv" \
    --output-dir "outputs" \
    --force

if [ $? -ne 0 ]; then
    echo "Error creating global feature processor. Exiting."
    exit 1
fi

echo "✓ Global feature processor ready at outputs/global_feature_processor.pkl"
echo ""

# Run the complete federated learning pipeline using the bagging experiment configuration
echo "Step 2: Running federated learning simulation with bagging configuration..."
python run.py +experiment=bagging

if [ $? -ne 0 ]; then
    echo "Error running federated learning simulation. Exiting."
    exit 1
fi

echo ""
echo "✓ Federated learning with bagging strategy completed successfully!"
echo "Results saved to: outputs/"
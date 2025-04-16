# Ray Tune XGBoost Hyperparameter Optimization

This guide explains how to use Ray Tune for optimizing the hyperparameters of the XGBoost classifier in our federated learning pipeline.

## Overview

Ray Tune is a powerful library for hyperparameter tuning that can efficiently find optimal parameters for machine learning models. We've integrated Ray Tune with our XGBoost implementation to achieve better model performance through automated hyperparameter search.

## Requirements

Make sure you have all the required packages installed:

```bash
pip install -r requirements.txt
```

## Hyperparameter Tuning Process

### 1. Run Ray Tune Optimization

Use the `run_ray_tune.sh` script to start the optimization process:

```bash
# Basic usage
bash run_ray_tune.sh --data-file path/to/your/data.csv

# Advanced usage with more options
bash run_ray_tune.sh \
  --data-file path/to/your/data.csv \
  --num-samples 20 \
  --cpus-per-trial 2 \
  --gpu-fraction 0.2 \
  --output-dir ./my_tuning_results
```

#### Available Options

- `--data-file`: Path to the CSV data file (required)
- `--num-samples`: Number of hyperparameter combinations to try (default: 10)
- `--cpus-per-trial`: CPUs to allocate per trial (default: 1)
- `--gpu-fraction`: Fraction of GPU to use per trial (e.g., 0.1 for 10%)
- `--output-dir`: Directory to save results (default: ./tune_results)

### 2. Apply Tuned Parameters to Federated Learning

After tuning is complete, use the `use_tuned_params.py` script to integrate the best parameters into your federated learning system:

```bash
python use_tuned_params.py --params-file ./tune_results/best_params.json
```

This will:
1. Backup the original parameters to `original_bst_params.json`
2. Create a `tuned_params.py` file with the optimized parameters
3. Provide instructions on how to use these parameters

### 3. Run Federated Learning with Tuned Parameters

The federated learning system will automatically detect and use the tuned parameters if:

1. The `tuned_params.py` file exists in the project directory
2. You haven't explicitly provided custom parameters to the XgbClient

## How It Works

The Ray Tune optimization process:

1. **Search Space Definition**: We define a search space for hyperparameters like `max_depth`, `min_child_weight`, `eta`, etc.
2. **ASHA Scheduler**: We use the Asynchronous Successive Halving Algorithm (ASHA) for early stopping of poorly performing trials
3. **Parallel Execution**: Multiple hyperparameter combinations are evaluated in parallel
4. **Best Model Selection**: The best performing model is identified based on validation metrics
5. **Model Persistence**: The best model and its parameters are saved for later use

## Parameters Being Tuned

The following XGBoost parameters are optimized:

- `max_depth`: Maximum depth of a tree
- `min_child_weight`: Minimum sum of instance weight needed in a child
- `eta`: Learning rate
- `subsample`: Subsample ratio of the training instances
- `colsample_bytree`: Subsample ratio of columns when constructing each tree
- `reg_alpha`: L1 regularization term on weights
- `reg_lambda`: L2 regularization term on weights
- `num_boost_round`: Number of boosting rounds

## GPU Support

If you have a GPU available, you can use it to speed up the tuning process by specifying a `--gpu-fraction` value. This enables XGBoost's GPU acceleration via the `gpu_hist` tree method.

## Monitoring and Results

During tuning, progress is logged to the console. After completion, you'll find these files in the output directory:

- `best_params.json`: JSON file containing the best hyperparameters
- `best_model.json`: XGBoost model trained with the best hyperparameters
- `progress.csv`: CSV file with metrics for all trials
- Detailed trial information in subdirectories

## Troubleshooting

- **Memory Issues**: If you encounter memory errors, try reducing `--num-samples` or `--cpus-per-trial`
- **GPU Errors**: If you face GPU-related errors, try removing the `--gpu-fraction` option or installing the appropriate CUDA toolkit
- **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

## Advanced Usage

### Custom Search Space

To customize the hyperparameter search space, modify the `search_space` dictionary in `ray_tune_xgboost.py`.

### Integration with Existing Code

The `client_utils.py` file has been updated to automatically use tuned parameters when available. You can control this behavior by setting the `use_tuned_params` parameter when initializing the `XgbClient`. 
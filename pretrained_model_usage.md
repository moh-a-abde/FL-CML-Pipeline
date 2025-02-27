# Using the Saved Federated Learning Model

This document explains how to use the saved XGBoost model that was trained using the federated learning process.

## Overview

The federated learning process now saves the final trained model after the training process completes. The model is saved in two formats:

1. JSON format (`final_model.json`) - Human-readable format
2. Binary format (`final_model.bin`) - More efficient for loading

These files are saved in the output directory created for each training run, which follows the pattern:
```
outputs/YYYY-MM-DD/HH-MM-SS/
```

## Making Predictions with the Saved Model

We provide a utility script `use_saved_model.py` that demonstrates how to load the saved model and use it for making predictions on new data.

### Prerequisites

Make sure you have all the required dependencies installed:
- XGBoost
- Pandas
- NumPy
- Scikit-learn (for evaluation metrics)
- Flower (for logging utilities)

### Basic Usage

```bash
python use_saved_model.py --model_path <path_to_model> --data_path <path_to_data> --output_path <path_for_predictions>
```

#### Arguments:

- `--model_path`: Path to the saved model file (.json or .bin)
- `--data_path`: Path to the data file (.csv)
- `--output_path`: Path to save the predictions (default: predictions.csv)
- `--has_labels`: Flag to indicate if the data file contains labels (for evaluation)
- `--info_only`: Display model information without making predictions

### Examples

#### Viewing model information:

```bash
python use_saved_model.py --model_path outputs/2023-05-01/12-34-56/final_model.json --info_only
```

This will display information about the model such as the number of trees, feature importance, and model parameters.

#### Making predictions on unlabeled data:

```bash
python use_saved_model.py --model_path outputs/2023-05-01/12-34-56/final_model.json --data_path data/unlabeled_data.csv --output_path predictions.csv
```

#### Evaluating the model on labeled data:

```bash
python use_saved_model.py --model_path outputs/2023-05-01/12-34-56/final_model.json --data_path data/test_data.csv --output_path predictions.csv --has_labels
```

When using the `--has_labels` flag, the script will also calculate and display evaluation metrics such as accuracy, precision, recall, and F1 score.

## Troubleshooting

### Model File Not Found
Make sure the path to the model file is correct. The model files are saved in the output directory created for each training run.

### Model Loading Issues
If you encounter issues loading the model, the system will try multiple approaches:
1. Direct loading using `load_model`
2. Reading the file as bytes and loading into a Booster
3. Creating a new Booster with parameters and then loading the model

### Data Format Issues
The data should be in a format that can be converted to an XGBoost DMatrix. If you're having issues, check that your data has the same features that were used during training.

If you use the `--has_labels` flag but the system can't process the data as labeled, it will automatically fall back to processing it as unlabeled data.

### Memory Issues
If you're working with large datasets, you might encounter memory issues. Consider processing the data in batches or using a machine with more memory.

## Additional Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Flower Documentation](https://flower.dev/docs/) 

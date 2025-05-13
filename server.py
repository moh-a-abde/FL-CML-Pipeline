import warnings
from logging import INFO, WARNING
import os

import flwr as fl
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
import xgboost as xgb

from utils import server_args_parser, BST_PARAMS
from server_utils import (
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    get_evaluate_fn,
    CyclicClientManager,
    setup_output_directory,
    save_results_pickle,
)

from dataset import transform_dataset_to_dmatrix, load_csv_data



warnings.filterwarnings("ignore", category=UserWarning)

# Create output directory structure
output_dir = setup_output_directory()

# Parse arguments for experimental settings
args = server_args_parser()
train_method = args.train_method
pool_size = args.pool_size
num_rounds = args.num_rounds
num_clients_per_round = args.num_clients_per_round
num_evaluate_clients = args.num_evaluate_clients
centralised_eval = args.centralised_eval


# Load centralised test set
if centralised_eval:
    log(INFO, "Loading centralised test set...")
    test_set = load_csv_data("data/static_data.csv")["test"]
    test_set.set_format("pandas")
    test_dmatrix = transform_dataset_to_dmatrix(test_set)

# Define a custom config function that includes the output directory
def custom_eval_config(rnd: int):
    return eval_config(rnd, output_dir)

class CustomFedXgbBagging(FedXgbBagging):
    def aggregate_evaluate(self, server_round, results, failures):
        if self.evaluate_metrics_aggregation_fn is not None:
            eval_metrics = []
            for r in results:
                # Case 1: Object with num_examples and metrics
                if hasattr(r, "num_examples") and hasattr(r, "metrics"):
                    eval_metrics.append((r.num_examples, r.metrics))
                # Case 2: Tuple of (num_examples, metrics_dict)
                elif (
                    isinstance(r, tuple)
                    and len(r) == 2
                    and isinstance(r[0], (int, float))
                    and isinstance(r[1], dict)
                ):
                    eval_metrics.append(r)
                # Case 3: Tuple of (client_proxy, EvaluateRes)
                elif (
                    isinstance(r, tuple)
                    and len(r) == 2
                    and hasattr(r[1], "num_examples")
                    and hasattr(r[1], "metrics")
                ):
                    eval_metrics.append((r[1].num_examples, r[1].metrics))
                else:
                    raise TypeError(
                        f"aggregate_evaluate: Unexpected result format: {type(r)}, value: {r}"
                    )
            aggregated_result = self.evaluate_metrics_aggregation_fn(eval_metrics)
            if not (isinstance(aggregated_result, tuple) and len(aggregated_result) == 2):
                raise TypeError("aggregate_evaluate must return (loss, dict)")
            loss, metrics = aggregated_result
            if not isinstance(metrics, dict):
                raise TypeError("Metrics returned from aggregation must be a dictionary.")
            return loss, metrics
        return super().aggregate_evaluate(server_round, results, failures)

# Define strategy
if train_method == "bagging":
    # Bagging training
    strategy = CustomFedXgbBagging(
        evaluate_function=get_evaluate_fn(test_dmatrix) if centralised_eval else None,
        fraction_fit=(float(num_clients_per_round) / pool_size),
        min_fit_clients=num_clients_per_round,
        min_available_clients=pool_size,
        min_evaluate_clients=num_evaluate_clients if not centralised_eval else 0,
        fraction_evaluate=1.0 if not centralised_eval else 0.0,
        on_evaluate_config_fn=custom_eval_config,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=(
            evaluate_metrics_aggregation if not centralised_eval else None
        ),
    )
    
    # Add a monkey patch to log the loss value before it's returned
    original_aggregate_evaluate = strategy.aggregate_evaluate
    
    def patched_aggregate_evaluate(server_round, eval_results, failures):
        log(INFO, "Aggregating evaluation results for round %s", server_round)
        # Call the original function
        aggregated_result = original_aggregate_evaluate(server_round, eval_results, failures)
        log(INFO, "[DEBUG] aggregate_evaluate received aggregated_result type: %s, value: %s", type(aggregated_result), aggregated_result)
        # Expect (loss, metrics_dict)
        if isinstance(aggregated_result, tuple) and len(aggregated_result) == 2:
            loss, metrics = aggregated_result
            log(INFO, "Aggregated loss for round %s: %s", server_round, loss)
            if isinstance(metrics, dict):
                log(INFO, "Metrics for round %s: %s", server_round, metrics.keys())
            else:
                log(INFO, "[ERROR] Metrics for round %s is not a dictionary: %s", server_round, type(metrics))
                raise TypeError("Metrics returned from aggregation must be a dictionary.")
            return loss, metrics
        log(INFO, "[ERROR] Unexpected format from aggregate_evaluate: %s", type(aggregated_result))
        raise TypeError("aggregate_evaluate must return (loss, dict)")
    
    strategy.aggregate_evaluate = patched_aggregate_evaluate
else:
    # Cyclic training
    strategy = FedXgbCyclic(
        fraction_fit=1.0,
        min_available_clients=pool_size,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=custom_eval_config,
        on_fit_config_fn=fit_config,
    )
    
    # Add a monkey patch to handle the new return format from evaluate_metrics_aggregation
    original_aggregate_evaluate_cyclic = strategy.aggregate_evaluate
    
    def patched_aggregate_evaluate_cyclic(server_round, eval_results, failures):
        log(INFO, "Aggregating evaluation results for round %s (cyclic)", server_round)
        
        # Call the original function
        aggregated_result = original_aggregate_evaluate_cyclic(server_round, eval_results, failures)
        
        # Check the format of the result
        if isinstance(aggregated_result, tuple) and len(aggregated_result) == 2:
            # The result is already in the correct format (loss, metrics)
            loss, metrics = aggregated_result
            
            log(INFO, "Aggregated loss for round %s: %s", server_round, loss)
            
            # Check if metrics is a dictionary before trying to access keys
            if isinstance(metrics, dict):
                log(INFO, "Metrics for round %s: %s", server_round, metrics.keys())
            else:
                log(INFO, "Metrics for round %s is not a dictionary: %s", server_round, type(metrics))
                
                # If metrics is not a dictionary, create a new dictionary
                if metrics is None:
                    metrics = {}
                elif not isinstance(metrics, dict):
                    # Try to convert to dictionary if possible
                    try:
                        metrics = dict(metrics)
                    except (TypeError, ValueError):
                        # If conversion fails, create a new dictionary with the original metrics as a value
                        metrics = {"original_metrics": metrics}
                
                log(INFO, "Created new metrics dictionary: %s", metrics)
            
            # Return the result in the correct format
            return loss, metrics
        
        # The result is not in the expected format
        log(INFO, "Unexpected format from original_aggregate_evaluate_cyclic: %s", type(aggregated_result))
        
        # Try to extract loss and metrics
        if isinstance(aggregated_result, (int, float)):
            # Only loss was returned
            loss = aggregated_result
            metrics = {}
        elif isinstance(aggregated_result, dict):
            # Only metrics were returned
            loss = aggregated_result.get("loss", 0.0)
            metrics = aggregated_result
        else:
            # Unknown format, use defaults
            loss = 0.0
            metrics = {}
        
        log(INFO, "Extracted loss: %s, metrics: %s", loss, metrics)
        
        # Return in the correct format
        return loss, metrics
    
    strategy.aggregate_evaluate = patched_aggregate_evaluate_cyclic

# Start Flower server
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_manager=CyclicClientManager() if train_method == "cyclic" else None,
)

# Save the results after training is complete
log(INFO, "Training complete. Saving results...")

# Create a dictionary to store the results
results = {}

# Add distributed losses if available
if hasattr(history, 'losses_distributed') and history.losses_distributed:
    results["losses_distributed"] = history.losses_distributed
else:
    results["losses_distributed"] = []
    log(INFO, "No distributed losses found in history")

# Add centralized losses if available
if hasattr(history, 'losses_centralized') and history.losses_centralized:
    results["losses_centralized"] = history.losses_centralized
else:
    results["losses_centralized"] = []
    log(INFO, "No centralized losses found in history")

# Add distributed metrics if available
if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
    results["metrics_distributed"] = history.metrics_distributed
else:
    results["metrics_distributed"] = {}
    log(INFO, "No distributed metrics found in history")

# Add centralized metrics if available
if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
    results["metrics_centralized"] = history.metrics_centralized
else:
    results["metrics_centralized"] = {}
    log(INFO, "No centralized metrics found in history")

# Save the results
save_results_pickle(results, output_dir)

# Save the final trained model
log(INFO, "Saving the final trained model...")
if hasattr(strategy, 'global_model') and strategy.global_model is not None:
    # If the strategy has a global_model attribute, convert it to a Booster and save it
    try:
        # Create a booster with the same parameters used in training
        bst = xgb.Booster(params=BST_PARAMS)
        
        # Check if global_model is bytes or bytearray
        if isinstance(strategy.global_model, (bytes, bytearray)):
            # Load the bytes into the booster
            bst.load_model(bytearray(strategy.global_model))
        else:
            # If it's already a Booster, use it directly
            bst = strategy.global_model
            
        # Save the model to a file
        model_path = os.path.join(output_dir, "final_model.json")
        bst.save_model(model_path)
        
        # Also save in binary format for better compatibility
        bin_model_path = os.path.join(output_dir, "final_model.bin")
        bst.save_model(bin_model_path)
        
        log(INFO, "Final model saved to: %s and %s", model_path, bin_model_path)
    except Exception as e:
        log(INFO, "Error saving global model: %s", str(e))
elif hasattr(history, 'parameters_aggregated') and history.parameters_aggregated:
    # If the strategy doesn't have a global_model attribute but history has parameters
    try:
        # Get the final parameters
        final_parameters = history.parameters_aggregated[-1]
        
        # Create a booster with the same parameters used in training
        bst = xgb.Booster(params=BST_PARAMS)
        
        # Load the parameters into the booster
        para_b = bytearray()
        for para in final_parameters.tensors:
            para_b.extend(para)
        
        bst.load_model(para_b)
        
        # Save the model to a file
        model_path = os.path.join(output_dir, "final_model.json")
        bst.save_model(model_path)
        
        # Also save in binary format for better compatibility
        bin_model_path = os.path.join(output_dir, "final_model.bin")
        bst.save_model(bin_model_path)
        
        log(INFO, "Final model saved to: %s and %s", model_path, bin_model_path)
    except Exception as e:
        log(INFO, "Error saving final model: %s", str(e))
else:
    log(INFO, "No final model parameters available to save")

# Also save the final evaluation results
if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
    from server_utils import save_evaluation_results
    final_round = num_rounds
    
    # Check if metrics_distributed is a dictionary or a list
    if isinstance(history.metrics_distributed, dict):
        final_metrics = history.metrics_distributed
    elif isinstance(history.metrics_distributed, list) and len(history.metrics_distributed) > 0:
        final_metrics = history.metrics_distributed[-1][1]  # Get the metrics from the last round
    else:
        final_metrics = {}
        log(INFO, "No metrics available to save")
    
    save_evaluation_results(final_metrics, final_round, output_dir)
else:
    log(INFO, "No metrics available to save")

log(INFO, "Generating additional visualizations...")

# Define CLASS_NAMES based on UNSW_NB15 common mapping
CLASS_NAMES = [
    'Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits',
    'Analysis', 'Fuzzers', 'Worms', 'Shellcode', 'Generic'
]

# Import visualization functions and other necessary modules
from visualization_utils import (
    plot_learning_curves,
    plot_confusion_matrix as vis_plot_confusion_matrix, # Alias to avoid conflict
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_class_distribution,
    plot_per_class_metrics,
    plot_prediction_probability_distributions
)
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. Plot Learning Curves (Loss and Metrics over rounds)
try:
    metrics_for_learning_curve = ['accuracy', 'precision', 'recall', 'f1', 'mlogloss'] # Common metrics
    results_pkl_path = os.path.join(output_dir, "results.pkl")
    if os.path.exists(results_pkl_path):
        plot_learning_curves(results_pkl_path, metrics_for_learning_curve, output_dir)
    else:
        log(WARNING, "results.pkl not found at %s, skipping learning curve plots.", results_pkl_path)
except Exception as e:
    log(WARNING, "Failed to generate learning curve plots: %s", e)

# 2. Generate other plots if centralised evaluation was performed and model is available
if centralised_eval and hasattr(strategy, 'global_model') and strategy.global_model is not None and 'test_dmatrix' in globals():
    log(INFO, "Performing final evaluation on centralised test set for detailed visualizations...")
    try:
        # Reconstruct the final model (Booster)
        final_bst = xgb.Booster(params=BST_PARAMS)
        if isinstance(strategy.global_model, (bytes, bytearray)):
            final_bst.load_model(bytearray(strategy.global_model))
        elif isinstance(strategy.global_model, xgb.Booster): # if it was already a booster (e.g. from a custom strategy)
            final_bst = strategy.global_model
        else:
            raise TypeError("Unsupported global_model type in strategy for visualization.")

        y_true = test_dmatrix.get_label()
        y_pred_proba = final_bst.predict(test_dmatrix)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Plot Confusion Matrix
        conf_matrix_data = confusion_matrix(y_true, y_pred)
        vis_plot_confusion_matrix(conf_matrix_data, CLASS_NAMES, os.path.join(output_dir, "final_confusion_matrix.png"))

        # Plot ROC Curves
        plot_roc_curves(y_true, y_pred_proba, CLASS_NAMES, os.path.join(output_dir, "final_roc_curves.png"))

        # Plot Precision-Recall Curves
        plot_precision_recall_curves(y_true, y_pred_proba, CLASS_NAMES, os.path.join(output_dir, "final_pr_curves.png"))

        # Plot Class Distribution (True vs Predicted on test set)
        plot_class_distribution(y_true, y_pred, CLASS_NAMES, os.path.join(output_dir, "final_class_distribution.png"))

        # Plot Per-Class Metrics (Precision, Recall, F1)
        plot_per_class_metrics(y_true, y_pred, CLASS_NAMES, os.path.join(output_dir, "final_per_class_metrics.png"))

        # Plot Prediction Probability Distributions
        # This function saves to output_dir/prediction_probability_distributions.png by default
        plot_prediction_probability_distributions(y_true, y_pred_proba, CLASS_NAMES, output_dir)
        
        log(INFO, "Successfully generated all detailed visualizations for the final model.")

    except Exception as e:
        log(WARNING, "Failed to generate final model visualizations: %s", e)
elif not centralised_eval:
    log(INFO, "Centralised evaluation was not enabled. Skipping final model detailed visualizations.")
elif not (hasattr(strategy, 'global_model') and strategy.global_model is not None):
    log(INFO, "No final global model available in strategy. Skipping final model detailed visualizations.")
elif 'test_dmatrix' not in globals():
    log(INFO, "Centralised test_dmatrix not available. Skipping final model detailed visualizations.")

log(INFO, "Server process finished.")

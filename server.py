import warnings
from logging import INFO

import flwr as fl
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from utils import server_args_parser
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

# Define strategy
if train_method == "bagging":
    # Bagging training
    strategy = FedXgbBagging(
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
        else:
            # The result is not in the expected format
            log(INFO, "Unexpected format from original_aggregate_evaluate: %s", type(aggregated_result))
            
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
        else:
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

# Add losses if available
if hasattr(history, 'losses_distributed') and history.losses_distributed:
    results["loss"] = history.losses_distributed
else:
    results["loss"] = []
    log(INFO, "No distributed losses found in history")

# Add metrics if available
if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
    results["metrics"] = history.metrics_distributed
else:
    results["metrics"] = {}
    log(INFO, "No distributed metrics found in history")

# Save the results
save_results_pickle(results, output_dir)

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

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
        aggregated_result = original_aggregate_evaluate(server_round, eval_results, failures)
        
        # The aggregated_result is already in the correct format (loss, metrics)
        # No need to extract loss from metrics as evaluate_metrics_aggregation now returns it directly
        loss, metrics = aggregated_result
        
        log(INFO, "Aggregated loss for round %s: %s", server_round, loss)
        log(INFO, "Metrics for round %s: %s", server_round, metrics.keys())
        
        # Return the result as is
        return aggregated_result
    
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

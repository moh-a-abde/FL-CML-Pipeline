import warnings
from logging import INFO

import flwr as fl
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
from flwr.server.history import ServerCallback

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
    
    def patched_aggregate_evaluate(server_round, results, failures):
        log(INFO, "Aggregating evaluation results for round %s", server_round)
        aggregated_result = original_aggregate_evaluate(server_round, results, failures)
        
        # Extract the actual loss from the metrics
        metrics = aggregated_result[1]
        actual_loss = metrics.get("loss", 0.0)
        
        log(INFO, "Original aggregated loss for round %s: %s", server_round, aggregated_result[0])
        log(INFO, "Actual loss from metrics for round %s: %s", server_round, actual_loss)
        
        # Return the corrected result with the actual loss value
        return (actual_loss, aggregated_result[1])
    
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

# Create a custom callback to save results after training
class SaveResultsCallback(ServerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = {"loss": [], "metrics": {}}
        
    def on_evaluate_end(self, server_round, loss, metrics, failures):
        # Save round results
        self.history["loss"].append((server_round, loss))
        
        # Initialize metric dictionaries if they don't exist
        for key, value in metrics.items():
            if key not in self.history["metrics"]:
                self.history["metrics"][key] = []
            self.history["metrics"][key].append((server_round, value))
        
        # Save results after each round
        save_results_pickle(self.history, self.output_dir)
        
        # Also save to the output directory
        if "loss" in metrics:
            eval_metrics = metrics.copy()
            evaluate_metrics_aggregation_fn = strategy.evaluate_metrics_aggregation_fn
            if evaluate_metrics_aggregation_fn is not None:
                # Use the same aggregation function as the strategy
                aggregated_metrics = evaluate_metrics_aggregation_fn([(1, metrics)])
                from server_utils import save_evaluation_results
                save_evaluation_results(aggregated_metrics, server_round, self.output_dir)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_manager=CyclicClientManager() if train_method == "cyclic" else None,
    server_callback=SaveResultsCallback(output_dir),
)

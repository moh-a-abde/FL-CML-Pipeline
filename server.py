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
)

from dataset import (
    transform_dataset_to_dmatrix,
    load_csv_data,
    load_test_data
)

warnings.filterwarnings("ignore", category=UserWarning)

# Parse arguments for experimental settings
args = server_args_parser()
train_method = args.train_method
pool_size = args.pool_size
num_rounds = args.num_rounds
num_clients_per_round = args.num_clients_per_round
num_evaluate_clients = args.num_evaluate_clients
centralised_eval = args.centralised_eval

# Load centralised test set and unlabeled data
if centralised_eval:
    log(INFO, "Loading centralised test set and unlabeled data...")
    test_set = load_test_data() 
    test_set.set_format("pandas")
    test_dmatrix = transform_dataset_to_dmatrix(test_set)
    
    # Load unlabeled data
    unlabeled_data = load_csv_data("data/combined_unlabelled.csv")
    unlabeled_data.set_format("pandas")
    unlabeled_dmatrix = transform_dataset_to_dmatrix(unlabeled_data["train"])

# Define strategy
if train_method == "bagging":
    # Bagging training
    strategy = FedXgbBagging(
        evaluate_function=get_evaluate_fn(test_dmatrix, unlabeled_dmatrix) if centralised_eval else None,
        fraction_fit=(float(num_clients_per_round) / pool_size),
        min_fit_clients=num_clients_per_round,
        min_available_clients=pool_size,
        min_evaluate_clients=num_evaluate_clients if not centralised_eval else 0,
        fraction_evaluate=1.0 if not centralised_eval else 0.0,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=(
            evaluate_metrics_aggregation if not centralised_eval else None
        ),
    )
else:
    # Cyclic training
    strategy = FedXgbCyclic(
        fraction_fit=1.0,
        min_available_clients=pool_size,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
    )

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_manager=CyclicClientManager() if train_method == "cyclic" else None,
)

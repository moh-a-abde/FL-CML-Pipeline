import argparse


# Hyper-parameters for xgboost training
NUM_LOCAL_ROUND = 20  # Increased from 2 for better convergence
BST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 11,  # Fixed to match dataset - has classes 0-10 (11 classes total)
    "eta": 0.05,  # Reasonable learning rate
    "max_depth": 8,  # Increased from 6 for more complex patterns
    "min_child_weight": 5,  # Decreased from 10 for better learning
    "gamma": 0.5,  # Decreased from 1.0 for less aggressive pruning
    "subsample": 0.8,  # Increased from 0.7 for more data per tree
    "colsample_bytree": 0.8,  # Increased from 0.6 for more features per tree
    "colsample_bylevel": 0.8,  # Increased from 0.6 for more features per level
    "nthread": 16,
    "tree_method": "hist",
    "eval_metric": ["mlogloss", "merror"],
    "max_delta_step": 1,  # Decreased from 5 for better convergence
    "reg_alpha": 0.1,  # Decreased from 0.8 for less L1 regularization
    "reg_lambda": 1.0,  # Increased from 0.8 for better L2 regularization
    "base_score": 0.5,  # Neutral starting point
    "scale_pos_weight": 1.0,  # Simplified for multi-class
    "grow_policy": "depthwise",  # Changed from lossguide for more balanced trees
    "normalize_type": "tree",  # Helps with interpretability
    "random_state": 42  # Fixed seed for reproducibility
}


def client_args_parser():
    """Parse arguments to define experimental settings on client side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation or cyclic training.",
    )
    parser.add_argument(
        "--num-partitions", default=10, type=int, help="Number of partitions."
    )
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types.",
    )
    parser.add_argument(
        "--partition-id",
        default=0,
        type=int,
        help="Partition ID used for the current client.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed used for train/test splitting."
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting.",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct evaluation on centralised test set (True), or on hold-out data (False).",
    )
    parser.add_argument(
        "--scaled-lr",
        action="store_true",
        help="Perform scaled learning rate based on the number of clients (True).",
    )
    
    parser.add_argument(
    "--csv-file",
    type=str,
    help="Path to the CSV file for dataset."
)


    args = parser.parse_args()
    return args


def server_args_parser():
    """Parse arguments to define experimental settings on server side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation or cyclic training.",
    )
    parser.add_argument(
        "--pool-size", default=2, type=int, help="Number of total clients."
    )
    parser.add_argument(
        "--num-rounds", default=5, type=int, help="Number of FL rounds."
    )
    parser.add_argument(
        "--num-clients-per-round",
        default=2,
        type=int,
        help="Number of clients participate in training each round.",
    )
    parser.add_argument(
        "--num-evaluate-clients",
        default=2,
        type=int,
        help="Number of clients selected for evaluation.",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )

    args = parser.parse_args()
    return args


def sim_args_parser():
    """Parse arguments to define experimental settings on server side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation or cyclic training.",
    )

    # Server side
    parser.add_argument(
        "--pool-size", default=5, type=int, help="Number of total clients."
    )
    parser.add_argument(
        "--num-rounds", default=30, type=int, help="Number of FL rounds."
    )
    parser.add_argument(
        "--num-clients-per-round",
        default=5,
        type=int,
        help="Number of clients participate in training each round.",
    )
    parser.add_argument(
        "--num-evaluate-clients",
        default=5,
        type=int,
        help="Number of clients selected for evaluation.",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )
    parser.add_argument(
        "--num-cpus-per-client",
        default=2,
        type=int,
        help="Number of CPUs used for per client.",
    )

    # Client side
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed used for train/test splitting."
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting.",
    )
    parser.add_argument(
        "--centralised-eval-client",
        action="store_true",
        help="Conduct evaluation on centralised test set (True), or on hold-out data (False).",
    )
    parser.add_argument(
        "--scaled-lr",
        action="store_true",
        help="Perform scaled learning rate based on the number of clients (True).",
    )
    
    parser.add_argument(
    "--csv-file",
    type=str,
    help="Path to the CSV file for dataset."
)

    args = parser.parse_args()
    return args

"""
client.py

This module implements the Federated Learning client functionality for distributed XGBoost training.
It handles data loading, preprocessing, and client-side model training operations.

Key Components:
- Data loading and partitioning
- Client initialization
- Model training configuration
- Connection to FL server
"""

import warnings
from logging import INFO
import os

import flwr as fl
from flwr.common.logger import log

from dataset import (
    load_csv_data,
    instantiate_partitioner,
    train_test_split,
    transform_dataset_to_dmatrix,
    resplit,
)
from utils import client_args_parser, BST_PARAMS, NUM_LOCAL_ROUND
from client_utils import XgbClient

warnings.filterwarnings("ignore", category=UserWarning)

def get_latest_csv(directory: str) -> str:
    """
    Retrieves the most recently modified CSV file from the specified directory.

    Args:
        directory (str): Path to the directory containing CSV files

    Returns:
        str: Full path to the most recent CSV file

    Example:
        latest_file = get_latest_csv("/path/to/data/directory")
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)

if __name__ == "__main__":
    # Parse command line arguments for experimental settings
    args = client_args_parser()

    data_directory = "data/received"
    #latest_csv_path = get_latest_csv(data_directory)
    #labeled_dataset = load_csv_data(latest_csv_path)
    #unlabeled_dataset = load_csv_data(latest_csv_path)
    
    # Load labeled data for training
    labeled_csv_path = "data/received/network_train_60.csv"
    labeled_dataset = load_csv_data(labeled_csv_path)
    
    # Load unlabeled data for prediction
    unlabeled_csv_path = "data/received/network_test_40_nolabel.csv"
    unlabeled_dataset = load_csv_data(unlabeled_csv_path)
    
    # Initialize data partitioner based on specified strategy
    partitioner = instantiate_partitioner(
        partitioner_type=args.partitioner_type,
        num_partitions=args.num_partitions
    )
    
    # Load the specific partition for training
    log(INFO, "Loading training partition...")
    train_partition = labeled_dataset["train"]
    train_partition.set_format("numpy")
    
    # Handle data splitting based on evaluation strategy
    if args.centralised_eval:
        # Use centralized test set for evaluation
        train_data = train_partition
        valid_data = labeled_dataset["test"]
        valid_data.set_format("numpy")
        num_train = train_data.shape[0]
        num_val = valid_data.shape[0]
    else:
        # Perform local train/test split using the updated function
        log(INFO, "Performing local train/test split...")
        train_dmatrix, valid_dmatrix = train_test_split(
            train_partition,
            test_fraction=args.test_fraction,
            random_state=args.seed  # Use random_state instead of seed
        )
        # Get counts from the DMatrix objects
        num_train = train_dmatrix.num_row()
        num_val = valid_dmatrix.num_row()
        log(INFO, f"Local split: {num_train} train samples, {num_val} validation samples")
    
    # Transform unlabeled data for prediction (train/valid are already DMatrix)
    log(INFO, "Reformatting unlabeled data...")
    unlabeled_data = unlabeled_dataset["train"]
    # Convert unlabeled data to pandas DataFrame first
    if not isinstance(unlabeled_data, pd.DataFrame):
        unlabeled_data = unlabeled_data.to_pandas()
    
    # We need a FeatureProcessor instance. Since we don't have the one from train_test_split,
    # let's re-create and fit it on the training data DMatrix (requires converting back temporarily)
    # This is inefficient, ideally the processor should be passed around.
    # TODO: Refactor to pass the fitted FeatureProcessor instance from train_test_split
    temp_train_df = train_dmatrix.get_data() # Assuming get_data() returns a suitable format, might need adjustment
    if isinstance(temp_train_df, xgb.QuantileDMatrix): # Handle potential QuantileDMatrix if using GPU hist
        log(WARNING, "Cannot reliably reconstruct DataFrame from QuantileDMatrix for refitting processor. Unlabeled data preprocessing might be inconsistent.")
        # Fallback or raise error? For now, create an empty processor.
        processor = FeatureProcessor() 
    else:
         # Attempt to reconstruct DataFrame, assuming columns match original order
         # This part is fragile and depends on DMatrix internals/assumptions
         try:
             # We need feature names to reconstruct the DataFrame correctly
             # Assuming train_dmatrix has feature_names attribute
             temp_train_df_pd = pd.DataFrame(temp_train_df, columns=train_dmatrix.feature_names)
             processor = FeatureProcessor()
             processor.fit(temp_train_df_pd) # Fit processor on reconstructed training data
         except Exception as e:
             log(ERROR, f"Failed to reconstruct DataFrame from DMatrix for processor fitting: {e}. Proceeding with unfitted processor for unlabeled data.")
             processor = FeatureProcessor()


    # Preprocess unlabeled data using the (potentially refitted) processor
    unlabeled_features, _ = preprocess_data(unlabeled_data, processor=processor, is_training=False)
    unlabeled_dmatrix = xgb.DMatrix(unlabeled_features, missing=np.nan)
    
    # Configure training parameters
    num_local_round = NUM_LOCAL_ROUND
    params = BST_PARAMS
    
    # Adjust learning rate for bagging method if specified
    if args.train_method == "bagging" and args.scaled_lr:
        new_lr = params["eta"] / args.num_partitions
        params.update({"eta": new_lr})
    
    # Create client with both training and prediction data
    client = XgbClient(
        train_dmatrix=train_dmatrix,
        valid_dmatrix=valid_dmatrix,
        num_train=num_train,
        num_val=num_val,
        num_local_round=num_local_round,
        params=params,
        train_method=args.train_method,
        is_prediction_only=False,  # Set to False for training
        unlabeled_dmatrix=unlabeled_dmatrix  # Add unlabeled data for prediction
    )
    
    # Initialize and start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

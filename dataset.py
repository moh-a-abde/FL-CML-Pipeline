"""
dataset.py

This module handles all dataset-related operations for the federated learning system.
It provides functionality for loading, preprocessing, partitioning, and transforming
network traffic data for XGBoost training.

Key Components:
- Data loading and preprocessing
- Feature engineering (numerical and categorical)
- Dataset partitioning strategies
- Data format conversions
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from datasets import Dataset, DatasetDict, concatenate_datasets
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)
from typing import Union, Tuple

# Mapping between partitioning strategy names and their implementations
CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}

def load_csv_data(file_path: str) -> DatasetDict:
    """
    Load and prepare CSV data into a Hugging Face DatasetDict format.

    Args:
        file_path (str): Path to the CSV file containing network traffic data

    Returns:
        DatasetDict: Dataset dictionary containing train and test splits

    Example:
        dataset = load_csv_data("path/to/network_data.csv")
    """
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # print dataset statistics
    print(f"Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    
    # Convert to Dataset format
    dataset = Dataset.from_pandas(df)
    return DatasetDict({"train": dataset, "test": dataset})

def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """
    Create a data partitioner based on specified strategy and number of partitions.

    Args:
        partitioner_type (str): Type of partitioning strategy 
            ('uniform', 'linear', 'square', 'exponential')
        num_partitions (int): Number of partitions to create

    Returns:
        Partitioner: Initialized partitioner object
    """
    partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
        num_partitions=num_partitions
    )
    return partitioner
    
def preprocess_data(data):
    """
    Preprocess the data by encoding categorical features and separating features and labels.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        
    Returns:
        tuple: (features DataFrame, labels Series or None if unlabeled)
    """
    # Define categorical and numerical features
    categorical_features = ['id.orig_h', 'id.resp_h', 'proto', 'conn_state', 'history']
    numerical_features = ['id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
                         'local_orig', 'local_resp', 'missed_bytes', 'orig_pkts', 
                         'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Convert categorical features to category type
    for col in categorical_features:
        df[col] = df[col].astype('category')
        # Get numerical codes for categories
        df[col] = df[col].cat.codes
    
    # Ensure numerical features are float type
    for col in numerical_features:
        df[col] = df[col].astype(float)
    
    # Check if this is labeled or unlabeled data
    if 'label' in df.columns:
        # For labeled data
        features = df.drop(columns=['label'])
        labels = df['label'].astype(float)
        return features, labels
    else:
        # For unlabeled data
        return df, None

def separate_xy(data):
    """
    Separate features and labels from the dataset.
    
    Args:
        data: Input dataset
        
    Returns:
        tuple: (features, labels or None if unlabeled)
    """
    return preprocess_data(data.to_pandas())

def transform_dataset_to_dmatrix(data):
    """
    Transform dataset to DMatrix format.
    
    Args:
        data: Input dataset
        
    Returns:
        xgb.DMatrix: Transformed dataset
    """
    x, y = separate_xy(data)
    
    # For unlabeled data, create DMatrix without labels
    if y is None:
        return xgb.DMatrix(x)
    
    # For labeled data, create DMatrix with labels
    return xgb.DMatrix(x, label=y, enable_categorical=True)

def train_test_split(
    partition: Dataset, 
    test_fraction: float, 
    seed: int
) -> Tuple[Dataset, Dataset, int, int]:
    """
    Split dataset into training and validation sets.

    Args:
        partition (Dataset): Input dataset to split
        test_fraction (float): Fraction of data to use for testing
        seed (int): Random seed for reproducibility

    Returns:
        Tuple containing:
            - Training dataset
            - Test dataset
            - Number of training samples
            - Number of test samples
    """
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test

def resplit(dataset: DatasetDict) -> DatasetDict:
    """
    Increase the quantity of centralized test samples by reallocating from training set.

    Args:
        dataset (DatasetDict): Input dataset with train/test splits

    Returns:
        DatasetDict: Dataset with adjusted train/test split sizes

    Note:
        Moves 10K samples from training to test set (if available)
    """
    train_size = dataset["train"].num_rows
    test_size = dataset["test"].num_rows
    
    # Ensure we don't exceed the number of samples in the training set
    additional_test_samples = min(10000, train_size)
    
    return DatasetDict(
        {
            "train": dataset["train"].select(
                range(0, train_size - additional_test_samples)
            ),
            "test": concatenate_datasets(
                [
                    dataset["train"].select(
                        range(
                            train_size - additional_test_samples,
                            train_size,
                        )
                    ),
                    dataset["test"],
                ]
            ),
        }
    )

class ModelPredictor:
    """
    Handles model prediction and dataset labeling
    """
    def __init__(self, model_path: str):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
    
    def predict_and_save(
        self,
        input_data: Union[str, pd.DataFrame],
        output_path: str,
        include_confidence: bool = True
    ):
        """
        Predict on new data and save labeled dataset
        """
        # Load/preprocess input data
        data = self._prepare_data(input_data)
        
        # Generate predictions
        predictions = self.model.predict(data)
        confidence = None
        if include_confidence:
            confidence = self.model.predict(data, output_margin=True)
        
        # Save labeled dataset
        self._save_output(data, predictions, confidence, output_path)




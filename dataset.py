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
    
def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess network traffic data by encoding categorical features and scaling numerical features.

    Args:
        data (pd.DataFrame): Raw network traffic data

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed features and encoded labels

    Note:
        Categorical features are one-hot encoded
        Numerical features are standardized
    """
    # Define feature groups
    categorical_features = ['id.orig_h', 'id.resp_h', 'proto', 'history', 'conn_state']
    numerical_features = ['id.orig_p', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'missed_bytes',
                         'local_resp', 'local_orig', 'resp_bytes', 'orig_bytes', 'duration', 'id.resp_p']
    
    # Initialize preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Separate features and labels
    features = data.drop(columns=['label'])
    labels = data['label']
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Transform features
    features_transformed = preprocessor.fit_transform(features)
    
    return features_transformed, labels_encoded

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

def transform_dataset_to_dmatrix(data: Union[Dataset, DatasetDict], dataset_name: str = "unnamed") -> xgb.DMatrix:
    """
    Transform dataset into XGBoost's DMatrix format.

    Args:
        data (Union[Dataset, DatasetDict]): Input dataset
        dataset_name (str): Name/identifier for the dataset

    Returns:
        xgb.DMatrix: Data in XGBoost's optimized format

    Note:
        Automatically reshapes features to 2D if needed
    """
    x, y = separate_xy(data)
    
    #print transformation details
    print(f"Transforming dataset '{dataset_name}':")
    print(f"Features shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Reshape x to 2D if needed
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)
        print(f"Reshaped features to: {x.shape}")
        
    new_data = xgb.DMatrix(x, label=y)
    return new_data

def separate_xy(data: Union[Dataset, DatasetDict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate features and labels from dataset.

    Args:
        data (Union[Dataset, DatasetDict]): Input dataset

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels arrays
    """
    x, y = preprocess_data(data.to_pandas())
    return x, y

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




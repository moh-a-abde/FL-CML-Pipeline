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
from typing import Union

CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}

def load_csv_data(file_path: str) -> DatasetDict:
    """Load CSV data into a DatasetDict format."""
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    return DatasetDict({"train": dataset, "test": dataset})

def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """Initialise partitioner based on selected partitioner type and number of
    partitions."""
    partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
        num_partitions=num_partitions
    )
    return partitioner
    
def preprocess_data(data: pd.DataFrame):
    """Preprocess data by encoding categorical features and scaling numerical features."""
    # Define categorical and numerical features
    categorical_features = ['id.orig_h', 'id.resp_h', 'proto', 'history', 'conn_state']
    numerical_features = ['id.orig_p', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'missed_bytes',
                          'local_resp', 'local_orig', 'resp_bytes', 'orig_bytes', 'duration', 'id.resp_p']
    
    # Define the column transformer
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
    
    # Fit and transform the features
    features_transformed = preprocessor.fit_transform(features)
    
    return features_transformed, labels_encoded

def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test

def transform_dataset_to_dmatrix(data: Union[Dataset, DatasetDict]) -> xgb.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x, y = separate_xy(data)
    # Reshape x to 2D if it's not already
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)
    new_data = xgb.DMatrix(x, label=y)
    return new_data

def separate_xy(data: Union[Dataset, DatasetDict]):
    """Return outputs of x (data) and y (labels)."""
    x, y = preprocess_data(data.to_pandas())
    return x, y

def resplit(dataset: DatasetDict) -> DatasetDict:
    """Increase the quantity of centralised test samples from 10K to 20K by taking from the training set."""
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


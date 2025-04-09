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
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict, concatenate_datasets
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)
from typing import Union, Tuple
from sklearn.model_selection import train_test_split as train_test_split_pandas
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from flwr.common.logger import log
from logging import INFO, WARNING

# Mapping between partitioning strategy names and their implementations
CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}

class FeatureProcessor:
    """Handles feature preprocessing while preventing data leakage."""
    
    def __init__(self):
        self.categorical_encoders = {}
        self.numerical_stats = {}
        self.is_fitted = False
        
        # Define feature groups
        self.categorical_features = [
            'id.orig_h', 'id.resp_h', 'proto', 'conn_state', 'history', 
            'validation_status', 'method', 'status_msg', 'is_orig',
            'local_orig', 'local_resp'
        ]
        self.numerical_features = [
            'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
            'missed_bytes', 'orig_pkts',
            'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'ts_delta',
            'rtt', 'acks', 'percent_lost', 'request_body_len',
            'response_body_len', 'seen_bytes', 'missing_bytes', 'overflow_bytes'
        ]
        # Removed object_columns as they may cause data leakage
        log(INFO, "Note: Removed object_columns (uid, client_initial_dcid, server_scid) to prevent potential data leakage")

    def fit(self, df: pd.DataFrame) -> None:
        """Fit preprocessing parameters on training data only."""
        if self.is_fitted:
            return
            
        # Initialize encoders for categorical features
        for col in self.categorical_features:
            if col in df.columns:
                unique_values = df[col].unique()
                # Create a mapping for each unique value to an integer
                self.categorical_encoders[col] = {
                    val: idx for idx, val in enumerate(unique_values)
                }
                # Log warning if a categorical feature is highly predictive
                if len(unique_values) > 1 and len(unique_values) < 10:
                    for val in unique_values:
                        subset = df[df[col] == val]
                        if 'label' in df.columns and len(subset) > 0:
                            most_common_label = subset['label'].value_counts().idxmax()
                            label_pct = subset['label'].value_counts()[most_common_label] / len(subset)
                            if label_pct > 0.9:  # If >90% of rows with this value have the same label
                                log(WARNING, "Potential data leakage detected: Feature '%s' value '%s' is highly predictive of label %s (%.1f%% match)",
                                    col, val, most_common_label, label_pct * 100)

        # Store numerical feature statistics
        for col in self.numerical_features:
            if col in df.columns:
                self.numerical_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'median': df[col].median(),
                    'q99': df[col].quantile(0.99)
                }

        self.is_fitted = True

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        if not self.is_fitted and is_training:
            self.fit(df)
        elif not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")

        df = df.copy()
        
        # Drop object columns since they might be causing data leakage
        object_columns_to_remove = ['uid', 'client_initial_dcid', 'server_scid']
        columns_dropped = []
        for col in object_columns_to_remove:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                columns_dropped.append(col)
        
        if columns_dropped and not is_training:
            log(INFO, "Dropped potential leakage columns: %s", columns_dropped)

        # Transform categorical features
        for col in self.categorical_features:
            if col in df.columns and col in self.categorical_encoders:
                # Map known categories, set unknown to -1
                df[col] = df[col].map(self.categorical_encoders[col]).fillna(-1)

        # Handle numerical features with added noise for validation data
        for col in self.numerical_features:
            if col in df.columns and col in self.numerical_stats:
                # Replace infinities
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Add noise to all numerical features for validation data
                # This ensures validation is a truly independent test
                if not is_training:
                    # Add noise proportional to the standard deviation of each feature
                    std = self.numerical_stats[col]['std']
                    # If std is 0 or NaN, use a small default value
                    noise_scale = max(std, 0.1) * 0.02  # 2% of standard deviation (reduced from 5%)
                    noise = np.random.normal(0, noise_scale, size=df.shape[0])
                    df[col] = df[col] + noise
                    
                # Cap outliers using 99th percentile
                q99 = self.numerical_stats[col]['q99']
                df.loc[df[col] > q99, col] = q99  # Re-enabled outlier capping
                
                # Fill NaN with median plus small noise
                median = self.numerical_stats[col]['median']
                # Find NaN positions
                nan_mask = df[col].isna()
                
                # First fill with median value
                df[col] = df[col].fillna(median)
                
                # Then add noise only to the previously NaN positions if not training
                if not is_training and nan_mask.any():
                    # Add a tiny bit of noise to medians for previously NaN values
                    noise_scale = median * 0.005 if median != 0 else 0.0005  # 0.5% of median (reduced from 1%)
                    noise = np.random.normal(0, noise_scale, size=nan_mask.sum())
                    df.loc[nan_mask, col] += noise

        return df

def preprocess_data(data: pd.DataFrame, processor: FeatureProcessor = None, is_training: bool = False):
    """
    Preprocess the data by encoding categorical features and separating features and labels.
    Handles multi-class classification with three classes: benign (0), dns_tunneling (1), and icmp_tunneling (2).
    
    Args:
        data (pd.DataFrame): Input DataFrame
        processor (FeatureProcessor): Feature processor instance for consistent preprocessing
        is_training (bool): Whether this is training data
        
    Returns:
        tuple: (features DataFrame, labels Series or None if unlabeled)
    """
    if processor is None:
        processor = FeatureProcessor()
    
    # Process features
    df = processor.transform(data, is_training)
    
    # Handle labels
    if 'label' in df.columns:
        features = df.drop(columns=['label'])
        labels = df['label'].astype(int)
        
        # Validate labels
        unique_labels = labels.unique()
        if not all(label in [0, 1, 2] for label in unique_labels):
            print(f"Warning: Unexpected label values found: {unique_labels}")
            labels = labels.map(lambda x: x if x in [0, 1, 2] else -1)
        
        return features, labels
    return df, None

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
    print("Loading dataset from:", file_path)
    df = pd.read_csv(file_path)
    
    # print dataset statistics
    print("Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    
    # Check if this is an unlabeled test set (from filename)
    is_unlabeled = "nolabel" in file_path.lower()
    
    # Create appropriate dataset structure
    dataset = Dataset.from_pandas(df)
    
    if is_unlabeled:
        # For unlabeled data, keep the current structure (all data in both train/test)
        # This won't create issues since unlabeled data is only used for prediction
        return DatasetDict({"train": dataset, "test": dataset})
    else:
        # For labeled data, create a proper 80/20 split to avoid data leakage
        # Use a specific random seed for reproducibility
        train_test_dict = dataset.train_test_split(test_size=0.2, seed=42)
        return DatasetDict({
            "train": train_test_dict["train"],
            "test": train_test_dict["test"]
        })

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

def preprocess_data_deprec2(data):
    """/
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
    # For unlabeled data
    return df, None

def preprocess_data_deprec(data):
    """
    Preprocess the static_data.csv dataset by:
      - Dropping the 'Timestamp' column.
      - Converting 'Dst Port' and 'Protocol' to categorical features.
      - Converting remaining features (except 'Label') to numerical (float).
      - Separating features and target (Label), and encoding the target if necessary.
    
    Args:
        filepath (str): Path to the static_data.csv file.
    
    Returns:
        tuple: (features DataFrame, labels Series or None if unlabeled)
    """
    # Create a copy to avoid modifying original data
    df = data.copy()

    # Drop 'Timestamp' as it is not used for training directly
    if 'Timestamp' in df.columns:
        df.drop(columns=['Timestamp'], inplace=True)
    
    # Define which columns to treat as categorical based on domain knowledge
    categorical_features = []
    if 'Dst Port' in df.columns:
        categorical_features.append('Dst Port')
    if 'Protocol' in df.columns:
        categorical_features.append('Protocol')
    
    # Convert categorical features to type 'category' and then to numerical codes
    for col in categorical_features:
        df[col] = df[col].astype('category').cat.codes
    
    # The numerical features are all columns except the ones we have categorized or the target
    numerical_features = [col for col in df.columns if col not in categorical_features + ['Label']]
    
    # Convert these numerical features to float
    for col in numerical_features:
        df[col] = df[col].astype(float)

    # Replace inf with NaN and cap large values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in numerical_features:
        max_val = 1e15  # Example max value, adjust as needed
        df[col] = np.where(df[col] > max_val, np.nan, df[col])

    # Process the target variable if present
    if 'Label' in df.columns:
        # If the label column is non-numeric (object), encode it as categorical codes.
        if df['Label'].dtype == object:
            labels = df['Label'].astype('category').cat.codes
        else:
            labels = df['Label']
        features = df.drop(columns=['Label'])
        return features, labels
    # If no label column, return the processed DataFrame and None for labels
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

def transform_dataset_to_dmatrix(data, processor: FeatureProcessor = None, is_training: bool = False):
    """
    Transform dataset to DMatrix format.
    
    Args:
        data: Input dataset
        processor (FeatureProcessor): Feature processor instance for consistent preprocessing
        is_training (bool): Whether this is training data
        
    Returns:
        xgb.DMatrix: Transformed dataset
    """
    # The input 'data' should already be a pandas DataFrame in this context
    x, y = preprocess_data(data, processor=processor, is_training=is_training)
    
    # Handle case where preprocess_data might return None for labels (e.g., unlabeled data)
    if y is None:
        log(INFO, "No labels found in data. This appears to be unlabeled data for prediction only.")
        return xgb.DMatrix(x, missing=np.nan)
    
    # For validation data, log label distribution to help identify issues
    if not is_training:
        # Count occurrences of each label
        label_counts = np.bincount(y.astype(int))
        label_names = ['benign', 'dns_tunneling', 'icmp_tunneling']
        log(INFO, "Label distribution in validation data:")
        for i, count in enumerate(label_counts):
            class_name = label_names[i] if i < len(label_names) else f'unknown_{i}'
            log(INFO, f"  {class_name}: {count}")
    
    return xgb.DMatrix(x, label=y, missing=np.nan)

def train_test_split(
    data,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[xgb.DMatrix, xgb.DMatrix, FeatureProcessor]:
    """
    Split dataset into train and test sets, preprocess, and return DMatrices and the fitted processor.
    
    Args:
        data: Input dataset (Hugging Face Dataset or pandas DataFrame)
        test_fraction (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[xgb.DMatrix, xgb.DMatrix, FeatureProcessor]: 
            - Training DMatrix
            - Test DMatrix
            - Fitted FeatureProcessor instance
    """
    # Convert to pandas if needed
    if not isinstance(data, pd.DataFrame):
        data = data.to_pandas()
    
    # Use sklearn's train_test_split with shuffle=True to ensure data is properly randomized
    log(INFO, "Original data shape before splitting: %s", data.shape)
    
    # Set random seed for consistency
    np.random.seed(random_state)
    
    # Check if 'label' column exists in data
    if 'label' not in data.columns:
        log(INFO, "Warning: No 'label' column found in data. Available columns: %s", data.columns.tolist())
    else:
        # Report class distribution
        label_counts = data['label'].value_counts().to_dict()
        log(INFO, "Class distribution in original data: %s", label_counts)
    
    # Check for data leakage indicators
    if 'uid' in data.columns:
        uid_label_counts = data.groupby('uid')['label'].value_counts()
        uid_with_multiple_labels = uid_label_counts.index.get_level_values(0).duplicated(keep=False)
        if not any(uid_with_multiple_labels):
            log(WARNING, "CRITICAL: Each UID has only one label, indicating potential perfect data leakage through UIDs")
    
    # Generate a completely different random_state for validation split
    validation_random_state = (random_state * 17 + 3) % 10000
    log(INFO, "Using different random states for train/validation split: %d/%d", 
        random_state, validation_random_state)
    
    # Split data ensuring complete partition separation
    if 'uid' in data.columns:
        # If we have UIDs, use them to ensure no data leakage across train/test
        log(INFO, "Using UID-based splitting to ensure no data leakage")
        unique_uids = data['uid'].unique()
        np.random.seed(validation_random_state)
        np.random.shuffle(unique_uids)
        test_size = int(len(unique_uids) * test_fraction)
        test_uids = unique_uids[:test_size]
        train_uids = unique_uids[test_size:]
        
        # Split based on UIDs
        train_data = data[data['uid'].isin(train_uids)].copy()
        test_data = data[data['uid'].isin(test_uids)].copy()
        
        log(INFO, "Split by UIDs: %d train UIDs, %d test UIDs", len(train_uids), len(test_uids))
    else:
        # If no UIDs, use standard stratified split
        train_data, test_data = train_test_split_pandas(
            data,
            test_size=test_fraction,
            random_state=validation_random_state,
            shuffle=True,  # Ensure data is shuffled for a proper split
            stratify=data['label'] if 'label' in data.columns else None  # Use stratified split if possible
        )
    
    # Log the shapes to verify they're different sets
    log(INFO, "Train data shape: %s, Test data shape: %s", train_data.shape, test_data.shape)
    
    # Verify label distributions to ensure proper stratification
    if 'label' in data.columns:
        train_label_counts = train_data['label'].value_counts().to_dict()
        test_label_counts = test_data['label'].value_counts().to_dict()
        log(INFO, "Class distribution in train data: %s", train_label_counts)
        log(INFO, "Class distribution in test data: %s", test_label_counts)
    
    # Check for unique values in both sets to verify they're actually different
    if 'uid' in data.columns:
        train_uids = set(train_data['uid'].unique())
        test_uids = set(test_data['uid'].unique())
        common_uids = train_uids.intersection(test_uids)
        if common_uids:
            log(WARNING, "WARNING: Found %d UIDs in both train and test sets! This indicates data leakage.", 
                len(common_uids))
        else:
            log(INFO, "Good: Train and test sets have completely separate UIDs (no overlap).")
    
    # Add dataset-specific noise to test set to make it more challenging
    # This helps prevent the model from simply memorizing patterns
    for col in train_data.columns:
        if col.startswith('id.') or col in ['proto', 'conn_state', 'duration', 'bytes']:
            continue  # Skip these columns
        if pd.api.types.is_numeric_dtype(train_data[col]):
            # Add noise only to numerical columns in test set
            col_std = test_data[col].std()
            if col_std > 0 and not pd.isna(col_std):
                noise_scale = col_std * 0.05  # 5% of standard deviation (reduced from 10%)
                test_data[col] = test_data[col] + np.random.normal(0, noise_scale, size=test_data.shape[0])
                log(INFO, "Added noise to test column: %s", col)
    
    # Initialize feature processor
    processor = FeatureProcessor()
    
    # Fit processor on training data and transform both sets
    # Note: transform calls fit implicitly if is_training=True and not fitted
    train_dmatrix = transform_dataset_to_dmatrix(train_data, processor=processor, is_training=True)
    test_dmatrix = transform_dataset_to_dmatrix(test_data, processor=processor, is_training=False)
    
    # Log number of examples for verification
    log(INFO, "Train DMatrix has %d rows, Test DMatrix has %d rows", 
        train_dmatrix.num_row(), test_dmatrix.num_row())
    
    return train_dmatrix, test_dmatrix, processor # Return the fitted processor

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
    # test_size = dataset["test"].num_rows  # Removed unused variable

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

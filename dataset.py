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
from logging import INFO, WARNING, ERROR
import pickle
import os

# Mapping between partitioning strategy names and their implementations
CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}

class FeatureProcessor:
    """Handles feature preprocessing while preventing data leakage."""
    
    def __init__(self, dataset_type="unsw_nb15"):
        """
        Initialize the feature processor.
        
        Args:
            dataset_type (str): Type of dataset to process.
                Options: "unsw_nb15" (original) or "engineered" (new dataset)
        """
        self.categorical_encoders = {}
        self.numerical_stats = {}
        self.is_fitted = False
        self.label_encoder = LabelEncoder()
        self.dataset_type = dataset_type
        
        # Define feature groups based on dataset type
        if dataset_type == "unsw_nb15":
            # Original UNSW_NB15 dataset features
            self.categorical_features = [
                'proto', 'service', 'state', 'is_ftp_login', 'is_sm_ips_ports'
            ]
            self.numerical_features = [
                'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 
                'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 
                'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 
                'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 
                'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
                'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst'
            ]
        elif dataset_type == "engineered":
            # New engineered dataset features - all are numerical (pre-normalized)
            self.categorical_features = []
            self.numerical_features = [
                'dur', 'sbytes', 'dbytes', 'Sload', 'swin', 'smeansz', 'Sjit', 'Stime',
                'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_dst_src_ltm',
                'duration', 'jit_ratio', 'inter_pkt_ratio', 'tcp_setup_ratio',
                'byte_pkt_interaction_dst', 'load_jit_interaction_dst', 'tcp_seq_diff'
            ]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def fit(self, df: pd.DataFrame) -> None:
        """Fit preprocessing parameters on training data only."""
        if self.is_fitted:
            return
        
        # === LABEL COLUMN STANDARDIZATION DURING FIT ===
        # Ensure consistent label column naming during fitting
        df_copy = df.copy()
        if 'attack_cat' in df_copy.columns and 'label' not in df_copy.columns:
            log(INFO, "Standardizing 'attack_cat' column to 'label' during fit")
            df_copy = df_copy.rename(columns={'attack_cat': 'label'})
        elif 'Label' in df_copy.columns and 'label' not in df_copy.columns:
            log(INFO, "Standardizing 'Label' column to 'label' during fit")
            df_copy = df_copy.rename(columns={'Label': 'label'})
            
        # Initialize encoders for categorical features
        for col in self.categorical_features:
            if col in df_copy.columns:
                unique_values = df_copy[col].unique()
                # Create a mapping for each unique value to an integer
                self.categorical_encoders[col] = {
                    val: idx for idx, val in enumerate(unique_values)
                }
                # Log warning if a categorical feature is highly predictive
                if len(unique_values) > 1 and len(unique_values) < 10:
                    for val in unique_values:
                        subset = df_copy[df_copy[col] == val]
                        if 'label' in df_copy.columns and len(subset) > 0:
                            most_common_label = subset['label'].value_counts().idxmax()
                            label_pct = subset['label'].value_counts()[most_common_label] / len(subset)
                            if label_pct > 0.9:  # If >90% of rows with this value have the same label
                                log(WARNING, "Potential data leakage detected: Feature '%s' value '%s' is highly predictive of label %s (%.1f%% match)",
                                    col, val, most_common_label, label_pct * 100)

        # Store numerical feature statistics - for original dataset or data validation
        # For engineered dataset, this will be minimal since data is already normalized
        if self.dataset_type == "unsw_nb15":
            for col in self.numerical_features:
                if col in df_copy.columns:
                    self.numerical_stats[col] = {
                        'mean': df_copy[col].mean(),
                        'std': df_copy[col].std(),
                        'median': df_copy[col].median(),
                        'q99': df_copy[col].quantile(0.99)
                    }
        else:
            # For engineered dataset, we only track basic stats for validation
            # No need for extensive normalization since data is already normalized
            for col in self.numerical_features:
                if col in df_copy.columns:
                    self.numerical_stats[col] = {
                        'min': df_copy[col].min(),
                        'max': df_copy[col].max(),
                        'median': df_copy[col].median(),
                    }
        
        # Fit label encoder for standardized 'label' column
        if 'label' in df_copy.columns:
            label_values = df_copy['label']
            if label_values.dtype == 'object' or isinstance(label_values.iloc[0], str):
                log(INFO, "Fitting label encoder for categorical labels")
                self.label_encoder.fit(label_values)
            else:
                log(INFO, "Labels are already numeric, no encoding needed")
        
        # For engineered dataset with numeric labels, just record the unique labels
        if 'label' in df_copy.columns and self.dataset_type == "engineered":
            self.unique_labels = sorted(df_copy['label'].unique())
            log(INFO, f"Found {len(self.unique_labels)} unique labels in engineered dataset: {self.unique_labels}")

        self.is_fitted = True

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Transform data using fitted parameters."""
        if not self.is_fitted and is_training:
            self.fit(df)
        elif not self.is_fitted:
            # If not fitted and not training, we should fit it anyway to avoid errors
            # This is needed for the centralized evaluation case
            log(INFO, "FeatureProcessor not fitted but needed for transform. Fitting now.")
            self.fit(df)
            
        df = df.copy()
        
        # === LABEL COLUMN STANDARDIZATION ===
        # Ensure consistent label column naming throughout the pipeline
        if 'attack_cat' in df.columns and 'label' not in df.columns:
            # Convert attack_cat to standardized 'label' column for original dataset
            log(INFO, "Standardizing 'attack_cat' column to 'label' for consistent naming")
            df = df.rename(columns={'attack_cat': 'label'})
        elif 'Label' in df.columns and 'label' not in df.columns:
            # Convert uppercase 'Label' to lowercase 'label' for consistency
            log(INFO, "Standardizing 'Label' column to 'label' for consistent naming")
            df = df.rename(columns={'Label': 'label'})
        
        # Drop id column since it's just an identifier
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
        
        # Transform categorical features (only needed for original dataset)
        for col in self.categorical_features:
            if col in df.columns and col in self.categorical_encoders:
                # Map known categories, set unknown to -1
                df[col] = df[col].map(self.categorical_encoders[col]).fillna(-1)

        # Handle numerical features with different approaches based on dataset type
        if self.dataset_type == "unsw_nb15":
            # Original dataset needs normalization and outlier handling
            for col in self.numerical_features:
                if col in df.columns and col in self.numerical_stats:
                    # Replace infinities
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Cap outliers using 99th percentile - fix dtype compatibility
                    q99 = self.numerical_stats[col]['q99']
                    # Ensure q99 has the same dtype as the column to avoid FutureWarning
                    if df[col].dtype.kind in 'biufc':  # numeric dtypes
                        q99 = df[col].dtype.type(q99)
                    df.loc[df[col] > q99, col] = q99  # Cap outliers
                    
                    # Fill NaN with median
                    median = self.numerical_stats[col]['median']
                    # Ensure median has the same dtype as the column
                    if df[col].dtype.kind in 'biufc':  # numeric dtypes
                        median = df[col].dtype.type(median)
                    df[col] = df[col].fillna(median)
        else:
            # Engineered dataset is already normalized, just handle missing values
            for col in self.numerical_features:
                if col in df.columns:
                    # Replace infinities and NaN with 0 (since data is normalized, 0 is a reasonable default)
                    df[col] = df[col].replace([np.inf, -np.inf, np.nan], 0)

        # === IMPORTANT: DO NOT DROP THE STANDARDIZED 'label' COLUMN ===
        # The label column should be preserved so that downstream components can find it
        # The preprocess_data function will handle label extraction separately
        
        # Only drop the original attack_cat column if it still exists after renaming
        if 'attack_cat' in df.columns:
            log(INFO, "Dropping original 'attack_cat' column after standardization")
            df.drop(columns=['attack_cat'], inplace=True)

        return df

def preprocess_data(data: Union[pd.DataFrame, Dataset], processor: FeatureProcessor = None, is_training: bool = False):
    """
    Preprocess the data by encoding categorical features and separating features and labels.
    Handles multi-class classification for both original and engineered datasets.
    
    Args:
        data (Union[pd.DataFrame, Dataset]): Input DataFrame or Hugging Face Dataset
        processor (FeatureProcessor): Feature processor instance for consistent preprocessing
        is_training (bool): Whether this is training data
        
    Returns:
        tuple: (features DataFrame, labels Series or None if unlabeled)
    """
    # Convert Hugging Face Dataset to pandas DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = data.to_pandas()
    
    if processor is None:
        # Auto-detect dataset type based on columns
        if 'attack_cat' in data.columns:
            processor = FeatureProcessor(dataset_type="unsw_nb15")
        elif 'tcp_seq_diff' in data.columns:
            processor = FeatureProcessor(dataset_type="engineered")
        else:
            log(WARNING, "Could not automatically detect dataset type. Defaulting to 'unsw_nb15'.")
            processor = FeatureProcessor(dataset_type="unsw_nb15")
    
    # === STANDARDIZED LABEL HANDLING ===
    # Process features first (this will standardize label column names)
    features = processor.transform(data, is_training)
    
    # Now extract labels from the standardized 'label' column
    labels = None
    
    if 'label' in features.columns:
        log(INFO, "Found standardized 'label' column in processed data")
        labels = features['label'].copy()
        
        # Handle label encoding based on dataset type
        if processor.dataset_type == "unsw_nb15":
            # For original dataset, labels need to be encoded if they're categorical
            if labels.dtype == 'object' or isinstance(labels.iloc[0], str):
                log(INFO, "Encoding categorical labels for UNSW_NB15 dataset")
                # Ensure label encoder is fitted if needed (during training)
                if is_training and not hasattr(processor.label_encoder, 'classes_'):
                    log(INFO, "Fitting label encoder during training preprocessing.")
                    processor.label_encoder.fit(labels)
                elif not hasattr(processor.label_encoder, 'classes_') or processor.label_encoder.classes_.size == 0:
                    log(WARNING, "Label encoder not fitted, cannot transform categorical labels.")
                    try:
                        processor.label_encoder.fit(labels)
                        log(WARNING, "Fitted label encoder on non-training data chunk.")
                    except Exception as fit_err:
                        log(ERROR, f"Could not fit label encoder on non-training data: {fit_err}")
                        labels = np.full(len(data), -1, dtype=int)
                
                # Transform labels if encoder is ready
                if hasattr(processor.label_encoder, 'classes_') and processor.label_encoder.classes_.size > 0:
                    try:
                        labels = processor.label_encoder.transform(labels)
                    except ValueError as e:
                        log(ERROR, f"Error transforming labels: {e}. Unseen labels might exist.")
                        labels = np.full(len(data), -1, dtype=int)
            else:
                # Labels are already numeric
                labels = labels.astype(int)
        else:
            # For engineered dataset, labels should already be numeric
            log(INFO, "Using direct numeric labels from engineered dataset.")
            labels = labels.astype(int)
            
        # Log label distribution
        if labels is not None:
            try:
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts = dict(zip(unique_labels, counts))
                log(INFO, f"Label distribution: {label_counts}")
            except Exception as e:
                log(WARNING, f"Could not compute label distribution: {e}")
        
        # Remove label column from features to avoid data leakage
        features = features.drop(columns=['label'])
        log(INFO, "Removed 'label' column from features to prevent data leakage")
    else:
        # No label column found
        log(INFO, "No 'label' column found in processed data - assuming unlabeled data")
        labels = None

    return features, labels

def load_csv_data(file_path: str) -> DatasetDict:
    """
    Load and prepare CSV data into a Hugging Face DatasetDict format.
    Uses temporal splitting based on Stime column to avoid data leakage.

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
    
    # Auto-detect dataset type
    if 'attack_cat' in df.columns:
        print("Detected original UNSW_NB15 dataset with attack_cat column")
    elif 'tcp_seq_diff' in df.columns:
        print("Detected engineered dataset with normalized features")
    
    # Check if this is an unlabeled test set (from filename)
    is_unlabeled = "nolabel" in file_path.lower()
    
    # Create appropriate dataset structure
    dataset = Dataset.from_pandas(df)
    
    if is_unlabeled:
        # For unlabeled data, keep the current structure (all data in both train/test)
        # This won't create issues since unlabeled data is only used for prediction
        return DatasetDict({"train": dataset, "test": dataset})
    else:
        # For labeled data, use temporal splitting to avoid data leakage
        # Sort by Stime column if available for temporal integrity
        if 'Stime' in df.columns:
            print("Using temporal splitting based on Stime column to avoid data leakage")
            df_sorted = df.sort_values('Stime').reset_index(drop=True)
            
            # Split temporally: first 80% for training, last 20% for testing
            train_size = int(0.8 * len(df_sorted))
            train_df = df_sorted.iloc[:train_size]
            test_df = df_sorted.iloc[train_size:]
            
            # Optional: shuffle within each set to add randomness while maintaining temporal integrity
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"Temporal split: {len(train_df)} train samples, {len(test_df)} test samples")
            print(f"Train Stime range: {train_df['Stime'].min():.4f} to {train_df['Stime'].max():.4f}")
            print(f"Test Stime range: {test_df['Stime'].min():.4f} to {test_df['Stime'].max():.4f}")
            
            # Verify no temporal overlap
            if train_df['Stime'].max() <= test_df['Stime'].min():
                print("✓ No temporal overlap between train and test sets")
            else:
                print("⚠ Warning: Temporal overlap detected between train and test sets")
            
            return DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df)
            })
        else:
            # Fallback to stratified random split if no temporal column available
            print("No Stime column found, using stratified random split")
            train_test_dict = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label' if 'label' in df.columns else None)
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

def transform_dataset_to_dmatrix(data, processor: FeatureProcessor = None, is_training: bool = False):
    """
    Transform dataset to DMatrix format.
    
    Args:
        data: Input dataset (can be pandas DataFrame or Hugging Face Dataset)
        processor (FeatureProcessor): Feature processor instance for consistent preprocessing
        is_training (bool): Whether this is training data
        
    Returns:
        xgb.DMatrix: Transformed dataset
    """
    # Convert Hugging Face Dataset to pandas DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = data.to_pandas()
    
    # Now process the data using the pandas DataFrame
    x, y = preprocess_data(data, processor=processor, is_training=is_training)
    
    # --- Logging before DMatrix creation ---
    log(INFO, f"[transform_dataset_to_dmatrix] is_training={is_training}")
    log(INFO, f"[transform_dataset_to_dmatrix] Features shape: {x.shape}")
    if y is not None:
        log(INFO, f"[transform_dataset_to_dmatrix] Labels type: {type(y)}")
        log(INFO, f"[transform_dataset_to_dmatrix] Labels shape: {y.shape if hasattr(y, 'shape') else 'N/A'}")
        log(INFO, f"[transform_dataset_to_dmatrix] Labels head: {y[:5] if hasattr(y, '__len__') and len(y) > 0 else 'N/A'}")
    else:
        log(INFO, "[transform_dataset_to_dmatrix] Labels are None.")
    # --- End Logging ---

    # Handle case where preprocess_data might return None for labels (e.g., unlabeled data)
    if y is None:
        log(INFO, "No labels found in data. Creating DMatrix without labels.")
        return xgb.DMatrix(x, missing=np.nan)
    
    # For validation data, log label distribution to help identify issues
    if not is_training:
        # Count occurrences of each label
        label_counts = np.bincount(y.astype(int))
        # Use the correct class names for UNSW_NB15 dataset if possible
        if hasattr(processor, 'dataset_type') and processor.dataset_type == "unsw_nb15":
            label_names = ['Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits', 'Analysis', 'Fuzzers', 'Worms', 'Shellcode', 'Generic'] 
        else:
            # For engineered dataset, use numeric labels as names
            label_names = [str(i) for i in range(len(label_counts))]
            
        log(INFO, "Label distribution in validation data:")
        for i, count in enumerate(label_counts):
            if i < len(label_names):
                class_name = label_names[i]
            else:
                class_name = f'unknown_{i}'
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
    
    # Auto-detect dataset type
    if 'attack_cat' in data.columns:
        dataset_type = "unsw_nb15"
    elif 'tcp_seq_diff' in data.columns:
        dataset_type = "engineered"
    else:
        dataset_type = "unsw_nb15"  # Default
        log(WARNING, "Could not auto-detect dataset type. Defaulting to 'unsw_nb15'.")
    
    # Initialize appropriate feature processor
    processor = FeatureProcessor(dataset_type=dataset_type)
        
    # Check if 'label' column exists in data
    label_col = 'label' if dataset_type == "engineered" else 'label' 
    if label_col not in data.columns:
        log(INFO, "Warning: No '%s' column found in data. Available columns: %s", 
            label_col, data.columns.tolist())
    else:
        # Report class distribution
        label_counts = data[label_col].value_counts().to_dict()
        log(INFO, "Class distribution in original data: %s", label_counts)
    
    # Check for data leakage indicators
    if 'uid' in data.columns:
        uid_label_counts = data.groupby('uid')[label_col].value_counts()
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
            stratify=data[label_col] if label_col in data.columns else None  # Use stratified split if possible
        )
    
    # Log the shapes to verify they're different sets
    log(INFO, "Train data shape: %s, Test data shape: %s", train_data.shape, test_data.shape)
    
    # Verify label distributions to ensure proper stratification
    if label_col in data.columns:
        train_label_counts = train_data[label_col].value_counts().to_dict()
        test_label_counts = test_data[label_col].value_counts().to_dict()
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
    
    # Fit processor on training data and transform both sets
    # Note: transform calls fit implicitly if is_training=True and not fitted
    train_dmatrix = transform_dataset_to_dmatrix(train_data, processor=processor, is_training=True)
    test_dmatrix = transform_dataset_to_dmatrix(test_data, processor=processor, is_training=False)
    
    # Log number of examples for verification
    log(INFO, "Train DMatrix has %d rows, Test DMatrix has %d rows", 
        train_dmatrix.num_row(), test_dmatrix.num_row())
    
    # Return the fitted processor along with DMatrices
    return train_dmatrix, test_dmatrix, processor

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

def create_global_feature_processor(data_file: str, output_dir: str = "outputs") -> str:
    """
    Create a global feature processor fitted on the full training dataset.
    This ensures consistent preprocessing across Ray Tune and Federated Learning.
    
    Args:
        data_file (str): Path to the dataset file
        output_dir (str): Directory to save the processor
        
    Returns:
        str: Path to the saved processor file
    """
    log(INFO, f"Creating global feature processor from: {data_file}")
    
    # Load full dataset
    dataset = load_csv_data(data_file)
    train_data = dataset["train"]
    train_df = train_data.to_pandas()
    
    # Auto-detect dataset type
    if 'attack_cat' in train_df.columns:
        dataset_type = "unsw_nb15"
    elif 'tcp_seq_diff' in train_df.columns:
        dataset_type = "engineered"
    else:
        dataset_type = "unsw_nb15"  # Default
        log(WARNING, "Could not auto-detect dataset type. Defaulting to 'unsw_nb15'.")
    
    # Create and fit processor on full training data
    processor = FeatureProcessor(dataset_type=dataset_type)
    processor.fit(train_df)
    
    # Save processor to file
    os.makedirs(output_dir, exist_ok=True)
    processor_path = os.path.join(output_dir, "global_feature_processor.pkl")
    
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    
    log(INFO, f"Global feature processor saved to: {processor_path}")
    log(INFO, f"Processor type: {dataset_type}")
    log(INFO, f"Categorical features: {len(processor.categorical_features)}")
    log(INFO, f"Numerical features: {len(processor.numerical_features)}")
    
    return processor_path

def load_global_feature_processor(processor_path: str) -> FeatureProcessor:
    """
    Load a pre-fitted global feature processor.
    
    Args:
        processor_path (str): Path to the saved processor file
        
    Returns:
        FeatureProcessor: The loaded processor
    """
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Global feature processor not found at: {processor_path}")
    
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    log(INFO, f"Loaded global feature processor from: {processor_path}")
    return processor

# Comment out or remove ModelPredictor if not used or complete
# class ModelPredictor:
#     """
#     Handles model prediction and dataset labeling
#     """
#     def __init__(self, model_path: str):
#         self.model = xgb.Booster()
#         self.model.load_model(model_path)
    
#     def predict_and_save(
#         self,
#         input_data: Union[str, pd.DataFrame],
#         output_path: str,
#         include_confidence: bool = True
#     ):
#         """
#         Predict on new data and save labeled dataset
#         """
#         # Load/preprocess input data
#         # data = self._prepare_data(input_data) # Requires _prepare_data method
        
#         # Generate predictions
#         # predictions = self.model.predict(data)
#         # confidence = None
#         # if include_confidence:
#         #     confidence = self.model.predict(data, output_margin=True)
        
#         # Save labeled dataset
#         # self._save_output(data, predictions, confidence, output_path) # Requires _save_output method

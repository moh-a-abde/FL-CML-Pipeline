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
    Uses hybrid temporal-stratified splitting to avoid data leakage while ensuring class coverage.

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
        # For labeled data, use hybrid temporal-stratified splitting to avoid data leakage
        # while ensuring all classes are present in both train and test splits
        if 'Stime' in df.columns and 'label' in df.columns:
            print("Using hybrid temporal-stratified split to preserve time order while ensuring class coverage")
            
            # Sort by time first to maintain temporal integrity
            df_sorted = df.sort_values('Stime').reset_index(drop=True)
            
            # Create temporal windows to split data while preserving time order
            n_windows = 10  # Split into 10 temporal windows
            window_size = len(df_sorted) // n_windows
            
            train_dfs = []
            test_dfs = []
            
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < n_windows - 1 else len(df_sorted)
                window_df = df_sorted.iloc[start_idx:end_idx]
                
                # Within each window, use stratified split to ensure all classes are represented
                if len(window_df) > 0:
                    try:
                        from sklearn.model_selection import train_test_split
                        train_window, test_window = train_test_split(
                            window_df, test_size=0.2, random_state=42, 
                            stratify=window_df['label']
                        )
                        train_dfs.append(train_window)
                        test_dfs.append(test_window)
                    except ValueError as e:
                        # Some classes missing in this window - use temporal split as fallback
                        print(f"  Warning: Stratification failed for window {i}: {e}")
                        print(f"  Falling back to temporal split for this window")
                        window_train_size = int(0.8 * len(window_df))
                        train_dfs.append(window_df.iloc[:window_train_size])
                        test_dfs.append(window_df.iloc[window_train_size:])
            
            # Combine all windows
            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = pd.concat(test_dfs, ignore_index=True)
            
            # Verify all classes are present in both splits
            train_classes = set(train_df['label'].unique())
            test_classes = set(test_df['label'].unique())
            all_classes = set(df['label'].unique())
            
            print(f"Hybrid split: {len(train_df)} train samples, {len(test_df)} test samples")
            print(f"Train classes: {sorted(train_classes)} ({len(train_classes)} classes)")
            print(f"Test classes: {sorted(test_classes)} ({len(test_classes)} classes)")
            print(f"All classes: {sorted(all_classes)} ({len(all_classes)} classes)")
            
            # Check for missing classes and apply fallback if needed
            missing_train_classes = all_classes - train_classes
            missing_test_classes = all_classes - test_classes
            
            if missing_train_classes or missing_test_classes:
                print(f"⚠️ WARNING: Missing classes detected!")
                if missing_train_classes:
                    print(f"  Missing from training: {sorted(missing_train_classes)}")
                if missing_test_classes:
                    print(f"  Missing from testing: {sorted(missing_test_classes)}")
                
                print("  Applying fallback: Pure stratified split to ensure all classes")
                from sklearn.model_selection import train_test_split
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42, stratify=df['label']
                )
                print(f"✓ Fallback complete: {len(train_df)} train, {len(test_df)} test samples")
                print(f"✓ All classes now present in both splits")
            else:
                print("✓ All classes successfully present in both train and test splits")
                
                # Show temporal ranges for verification
                print(f"Train Stime range: {train_df['Stime'].min():.4f} to {train_df['Stime'].max():.4f}")
                print(f"Test Stime range: {test_df['Stime'].min():.4f} to {test_df['Stime'].max():.4f}")
            
            # Final verification: print class distribution
            print("\nFinal class distribution:")
            train_counts = train_df['label'].value_counts().sort_index()
            test_counts = test_df['label'].value_counts().sort_index()
            
            for label in sorted(all_classes):
                train_count = train_counts.get(label, 0)
                test_count = test_counts.get(label, 0)
                total_count = train_count + test_count
                print(f"  Class {label}: {train_count} train, {test_count} test, {total_count} total")
            
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
        log(INFO, f"[transform_dataset_to_dmatrix] Labels shape: {y.shape}")
        log(INFO, f"[transform_dataset_to_dmatrix] Labels dtype: {y.dtype}")
        unique_labels, counts = np.unique(y, return_counts=True)
        log(INFO, f"[transform_dataset_to_dmatrix] Unique labels: {unique_labels.tolist()}")
        log(INFO, f"[transform_dataset_to_dmatrix] Label counts: {counts.tolist()}")
    else:
        log(INFO, "[transform_dataset_to_dmatrix] No labels provided (unlabeled data)")

    # Create DMatrix
    dmatrix = xgb.DMatrix(x, label=y)
    
    log(INFO, f"[transform_dataset_to_dmatrix] Created DMatrix with {dmatrix.num_row()} rows and {dmatrix.num_col()} features")
    
    return dmatrix

def train_test_split(
    data,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[xgb.DMatrix, xgb.DMatrix, FeatureProcessor]:
    """
    Split dataset into training and testing sets with proper feature processing.
    Returns training DMatrix, testing DMatrix, and fitted feature processor.
    
    Note: This function is DEPRECATED in favor of load_csv_data() which implements
    hybrid temporal-stratified splitting to prevent data leakage. This function
    remains for backward compatibility but should not be used for new code.
    
    Args:
        data: Input dataset (pandas DataFrame or Hugging Face Dataset)
        test_fraction (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dmatrix, test_dmatrix, fitted_processor)
    """
    log(WARNING, "train_test_split() is DEPRECATED. Use load_csv_data() with hybrid temporal-stratified splitting instead.")
    
    # Convert to pandas DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = data.to_pandas()
    
    # Auto-detect dataset type
    if 'attack_cat' in data.columns:
        processor = FeatureProcessor(dataset_type="unsw_nb15")
    elif 'tcp_seq_diff' in data.columns:
        processor = FeatureProcessor(dataset_type="engineered")
    else:
        log(WARNING, "Could not automatically detect dataset type. Defaulting to 'unsw_nb15'.")
        processor = FeatureProcessor(dataset_type="unsw_nb15")
    
    # Preprocess the data
    x, y = preprocess_data(data, processor=processor, is_training=True)
    
    if y is not None:
        # Use stratified split to maintain class distribution
        x_train, x_test, y_train, y_test = train_test_split_pandas(
            x, y, test_size=test_fraction, random_state=random_state,
            stratify=y
        )
        
        # Create DMatrix objects
        train_dmatrix = xgb.DMatrix(x_train, label=y_train)
        test_dmatrix = xgb.DMatrix(x_test, label=y_test)
        
        log(INFO, f"Split dataset: {train_dmatrix.num_row()} training samples, {test_dmatrix.num_row()} testing samples")
        log(INFO, f"Features: {train_dmatrix.num_col()}")
        
        return train_dmatrix, test_dmatrix, processor
    else:
        # No labels available - split features only
        x_train, x_test = train_test_split_pandas(
            x, test_size=test_fraction, random_state=random_state
        )
        
        train_dmatrix = xgb.DMatrix(x_train)
        test_dmatrix = xgb.DMatrix(x_test)
        
        log(INFO, f"Split unlabeled dataset: {train_dmatrix.num_row()} training samples, {test_dmatrix.num_row()} testing samples")
        log(INFO, f"Features: {train_dmatrix.num_col()}")
        
        return train_dmatrix, test_dmatrix, processor

def resplit(dataset: DatasetDict) -> DatasetDict:
    """
    Resplit an existing DatasetDict to redistribute data between train/test splits.
    This function combines train and test data, then applies a new stratified split.
    
    Note: This function bypasses the hybrid temporal-stratified splitting logic 
    that prevents data leakage. Use with caution and consider whether the original
    load_csv_data() approach is more appropriate for your use case.
    
    Args:
        dataset (DatasetDict): Dataset dictionary with 'train' and 'test' splits
        
    Returns:
        DatasetDict: New dataset dictionary with redistributed train/test splits
    """
    log(WARNING, "resplit() bypasses temporal-stratified splitting. Consider using load_csv_data() instead.")
    
    # Combine train and test data
    combined_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    
    # Convert to pandas for stratified splitting
    combined_df = combined_dataset.to_pandas()
    
    # Determine stratification column
    if 'label' in combined_df.columns:
        stratify_col = 'label'
    elif 'attack_cat' in combined_df.columns:
        stratify_col = 'attack_cat'
    else:
        stratify_col = None
        log(WARNING, "No label column found for stratification. Using random split.")
    
    # Split the combined dataset
    if stratify_col:
        try:
            train_df, test_df = train_test_split_pandas(
                combined_df, test_size=0.2, random_state=42,
                stratify=combined_df[stratify_col]
            )
            log(INFO, f"Resplit dataset with stratification on '{stratify_col}'")
        except ValueError as e:
            log(WARNING, f"Stratification failed: {e}. Using random split.")
            train_df, test_df = train_test_split_pandas(
                combined_df, test_size=0.2, random_state=42
            )
    else:
        train_df, test_df = train_test_split_pandas(
            combined_df, test_size=0.2, random_state=42
        )
    
    # Convert back to DatasetDict
    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

def create_global_feature_processor(data_file: str, output_dir: str = "outputs") -> str:
    """
    Create and save a global feature processor fitted on the entire dataset.
    This ensures consistent preprocessing across all federated learning clients.
    
    Args:
        data_file (str): Path to the CSV file containing the full dataset
        output_dir (str): Directory to save the fitted processor
        
    Returns:
        str: Path to the saved processor file
    """
    log(INFO, f"Creating global feature processor from {data_file}")
    
    # Load the full dataset
    df = pd.read_csv(data_file)
    log(INFO, f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
    
    # Auto-detect dataset type
    if 'attack_cat' in df.columns:
        processor = FeatureProcessor(dataset_type="unsw_nb15")
        log(INFO, "Detected original UNSW_NB15 dataset")
    elif 'tcp_seq_diff' in df.columns:
        processor = FeatureProcessor(dataset_type="engineered")
        log(INFO, "Detected engineered dataset")
    else:
        processor = FeatureProcessor(dataset_type="unsw_nb15")
        log(WARNING, "Could not detect dataset type. Defaulting to 'unsw_nb15'")
    
    # Fit the processor on the full dataset
    processor.fit(df)
    log(INFO, "Fitted feature processor on full dataset")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the fitted processor
    processor_path = os.path.join(output_dir, "global_feature_processor.pkl")
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    
    log(INFO, f"Saved global feature processor to {processor_path}")
    return processor_path

def load_global_feature_processor(processor_path: str) -> FeatureProcessor:
    """
    Load a previously saved feature processor.
    
    Args:
        processor_path (str): Path to the saved processor file
        
    Returns:
        FeatureProcessor: Loaded and fitted processor
    """
    try:
        with open(processor_path, 'rb') as f:
            processor = pickle.load(f)
        log(INFO, f"Loaded global feature processor from {processor_path}")
        
        if not processor.is_fitted:
            log(WARNING, "Loaded processor is not fitted!")
        
        return processor
    except FileNotFoundError:
        log(ERROR, f"Feature processor file not found: {processor_path}")
        raise
    except Exception as e:
        log(ERROR, f"Error loading feature processor: {e}")
        raise

def separate_xy(data):
    """
    Separate features (X) and labels (y) from a dataset.
    This is a convenience function that wraps preprocess_data.
    
    Args:
        data: Input dataset (pandas DataFrame or Hugging Face Dataset)
        
    Returns:
        tuple: (features, labels) where features is a numpy array and labels is a numpy array
    """
    # Convert Hugging Face Dataset to pandas DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = data.to_pandas()
    
    # Use preprocess_data to get features and labels
    features, labels = preprocess_data(data, processor=None, is_training=False)
    
    # Convert to numpy arrays for compatibility with existing code
    features_array = features.values if features is not None else None
    labels_array = labels.values if labels is not None else None
    
    return features_array, labels_array 
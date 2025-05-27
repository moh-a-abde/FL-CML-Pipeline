
--- Repository Documentation ---

```markdown
# Federated Learning with Flower (XGBoost Comprehensive) Documentation for AI

This document provides comprehensive documentation for the Federated Learning with Flower (XGBoost Comprehensive) repository, tailored for understanding and utilization by AI systems.

## 1. Repository Purpose and "What Is It" Summary

This repository implements a Federated Learning (FL) system using the Flower framework to train XGBoost models in a distributed manner. Its primary focus is on collaborative training of intrusion detection models on network traffic data without requiring raw data exchange between participants (clients).

The project integrates several key components:
*   **Data Handling:** Loading, preprocessing, and partitioning of network traffic datasets.
*   **Federated Learning Core:** Implementing server and client logic based on the Flower framework.
*   **Model Training:** Utilizing XGBoost for classification.
*   **Training Strategies:** Supporting both **FedXgbBagging** (parallel, aggregation-based) and **FedXgbCyclic** (sequential, model-passing) strategies from Flower.
*   **Configuration:** Using command-line arguments and implicit configuration (like `tuned_params.py`) for experimental settings.
*   **Hyperparameter Tuning:** Integration with Ray Tune to optimize XGBoost parameters.
*   **Consistent Preprocessing:** Implementation of a global feature processor to ensure uniform data handling across tuning and FL phases, preventing data leakage.
*   **Evaluation & Visualization:** Tools for evaluating model performance (metrics, confusion matrices, learning curves) and saving predictions.

The overall goal is to demonstrate a privacy-preserving approach to training powerful tree-based models like XGBoost on distributed datasets, with a focus on cybersecurity applications (network intrusion detection).

## 2. Quick Start

To get the project up and running for basic federated training:

**Prerequisites:**

*   Python 3.8+
*   pip (Python package manager)
*   Git
*   (Optional, but recommended for full CML workflow) Docker

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/moh-a-abde/FL-CML-Pipeline.git
    cd FL-CML-Pipeline
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This installs core dependencies including `flwr` simulation, `xgboost`, `ray[tune]`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `datasets`, `flwr-datasets`).*
    *Note: `pyproject.toml` also lists dependencies, but `requirements.txt` is the primary installation method used in scripts and workflows.*

**Data Preparation:**

*   Place your CSV data file in the `data/received/` directory. The system is configured to primarily use `data/received/final_dataset.csv`. Ensure this file exists or modify the scripts (`run.py`, `run_bagging.sh`, `run_ray_tune.sh`, `client.py`, `server.py`) to point to your data file. The `dataset.py` module supports multi-class UNSW_NB15 and an engineered dataset format.

**Running a Basic Federated Training Session (Bagging):**

This command starts a Flower server and multiple Flower clients using the `FedXgbBagging` strategy.

1.  **Ensure the global feature processor is created:**
    ```bash
    python create_global_processor.py --data-file data/received/final_dataset.csv --output-dir outputs --force
    ```
2.  **Run the bagging script:**
    ```bash
    ./run_bagging.sh
    ```
    *(This script will start the server and several clients in the background and wait for them to finish. Output will be logged to the console and saved in the `outputs/` directory).*

**Running a Basic Federated Training Session (Cyclic):**

This command starts a Flower server and multiple Flower clients using the `FedXgbCyclic` strategy.

```bash
./run_cyclic.sh
```
*(This script starts the server and clients similarly to `run_bagging.sh`).*

**Running the Full Pipeline (Includes Processor Creation, Ray Tune, and Simulation):**

The `run.py` script orchestrates the entire process: creating the global processor, running Ray Tune hyperparameter optimization, applying tuned parameters, and finally running the federated learning simulation.

```bash
python run.py
```
*(This is the most comprehensive quick start for testing the full integrated workflow).*

## 3. Configuration Options

The project uses a combination of command-line arguments and internal constants/generated files (`tuned_params.py`) for configuration.

**Command-Line Arguments:**

Scripts like `client.py`, `server.py`, `sim.py`, `ray_tune_xgboost_updated.py`, `create_global_processor.py`, `use_tuned_params.py`, and `use_saved_model.py` accept arguments defined in `utils.py`, `ray_tune_xgboost_updated.py`, `create_global_processor.py`, `use_tuned_params.py`, and `use_saved_model.py` respectively.

Key arguments managed via `utils.py` parsers:

*   **`--train-method`**: (`server.py`, `client.py`, `sim.py`) Choose between `bagging` or `cyclic` FL strategy.
*   **`--num-rounds`**: (`server.py`, `sim.py`) Number of federated learning rounds.
*   **`--pool-size`**: (`server.py`, `sim.py`) Total number of clients available in the simulation pool.
*   **`--num-clients-per-round`**: (`server.py`, `sim.py`) Number of clients selected for training in each round.
*   **`--num-evaluate-clients`**: (`server.py`, `sim.py`) Number of clients selected for evaluation in each round (used by some strategies).
*   **`--centralised-eval`**: (`server.py`, `sim.py`, `client_utils.py`, `client.py`) Flag to enable evaluation on a separate, centralized test set managed by the server.
*   **`--partitioner-type`**: (`client.py`, `sim.py`) Data partitioning strategy (`uniform`, `linear`, `square`, `exponential`).
*   **`--num-partitions`**: (`client.py`) Number of data partitions to create (matches `--pool-size` in `sim.py`).
*   **`--partition-id`**: (`client.py`) The ID of the data partition assigned to a specific client instance.
*   **`--seed`**: (`client.py`, `sim.py`) Random seed for data splitting.
*   **`--test-fraction`**: (`client.py`, `sim.py`) Fraction of local data to use for client-side evaluation.
*   **`--scaled-lr`**: (`client.py`, `sim.py`) Flag to scale the client-side learning rate based on the number of clients (relevant for bagging).
*   **`--csv-file`**: (`client.py`, `sim.py`, `create_global_processor.py`) Path to the main dataset CSV file. Overridden by `--data-file`, `--train-file`, `--test-file` in tuning scripts.

**Internal Configuration (`utils.py`):**

*   **`BST_PARAMS`**: A dictionary defining the default XGBoost parameters used by clients and the server. This is the base configuration which can be overridden by tuned parameters.
*   **`NUM_LOCAL_ROUND`**: Default number of local boosting rounds each client performs. This can be overridden by `tuned_params.py`.

**Dynamic Configuration (`tuned_params.py`):**

*   The `use_tuned_params.py` script generates `tuned_params.py` based on the output of `ray_tune_xgboost_updated.py`.
*   If `tuned_params.py` exists, it overrides `BST_PARAMS` and `NUM_LOCAL_ROUND` in `client_utils.py`, automatically applying the optimized parameters to the FL clients.

**Hydra Configuration (Indicated in `README.md`, but `conf/base.yaml` not provided):**

*   The `README.md` mentions Hydra for experiment settings, pointing to `conf/base.yaml`. While the file content is not available in this merged view, the description in `README.md` indicates it would define parameters like `num_rounds`, `num_clients`, `batch_size`, `num_clients_per_round_fit`, `num_clients_per_round_eval`, and `local_epochs`. It suggests these can be overridden via the command line (e.g., `python run.py num_rounds=20`).
*   The `server.py` and `client.py` scripts as provided **do not explicitly use Hydra**, relying on command-line arguments parsed by `utils.py`. The `run.py` script orchestrates other scripts via subprocess calls, effectively wrapping their command-line interfaces.
*   The `.hydra` directory in the output structure suggests Hydra *was* intended or previously used, and the `setup_output_directory` copies `.hydra` files if they exist in the current directory. However, based *only* on the provided Python scripts (`server.py`, `client.py`, `sim.py`), configuration is primarily via `argparse`.

**Summary of Configuration Sources:**

1.  **Default XGBoost Parameters:** `utils.py:BST_PARAMS`
2.  **Default Local Rounds:** `utils.py:NUM_LOCAL_ROUND`
3.  **Optimized Parameters:** `tuned_params.py:TUNED_PARAMS` (overrides `BST_PARAMS` in `client_utils.py` if present)
4.  **Optimized Local Rounds:** `tuned_params.py:NUM_LOCAL_ROUND` (overrides `utils.py:NUM_LOCAL_ROUND` if present)
5.  **Command-Line Arguments:** Parsed by `utils.py` functions (`client_args_parser`, `server_args_parser`, `sim_args_parser`) and specific scripts. These override defaults or tuned parameters where applicable (e.g., `--train-method`).
6.  **Global Feature Processor:** Configured implicitly via `create_global_processor.py` and then loaded by other modules. Its path and fitting state are configuration aspects.

**How to Configure:**

*   **Basic FL:** Modify `run_bagging.sh` or `run_cyclic.sh` scripts or the `sim.py` script directly, or pass command-line arguments when running them (e.g., `./run_bagging.sh --num-rounds 10 --num-clients-per-round 10`).
*   **Hyperparameter Tuning:** Modify `run_ray_tune.sh` or pass arguments like `--num-samples`, `--cpus-per-trial`, `--data-file`, `--output-dir`.
*   **Apply Tuned Params:** Run `python use_tuned_params.py --params-file <path_to_json>` (defaults to `./tune_results/best_params.json`).
*   **Use Saved Model:** Run `python use_saved_model.py --model_path <path> --data_path <path> --output_path <path> [--has_labels]`.
*   **Default XGBoost Params:** Modify `utils.py:BST_PARAMS` (less recommended if using tuning).
*   **Global Processor:** Configure input data and output directory via `create_global_processor.py` arguments.

## 4. Module Documentation

This section details the public interfaces and functionality of the key Python modules in the repository.

### `utils` Module

**Purpose:** Provides shared utility functions, default constants for XGBoost parameters and local rounds, and argument parsers for client, server, and simulation scripts.

**Installation / Import:** Standard Python module import after cloning the repository and installing dependencies (`pip install -r requirements.txt`).
```python
import utils # or from utils import BST_PARAMS, client_args_parser, etc.
```

**Public API:**

*   `NUM_LOCAL_ROUND: int`
    *   **Description:** Default number of local boosting rounds each client performs during fitting. Can be overridden by `tuned_params.py`.
    *   **Value:** 2
*   `BST_PARAMS: dict`
    *   **Description:** Dictionary containing default XGBoost training parameters for multi-class classification (tuned for UNSW_NB15). These parameters can be overridden by `tuned_params.py`.
    *   **Value:**
        ```python
        {
          "objective": "multi:softprob",
          "num_class": 11,
          "eta": 0.05,
          "max_depth": 6,
          "min_child_weight": 10,
          "gamma": 1.0,
          "subsample": 0.7,
          "colsample_bytree": 0.6,
          "colsample_bylevel": 0.6,
          "nthread": 16,
          "tree_method": "hist",
          "eval_metric": ["mlogloss", "merror"],
          "max_delta_step": 5,
          "reg_alpha": 0.8,
          "reg_lambda": 0.8,
          "base_score": 0.5,
          "scale_pos_weight": 1.0,
          "grow_policy": "lossguide",
          "normalize_type": "tree",
          "random_state": 42
        }
        ```
*   `client_args_parser() -> argparse.Namespace`
    *   **Description:** Creates and returns an `argparse.ArgumentParser` configured with standard command-line arguments for the client script (`client.py`).
    *   **Args:** None. Parses `sys.argv`.
    *   **Returns:** Parsed arguments as a `Namespace` object.
    *   **Key Arguments Handled:** `--train-method`, `--num-partitions`, `--partitioner-type`, `--partition-id`, `--seed`, `--test-fraction`, `--centralised-eval`, `--scaled-lr`, `--csv-file`.
*   `server_args_parser() -> argparse.Namespace`
    *   **Description:** Creates and returns an `argparse.ArgumentParser` configured with standard command-line arguments for the server script (`server.py`).
    *   **Args:** None. Parses `sys.argv`.
    *   **Returns:** Parsed arguments as a `Namespace` object.
    *   **Key Arguments Handled:** `--train-method`, `--pool-size`, `--num-rounds`, `--num-clients-per-round`, `--num-evaluate-clients`, `--centralised-eval`.
*   `sim_args_parser() -> argparse.Namespace`
    *   **Description:** Creates and returns an `argparse.ArgumentParser` configured with standard command-line arguments for the simulation script (`sim.py`). Combines arguments for both server and client aspects of the simulation.
    *   **Args:** None. Parses `sys.argv`.
    *   **Returns:** Parsed arguments as a `Namespace` object.
    *   **Key Arguments Handled:** `--train-method`, `--pool-size`, `--num-rounds`, `--num-clients-per-round`, `--num-evaluate-clients`, `--centralised-eval`, `--num-cpus-per-client`, `--partitioner-type`, `--seed`, `--test-fraction`, `--centralised-eval-client`, `--scaled-lr`, `--csv-file`.

**Dependencies:**
*   `argparse`

**Advanced Usage Examples:**

```python
# Example: Override default BST_PARAMS for a specific script
# Note: This requires modifying the script itself or using a wrapper
from utils import BST_PARAMS
# Create a modified version
MY_CUSTOM_PARAMS = BST_PARAMS.copy()
MY_CUSTOM_PARAMS['eta'] = 0.1 # Change learning rate
# Pass MY_CUSTOM_PARAMS to XgbClient constructor in client.py
```

```bash
# Example: Run server with different parameters from the command line
python server.py --num-rounds 20 --num-clients-per-round 10
```

### `dataset` Module

**Purpose:** Provides comprehensive functionality for loading, preprocessing, splitting, partitioning, and transforming network traffic datasets for XGBoost training. Crucially includes the `FeatureProcessor` for consistent data handling and functions to create/load a global processor.

**Installation / Import:** Standard Python module import after cloning the repository and installing dependencies (`pip install -r requirements.txt`).
```python
import dataset # or from dataset import load_csv_data, FeatureProcessor, etc.
```

**Public API:**

*   `CORRELATION_TO_PARTITIONER: dict`
    *   **Description:** Mapping from string names (`"uniform"`, `"linear"`, `"square"`, `"exponential"`) to `flwr_datasets` partitioner classes.
*   `FeatureProcessor` Class
    *   **Description:** Handles feature preprocessing (categorical encoding, numerical scaling/outlier handling) and label encoding while preventing data leakage by fitting only on training data. Supports different dataset types (`"unsw_nb15"`, `"engineered"`).
    *   **`__init__(self, dataset_type="unsw_nb15")`**
        *   **Description:** Initializes the processor.
        *   **Args:**
            *   `dataset_type (str)`: Specifies the dataset schema (`"unsw_nb15"` or `"engineered"`).
    *   **`fit(self, df: pd.DataFrame) -> None`**
        *   **Description:** Fits the preprocessing parameters (encoders, stats) using the provided training DataFrame. This method is idempotent.
        *   **Args:**
            *   `df (pd.DataFrame)`: Training data to fit the processor on.
    *   **`transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame`**
        *   **Description:** Transforms the input DataFrame using the fitted parameters. If `is_training` is True and the processor is not fitted, it will fit first. Handles standardizing label column names and dropping 'id'.
        *   **Args:**
            *   `df (pd.DataFrame)`: Data to transform.
            *   `is_training (bool)`: Flag indicating if the data is for training (influences implicit fitting).
        *   **Returns:** Transformed DataFrame with processed features and standardized 'label' column (if present).
    *   **`is_fitted: bool`**
        *   **Description:** Property indicating if the processor has been fitted.
    *   **`categorical_features: list`**
        *   **Description:** List of identified categorical feature names.
    *   **`numerical_features: list`**
        *   **Description:** List of identified numerical feature names.
*   `preprocess_data(data: Union[pd.DataFrame, Dataset], processor: FeatureProcessor = None, is_training: bool = False) -> Tuple[pd.DataFrame, Union[pd.Series, None]]`
    *   **Description:** High-level preprocessing function. Converts Hugging Face `Dataset` to pandas, applies the `FeatureProcessor`, extracts and encodes the 'label' column (if `is_training`), and removes the 'label' column from features.
    *   **Args:**
        *   `data (Union[pd.DataFrame, Dataset])`: Input data.
        *   `processor (FeatureProcessor, optional)`: An existing processor instance. If None, one is created and fitted (on training data).
        *   `is_training (bool)`: Flag passed to `processor.transform`.
    *   **Returns:** A tuple `(features_df, labels_series)`. `labels_series` is None if no label column is found.
*   `load_csv_data(file_path: str) -> DatasetDict`
    *   **Description:** Loads data from a CSV file into a Hugging Face `DatasetDict`. Attempts temporal splitting based on the 'Stime' column to avoid data leakage. Falls back to stratified random splitting if 'Stime' is not available. Handles unlabeled test sets.
    *   **Args:**
        *   `file_path (str)`: Path to the CSV file.
    *   **Returns:** A `DatasetDict` with 'train' and 'test' splits.
*   `instantiate_partitioner(partitioner_type: str, num_partitions: int) -> Partitioner`
    *   **Description:** Creates and returns a `flwr_datasets` partitioner instance based on the specified type and number of partitions.
    *   **Args:**
        *   `partitioner_type (str)`: Type of partitioner (must be a key in `CORRELATION_TO_PARTITIONER`).
        *   `num_partitions (int)`: Number of partitions.
    *   **Returns:** An initialized `flwr_datasets` Partitioner object.
*   `transform_dataset_to_dmatrix(data, processor: FeatureProcessor = None, is_training: bool = False) -> xgb.DMatrix`
    *   **Description:** Converts input data (DataFrame or Dataset) into an `xgboost.DMatrix` after applying preprocessing. Handles cases with and without labels.
    *   **Args:**
        *   `data`: Input data.
        *   `processor (FeatureProcessor, optional)`: Processor instance. If None, one is created and fitted.
        *   `is_training (bool)`: Flag passed to `preprocess_data`.
    *   **Returns:** An `xgboost.DMatrix`.
*   `train_test_split(data, test_fraction: float = 0.2, random_state: int = 42) -> Tuple[xgb.DMatrix, xgb.DMatrix, FeatureProcessor]`
    *   **Description:** Performs a train/test split on the data, preprocesses both splits using a *single fitted processor*, and returns the resulting DMatrices and the fitted processor. Attempts UID-based splitting if a 'uid' column exists to prevent leakage.
    *   **Args:**
        *   `data`: Input data (Dataset or DataFrame).
        *   `test_fraction (float)`: Fraction for the test set.
        *   `random_state (int)`: Seed for splitting.
    *   **Returns:** Tuple `(train_dmatrix, test_dmatrix, fitted_processor)`.
*   `resplit(dataset: DatasetDict) -> DatasetDict`
    *   **Description:** Adjusts the train/test split in a `DatasetDict`, specifically moving up to 10,000 samples from the training set to the test set. Used to increase centralized test set size.
    *   **Args:**
        *   `dataset (DatasetDict)`: Input dataset dictionary.
    *   **Returns:** Adjusted `DatasetDict`.
*   `create_global_feature_processor(data_file: str, output_dir: str = "outputs") -> str`
    *   **Description:** Loads the full training dataset from a file, creates and fits a `FeatureProcessor` on it, and saves the fitted processor to a pickle file. Ensures consistent preprocessing across different modules.
    *   **Args:**
        *   `data_file (str)`: Path to the dataset CSV file (assumed to contain training data or be split temporally).
        *   `output_dir (str)`: Directory to save the processor file.
    *   **Returns:** Absolute path to the saved processor file.
*   `load_global_feature_processor(processor_path: str) -> FeatureProcessor`
    *   **Description:** Loads a pre-fitted `FeatureProcessor` instance from a pickle file.
    *   **Args:**
        *   `processor_path (str)`: Path to the saved processor file.
    *   **Returns:** The loaded `FeatureProcessor`.

**Dependencies:**
*   `xgboost`
*   `pandas`
*   `numpy`
*   `sklearn` (specifically `sklearn.preprocessing`, `sklearn.model_selection`, `sklearn.compose`, `sklearn.pipeline`)
*   `datasets` (Hugging Face datasets library)
*   `flwr_datasets`
*   `flwr.common.logger`
*   `typing`
*   `pickle`
*   `os`

**Advanced Usage Examples:**

```python
import dataset
import pandas as pd
# Assume 'my_data.csv' is an engineered dataset
df = pd.read_csv("my_data.csv")
# Create and fit processor on the full data before splitting/partitioning for FL
# (This is what create_global_feature_processor does)
processor = dataset.FeatureProcessor(dataset_type="engineered")
processor.fit(df)
# Now you can use this fitted processor to transform data chunks for different clients
# or for prediction
client_data_df = df.sample(frac=0.1) # Example subset for a client
client_dmatrix = dataset.transform_dataset_to_dmatrix(client_data_df, processor=processor, is_training=False)
print(f"Transformed client data shape: {client_dmatrix.num_row()} rows, {client_dmatrix.num_col()} features")
```

```python
# Example: Manually load and preprocess data using the global processor
import dataset
global_processor_path = "outputs/global_feature_processor.pkl"
try:
    global_processor = dataset.load_global_feature_processor(global_processor_path)
    print("Loaded global processor.")
    # Assume you have a new, raw dataframe 'new_unlabeled_df'
    # transformed_df = global_processor.transform(new_unlabeled_df, is_training=False)
    # new_dmatrix = dataset.transform_dataset_to_dmatrix(transformed_df, processor=global_processor, is_training=False)
    # print(f"New data transformed into DMatrix with shape: {new_dmatrix.num_row()} rows, {new_dmatrix.num_col()} features")
except FileNotFoundError:
    print("Global processor not found. Run create_global_processor.py first.")
```

### `client_utils` Module

**Purpose:** Implements the Flower client-side logic for training and evaluating an XGBoost model. Contains the `XgbClient` class, which interfaces with the Flower framework. Handles parameter exchange, local training (`fit`), local evaluation (`evaluate`), and predictions.

**Installation / Import:** Standard Python module import. Used internally by `client.py`.
```python
import client_utils # or from client_utils import XgbClient
```

**Public API:**

*   `BST_PARAMS: dict`
    *   **Description:** Default XGBoost parameters. Imported directly from `utils.py`. Used if no `params` are provided to `XgbClient` constructor and `use_tuned_params` is False or `tuned_params.py` is not found.
    *   **Value:** (See `utils.py`)
*   `TUNED_PARAMS: dict`
    *   **Description:** Optimized XGBoost parameters. Attempted to be imported dynamically from `tuned_params.py` if the file exists. Used if `use_tuned_params` is True in `XgbClient` constructor. Defaults to `BST_PARAMS` if import fails or file is missing.
    *   **Value:** Loaded from `tuned_params.py` or `BST_PARAMS`.
*   `XgbClient` Class
    *   **Description:** A Flower `Client` implementation specifically for training XGBoost models.
    *   **`__init__(self, train_dmatrix, valid_dmatrix, num_train, num_val, num_local_round, cid, params=None, train_method="cyclic", is_prediction_only=False, unlabeled_dmatrix=None, use_tuned_params=True)`**
        *   **Description:** Initializes the XGBoost client with local data, configuration, and ID.
        *   **Args:**
            *   `train_dmatrix (xgb.DMatrix)`: Local training data.
            *   `valid_dmatrix (xgb.DMatrix)`: Local validation data.
            *   `num_train (int)`: Number of training examples.
            *   `num_val (int)`: Number of validation examples.
            *   `num_local_round (int)`: Number of local boosting rounds per FL round.
            *   `cid (str)`: Client ID.
            *   `params (dict, optional)`: XGBoost parameters. If None, uses `TUNED_PARAMS` (if `use_tuned_params` is True) or `BST_PARAMS`.
            *   `train_method (str)`: Training method (`"bagging"` or `"cyclic"`), affects `_local_boost`.
            *   `is_prediction_only (bool)`: If True, the client skips training (`fit`) and only performs evaluation/prediction. (Note: Current `client.py` sets this to False).
            *   `unlabeled_dmatrix (xgb.DMatrix, optional)`: Data for making predictions (if needed).
            *   `use_tuned_params (bool)`: If True, attempts to use `TUNED_PARAMS` loaded from `tuned_params.py` if `params` is None.
    *   **`get_parameters(self, ins: GetParametersIns) -> GetParametersRes`**
        *   **Description:** Implements the Flower `get_parameters` method. Returns empty parameters as XGBoost model parameters are transferred differently (serialized model in `FitRes`).
        *   **Args:**
            *   `ins (GetParametersIns)`: Input from server (ignored).
        *   **Returns:** `GetParametersRes` with empty parameters.
    *   **`fit(self, ins: FitIns) -> FitRes`**
        *   **Description:** Implements the Flower `fit` method. Performs local XGBoost training rounds, optionally updates an existing global model, and serializes the resulting model (or partial model for bagging) for the server. Includes sample weighting for class imbalance and handles first round training.
        *   **Args:**
            *   `ins (FitIns)`: Contains global parameters (if any) and configuration (including `"global_round"`).
        *   **Returns:** `FitRes` containing the local model parameters (as bytes), number of examples used, and empty metrics.
    *   **`evaluate(self, ins: EvaluateIns) -> EvaluateRes`**
        *   **Description:** Implements the Flower `evaluate` method. Loads the global model, evaluates it on the local validation data, calculates multi-class metrics (precision, recall, F1, accuracy, mlogloss, confusion matrix), and saves predictions locally.
        *   **Args:**
            *   `ins (EvaluateIns)`: Contains global parameters and configuration (including `"global_round"` and `"output_dir"`).
        *   **Returns:** `EvaluateRes` containing the evaluation loss (mlogloss), number of examples used, and a dictionary of metrics.
    *   **`_local_boost(self, bst_input) -> xgb.Booster`** (Internal Helper)
        *   **Description:** Performs the actual local boosting rounds on an XGBoost Booster instance. Handles extracting relevant trees for the "bagging" method.
        *   **Args:**
            *   `bst_input (xgb.Booster)`: The input XGBoost model.
        *   **Returns:** The updated `xgb.Booster` instance.

**Dependencies:**
*   `xgboost`
*   `sklearn.metrics`
*   `flwr` (`flwr.client`, `flwr.common`, `flwr.common.logger`)
*   `numpy`
*   `pandas`
*   `os`
*   `importlib.util`
*   `sklearn.utils.class_weight`
*   `server_utils` (for `save_predictions_to_csv`)

**Advanced Usage Examples:**

```python
# Example: Manually instantiate XgbClient with custom parameters (bypassing tuned_params.py)
import client_utils
import xgboost as xgb
# Assume train_dm, valid_dm, num_train, num_val, cid are defined
custom_params = {
    'objective': 'multi:softprob',
    'num_class': 11,
    'eta': 0.01,
    'max_depth': 3,
    # ... other params
}
client = client_utils.XgbClient(
    train_dmatrix=train_dm,
    valid_dmatrix=valid_dm,
    num_train=num_train,
    num_val=num_val,
    num_local_round=5, # Custom local rounds
    cid="client_manual",
    params=custom_params, # Explicitly provide params
    use_tuned_params=False # Ensure tuned params are not used
)
# Then start the client with fl.client.start_client(..., client=client, ...)
```

### `server_utils` Module

**Purpose:** Provides utility functions and classes for the Flower server, including handling output directories, saving results, managing evaluation configuration and aggregation, loading/saving models, making predictions with saved models, and generating visualizations. Includes a custom client manager for the cyclic strategy.

**Installation / Import:** Standard Python module import. Used internally by `server.py` and `client_utils.py`.
```python
import server_utils # or from server_utils import get_evaluate_fn, save_predictions_to_csv, etc.
```

**Public API:**

*   `setup_output_directory() -> str`
    *   **Description:** Creates a unique output directory based on the current date and time (`outputs/YYYY-MM-DD/HH-MM-SS/`). Also creates a `.hydra` subdirectory and copies existing `.hydra` files if found.
    *   **Args:** None.
    *   **Returns:** Path to the created output directory.
*   `save_results_pickle(results, output_dir: str)`
    *   **Description:** Saves a Python object (typically the Flower `history` object or a dictionary containing results) to a pickle file (`results.pkl`) within the specified directory.
    *   **Args:**
        *   `results`: The Python object to save.
        *   `output_dir (str)`: Directory path.
*   `eval_config(rnd: int, output_dir: str = None) -> Dict[str, str]`
    *   **Description:** Generates the configuration dictionary passed to clients for the `evaluate` round. Includes the `global_round` number and optionally the `output_dir`.
    *   **Args:**
        *   `rnd (int)`: Current server round number.
        *   `output_dir (str, optional)`: Output directory path to include in the config.
    *   **Returns:** Configuration dictionary.
*   `save_evaluation_results(eval_metrics: Dict, round_num: int, output_dir: str = None)`
    *   **Description:** Saves a dictionary of evaluation metrics to a JSON file (e.g., `eval_results_round_X.json`) in the specified directory.
    *   **Args:**
        *   `eval_metrics (Dict)`: Dictionary of metrics.
        *   `round_num (int or str)`: Round number or identifier (e.g., "aggregated").
        *   `output_dir (str, optional)`: Directory to save to

--- End of Documentation ---

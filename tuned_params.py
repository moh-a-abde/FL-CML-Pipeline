"""
tuned_params.py

This file contains tuned hyperparameters for the XGBoost model when using the UNSW_NB15 dataset.
These parameters are automatically loaded by client_utils.py when available.

Note: These are starter parameters that should be refined using ray_tune_xgboost.py
"""

# Tuned parameters for UNSW-NB15 multi-class classification
TUNED_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 11,  # Classes: 0-10 (Normal, Reconnaissance, Backdoor, DoS, Exploits, Analysis, Fuzzers, Worms, Shellcode, Generic, plus class 10)
    "eta": 0.03,
    "max_depth": 5,
    "min_child_weight": 5,
    "gamma": 0.8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "nthread": 16,
    "tree_method": "hist",
    "eval_metric": ["mlogloss", "merror"],
    "max_delta_step": 3,
    "reg_alpha": 1.5,
    "reg_lambda": 3.0,
    "base_score": 0.5,
    "scale_pos_weight": 1.0,
    "grow_policy": "lossguide",
    "normalize_type": "tree",
    "random_state": 42
}

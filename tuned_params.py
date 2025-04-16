# This file is generated automatically by use_tuned_params.py
# It contains optimized XGBoost parameters found by Ray Tune

TUNED_PARAMS = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eta': 0.0017574056471531343,
    'max_depth': 5,
    'min_child_weight': 19,
    'gamma': 1.0,
    'subsample': 0.7626600174065583,
    'colsample_bytree': 0.5836140774820693,
    'colsample_bylevel': 0.6,
    'nthread': 16,
    'tree_method': 'hist',
    'eval_metric': ['mlogloss', 'merror'],
    'max_delta_step': 5,
    'reg_alpha': 4.000877452784171,
    'reg_lambda': 6.121688592715271,
    'base_score': 0.5,
    'scale_pos_weight': [1.0, 2.0, 1.0],
    'grow_policy': 'lossguide',
    'normalize_type': 'tree',
    'random_state': 42,
}

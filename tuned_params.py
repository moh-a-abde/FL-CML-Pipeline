# This file is generated automatically by use_tuned_params.py
# It contains optimized XGBoost parameters found by Ray Tune

NUM_LOCAL_ROUND = 2

TUNED_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 11,
    'eta': 0.08320809934829289,
    'max_depth': 9,
    'min_child_weight': 19,
    'gamma': 1.0,
    'subsample': 0.7336272180598685,
    'colsample_bytree': 0.82011802398955,
    'colsample_bylevel': 0.6,
    'nthread': 16,
    'tree_method': 'hist',
    'eval_metric': ['mlogloss', 'merror'],
    'max_delta_step': 5,
    'reg_alpha': 0.0024157333696349283,
    'reg_lambda': 5.069298395764341,
    'base_score': 0.5,
    'scale_pos_weight': 1.0,
    'grow_policy': 'lossguide',
    'normalize_type': 'tree',
    'random_state': 42,
    'num_boost_round': 2,
}

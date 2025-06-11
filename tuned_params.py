# This file is generated automatically by use_tuned_params.py
# It contains optimized XGBoost parameters found by Ray Tune

NUM_LOCAL_ROUND = 100

TUNED_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 11,
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'nthread': 16,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
    'max_delta_step': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'base_score': 0.5,
    'scale_pos_weight': 1.0,
    'grow_policy': 'depthwise',
    'normalize_type': 'tree',
    'random_state': 42,
    'num_boost_round': 100,
}

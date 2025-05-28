# This file is generated automatically by use_tuned_params.py
# It contains optimized XGBoost parameters found by Ray Tune

NUM_LOCAL_ROUND = 82

TUNED_PARAMS = {
    'objective': 'multi:softprob',
    'tree_method': 'hist',
    'eval_metric': ['mlogloss', 'merror'],
    'num_class': 11,
    'random_state': 42,
    'nthread': 16,
    
    'max_depth': 8,
    'min_child_weight': 5,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'num_boost_round': 82,
}

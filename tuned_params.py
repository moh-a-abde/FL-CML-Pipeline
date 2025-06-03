# This file is generated automatically by use_tuned_params.py
# It contains optimized XGBoost parameters found by Ray Tune

NUM_LOCAL_ROUND = 357

TUNED_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 11,
    'eta': 0.22739425647144482,
    'max_depth': 12,
    'min_child_weight': 6,
    'gamma': 4.794063443803187e-07,
    'subsample': 0.8,
    'colsample_bytree': 0.7971063889888208,
    'colsample_bylevel': 0.7632997975138978,
    'nthread': 16,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
    'max_delta_step': 1,
    'reg_alpha': 3.175861162922875e-06,
    'reg_lambda': 0.20611881150439115,
    'base_score': 0.5,
    'scale_pos_weight': 1.0,
    'grow_policy': 'depthwise',
    'normalize_type': 'tree',
    'random_state': 42,
    'colsample_bynode': 0.5591294254162846,
    'num_boost_round': 357,
}

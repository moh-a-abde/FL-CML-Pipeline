# This file is generated automatically by use_tuned_params.py
# It contains optimized XGBoost parameters found by Ray Tune

NUM_LOCAL_ROUND = 251

TUNED_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 11,
    'eta': 0.2170507735414582,
    'max_depth': 7,
    'min_child_weight': 12,
    'gamma': 1.8052644075462313e-06,
    'subsample': 0.37433427868419117,
    'colsample_bytree': 0.9516784621770842,
    'colsample_bylevel': 0.5773301181312036,
    'colsample_bynode': 0.8722242796866464,
    'nthread': 16,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss',
    'max_delta_step': 1,
    'reg_alpha': 0.0001339543724613874,
    'reg_lambda': 459.4035035805314,
    'base_score': 0.5,
    'scale_pos_weight': 5.609712309267583,
    'grow_policy': 'lossguide',
    'max_leaves': 3065,
    'normalize_type': 'tree',
    'random_state': 42,
    'num_boost_round': 251,
}

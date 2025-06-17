# This file is generated automatically by use_tuned_params.py
# It contains optimized XGBoost parameters found by Ray Tune

NUM_LOCAL_ROUND = 702  # Updated from Ray Tune

TUNED_PARAMS = {
    'objective': 'multi:softprob',
    'tree_method': 'hist',
    'eval_metric': ['mlogloss', 'merror'],
    'num_class': 11,
    'random_state': 42,
    'nthread': 16,
    
    'eta': 0.020810960704017334,
    'max_depth': 18,
    'min_child_weight': 5,
    'gamma': 0.9251758546112311,
    'subsample': 0.9756753850290512,
    'colsample_bytree': 0.38159969563216367,
    'colsample_bylevel': 0.9005982104516306,
    'colsample_bynode': 0.6725376242659837,
    'reg_alpha': 2.2769018964159371e-10,
    'reg_lambda': 2.643013384540637e-07,
    'scale_pos_weight': 0.7776869643957478,
    'max_delta_step': 5,
    'grow_policy': 'lossguide',
    'max_leaves': 3065,
    'num_boost_round': 702
}

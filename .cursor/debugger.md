
Overall Goal: Achieve a robust and effective Federated Learning pipeline using XGBoost, where models generalize well, performance improves across rounds, and hyperparameters are meaningfully optimized via Ray Tune.

Diagnosis Summary (Based on Logs & Code Structure):

Severe Client-Side Overfitting: Local models memorize training data (train-merror: ~0) but fail on validation data (validate-merror: ~0.5), indicating poor generalization.
Poor Aggregated Performance: FL model accuracy stagnates near random chance (~50% for 3 classes) with low F1 scores and high logloss.
Model Bias & Class Imbalance: Models heavily favor predicting icmp_tunneling while neglecting benign, likely due to class imbalance or feature dominance.
Ineffective FL Training: Performance metrics show minimal improvement across FL rounds.
Server-Side Aggregation Bugs: Logs indicate errors in calculating/reporting aggregated loss and handling metric formats.
Insufficient HPO: Ray Tune runs (especially in CI) use too few samples (num_samples=3) for effective optimization.
Potential HPO Data Leakage: Risk that the FeatureProcessor in ray_tune_xgboost.py is not correctly isolated within each trial, leading to optimistic HPO results.
Implementation Plan

Part 1: Foundational Robustness - Fixing Core Training & Aggregation

(Priority: Highest - Establish a working baseline first)

Objective: Address overfitting, class imbalance, and server bugs to achieve stable, minimally viable FL training.

Task 1.1: Apply Aggressive Regularization (Client-Side)

Target Files: client.py (within fit method or function called by it), utils.py (if XGBoost params are defined there).
Action: Modify the default XGBoost parameters used during local client training (xgb.train or model.fit). Override any HPO parameters for now.
Set max_depth: 3 (Start low)
Set lambda (L2): 10.0 (Increase significantly)
Set alpha (L1): 2.0 (Increase)
Set min_child_weight: 5 (Increase)
Set subsample: 0.8
Set colsample_bytree: 0.8
Set eta: 0.1
Verification: Run a short FL simulation (2 clients, 3 rounds). Check client logs: Is the gap between train-merror and validate-merror significantly reduced? Does validate-merror drop below ~0.4?
Task 1.2: Implement Early Stopping (Client-Side)

Target File: client.py (within fit method or function called by it).
Action:
Ensure a local validation DMatrix (dval) is available within the fit scope.
Modify the xgb.train call:
Add evals=[(dtrain, 'train'), (dval, 'validate')].
Add early_stopping_rounds=20.
Ensure eval_metric includes mlogloss or merror.
Verification: Run a short FL simulation. Check client logs for "Stopping. Best iteration:" messages. Note the validation performance at the stopping point.
Task 1.3: Implement Sample Weighting for Class Imbalance (Client-Side)

Target File: client.py (within fit method, before xgb.DMatrix creation).
Action:
Get training labels (y_train_processed).
Calculate weights inversely proportional to class frequency in y_train_processed. (e.g., from sklearn.utils.class_weight import compute_sample_weight; sample_weights = compute_sample_weight('balanced', y=y_train_processed)).
Pass weights when creating the training DMatrix: dtrain = xgb.DMatrix(X_train_processed, label=y_train_processed, weight=sample_weights, ...)
Verification: Run a short FL simulation. Examine client evaluation logs (confusion matrix, classification report). Is recall for benign improved? Is the model less biased towards icmp_tunneling?
Task 1.4: Fix Server-Side Aggregated Loss Reporting Bug

Target Files: server.py, server_utils.py, or custom Strategy file.
Action: Locate the code logging Aggregated loss for round.... Debug why it reports 0. Ensure it correctly accesses the aggregated loss value (e.g., metrics['mlogloss'] or metrics['loss']) from the dictionary returned by the aggregation function (likely aggregate_evaluate).
Verification: Run a short FL simulation. Check server logs: Does the Aggregated loss now show a non-zero value reflecting the average client loss?
Task 1.5: Fix Server-Side Metrics Format Bug

Target Files: server.py, server_utils.py, custom Strategy file, potentially client.py (evaluate method).
Action: Find where the Metrics for round ... is not a dictionary: <class 'tuple'> error originates. Ensure the function performing metric aggregation (e.g., aggregate_evaluate) returns, or the subsequent logging function receives, a flat dictionary Dict[str, Scalar] (e.g., {'accuracy': 0.6, 'f1': 0.55, 'mlogloss': 0.9}). Remove the workaround code that creates original_metrics.
Verification: Run a short FL simulation. Check server logs: Is the not a dictionary error gone? Are aggregated metrics logged correctly by Flower?
Task 1.6: Baseline Performance Check

Action: Run a standard FL simulation (e.g., run_bagging.sh for 5-10 rounds) with all fixes from Part 1 applied.
Verification: Document baseline performance:
Reduced train vs. validate gap in client logs?
Aggregated accuracy/F1 significantly above 50%?
Does performance improve across rounds?
Is class bias reduced in evaluation reports?
Are server log errors fixed?
Part 2: Effective Hyperparameter Optimization (Ray Tune Enhancement)

(Priority: Medium - Refine HPO once the core training is stable)

Objective: Ensure Ray Tune is configured correctly, avoids data leakage, and runs with sufficient intensity to find genuinely useful hyperparameters.

Task 2.1: Isolate Data Handling within Ray Tune Objective

Target File: ray_tune_xgboost.py (specifically the train_xgboost objective function).
Action:
Move data loading (pd.read_csv or load_csv_data) inside the train_xgboost function.
Instantiate FeatureProcessor inside train_xgboost.
fit the processor only on the HPO training split created within that trial.
transform both HPO training and validation splits using the trial-specific fitted processor.
Convert to DMatrix inside the trial after transformation.
(Sub-task): Decide on HPO data split strategy:
Option A: Use --train-file for HPO-train, --test-file for HPO-validation (ensure test has labels).
Option B (Recommended): Load only --train-file, split it inside train_xgboost into HPO-train/HPO-validation (e.g., 80/20), leaving --test-file unused by HPO for later final model evaluation.
Verification: Code review confirms data loading/processing is fully encapsulated within the objective function. Add logging inside train_xgboost to confirm processor fitting happens in each trial.
Task 2.2: Standardize HPO Preprocessing

Target Files: ray_tune_xgboost.py, client.py, dataset.py.
Action: Ensure the exact preprocessing steps (including FeatureProcessor logic, handling of NaNs/infs, feature selection/dropping, sample weighting if applied before model training) used within the Ray Tune train_xgboost objective function identically match the steps used in the client's fit method during FL. Use shared functions from dataset.py or utils.py where possible.
Verification: Code review comparing preprocessing steps in ray_tune_xgboost.py::train_xgboost and client.py::fit.
Task 2.3: Configure Ray Tune Objective, Metric & Mode

Target File: ray_tune_xgboost.py.
Action:
Ensure train_xgboost uses parameters from the config dictionary passed by Ray Tune.
Ensure num_class is correctly set in XGBoost parameters.
Choose the primary metric to optimize (e.g., mlogloss - recommended, or accuracy).
Report this metric using tune.report(chosen_metric=value).
Set metric in tune.run and ASHAScheduler to match the reported key (e.g., metric="mlogloss").
Set mode to "min" for loss/error or "max" for accuracy/F1.
Verification: Run a minimal Ray Tune test (num_samples=2). Check logs: Does tune.report send the correct key? Do tune.run and scheduler use the matching metric/mode?
Task 2.4: Refine HPO Search Space

Target File: ray_tune_xgboost.py (the config / search_space dictionary).
Action: Review and adjust the hyperparameter search space:
Include key parameters: eta, max_depth, subsample, colsample_bytree, lambda, alpha, min_child_weight, potentially gamma.
Use appropriate distributions: tune.loguniform for rates/regularizers, tune.randint or tune.choice for counts/depths.
Set reasonable initial ranges (e.g., eta: tune.loguniform(1e-3, 0.2), max_depth: tune.randint(3, 8), lambda/alpha: tune.loguniform(0.1, 20.0)).
Verification: Code review of the search space definition.
Task 2.5: Configure HPO Scheduler

Target File: ray_tune_xgboost.py.
Action: Tune the ASHAScheduler parameters:
Set grace_period: Allow models a reasonable number of rounds (e.g., 15-25) before being considered for stopping. Needs coordination with early_stopping_rounds if used inside the objective as well. Often, rely primarily on ASHA's stopping during HPO.
Set reduction_factor: Typically 2 or 3.
Verification: Observe trial promotion/stopping behavior during a longer Ray Tune run.
Task 2.6: Add Reproducibility to HPO Trials

Target File: ray_tune_xgboost.py (train_xgboost function).
Action: Set random seeds for numpy, random, and XGBoost (seed parameter) at the beginning of the train_xgboost function, possibly using a seed derived from the trial ID or passed via config.
Verification: While hard to verify perfectly, running the same trial config multiple times (if possible via debugging) should yield identical results.
Task 2.7: Run Intensive HPO Locally

Target Files: run_ray_tune.sh.
Action: Execute run_ray_tune.sh locally (not in CI) with a significantly increased --num-samples value (e.g., --num-samples 100 or more). Ensure it uses data representative of the FL clients.
Verification: Ray Tune completes the specified number of trials. tune_results/best_params.json contains the best found hyperparameters. Analyze results using Ray Tune's tools or logs.
Part 3: Validation and Integration

(Priority: High - Verify the effectiveness of fixes and optimization)

Objective: Confirm that the robustness fixes and optimized hyperparameters lead to improved and reliable FL performance.

Task 3.1: Validate Best HPO Parameters

Target Files: use_tuned_params.py, run_bagging.sh.
Action:
Ensure use_tuned_params.py correctly loads best_params.json and makes them available to the FL process (e.g., updates a config file or sets environment variables used by client.py).
Run the full FL simulation (run_bagging.sh) using the best parameters found from the intensive local HPO run (Task 2.7).
Verification: Compare the performance (accuracy, F1, logloss improvement over rounds, final confusion matrix) of this run against the baseline run from Task 1.6. Is there a significant improvement?
Task 3.2: Ensure HPO Data Representativeness

Target Files: run_ray_tune.sh, data preparation scripts/logic.
Action: Review the data specified by --train-file and --test-file used for the intensive local HPO run. Does its distribution (features, class balance) reasonably reflect the combined or average distribution across FL clients? If not, consider creating a dedicated, representative dataset for HPO.
Verification: Statistical analysis (feature distributions, class counts) comparing HPO dataset vs. client datasets.
Task 3.3: Final Code Review and Cleanup

Action: Review all modified files (client.py, server.py, utils.py, dataset.py, ray_tune_xgboost.py, run_ray_tune.sh, cml.yaml, strategy file) for clarity, consistency, comments, and removal of dead code/temporary workarounds. Ensure CI (cml.yaml) still runs correctly (using low num_samples for HPO test).
Verification: Code passes linting and review checks. CI pipeline completes successfully.
Code Integration Notes:

Client Logic: Most robustness changes (regularization, early stopping, weighting) happen within the FlowerClient's fit method in client.py or helper functions it calls (potentially in utils.py).
Server Logic: Aggregation bug fixes target server.py or a custom Flower Strategy defined therein or in server_utils.py.
HPO Script: Ray Tune enhancements target ray_tune_xgboost.py and its execution script run_ray_tune.sh.
Data Handling: Preprocessing consistency might require changes in dataset.py, ensuring both client.py and ray_tune_xgboost.py use the same, updated logic.
Configuration: How parameters (default, HPO) are passed to the client needs review (e.g., via Flower config, separate files like params.yaml loaded by use_tuned_params.py, or environment variables).
Agentic Breakdown Considerations:

Prioritization: Execute Part 1 tasks sequentially first. Verify Task 1.6 thoroughly before starting Part 2.
Sub-Tasking: Complex tasks like "Implement Sample Weighting" (1.3) or "Isolate Data Handling" (2.1) might be broken down further (e.g., 1.3a: calculate weights, 1.3b: integrate weights into DMatrix creation).
Verification: Treat each "Verification" step as a mandatory checkpoint. Do not proceed if verification fails; loop back to debug the relevant action.
Code Location: Use file paths provided as primary targets for modifications.
Testing: After each significant change (e.g., adding regularization, fixing an aggregation bug), run short (2-client, 3-round) FL simulations to test the specific change before moving on. Reserve full runs for major checkpoints (1.6, 3.1).
Parameter Management: Be careful how default parameters and HPO parameters are managed and applied. Ensure the intended set is active during each phase (robustness testing vs. HPO vs. final validation).






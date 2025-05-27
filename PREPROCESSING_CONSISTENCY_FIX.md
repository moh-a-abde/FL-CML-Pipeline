# Preprocessing Consistency Fix

## Problem Fixed

Your federated learning pipeline was experiencing disconnection between two phases:

1. **Individual Client Training** (showing good learning curves with decreasing mlogloss)
2. **Federated Learning** (showing stagnant accuracy at 0.5453)

## Root Cause

The disconnection was caused by **inconsistent preprocessing** across different phases:

- **Ray Tune optimization**: Created its own `FeatureProcessor` for each trial
- **Federated Learning clients**: Each created their own `FeatureProcessor` fitted on local data
- **Server evaluation**: Created its own `FeatureProcessor` for centralized evaluation

This led to different scaling, encoding, and feature transformations across phases, making the hyperparameters learned during Ray Tune ineffective for federated learning.

## Solution Implemented

### 1. Global Feature Processor

Created a **single, global `FeatureProcessor`** that:
- Is fitted on the **full training dataset** (`data/received/final_dataset.csv`)
- Uses **temporal splitting** to prevent data leakage
- Is saved to `outputs/global_feature_processor.pkl`
- Is shared across **all phases** (Ray Tune, Server, Clients)

### 2. Updated Scripts

#### A. `run_bagging.sh` (Primary Fix)
Updated to create the global processor before starting federated learning:

```bash
# Step 1: Create global feature processor
python create_global_processor.py \
    --data-file "data/received/final_dataset.csv" \
    --output-dir "outputs" \
    --force

# Step 2: Start server (uses global processor)
python3 server.py --pool-size=5 --num-rounds=5 --num-clients-per-round=5 --centralised-eval

# Step 3: Start clients (use global processor)
python3 client.py --partition-id=0 --num-partitions=5 --partitioner-type=exponential
# ... (all clients)
```

#### B. GitHub Workflow (`.github/workflows/cml.yaml`)
Updated to use the same `final_dataset.csv` for both Ray Tune and federated learning:

```yaml
# Ray Tune step now uses the same dataset as federated learning
bash run_ray_tune.sh --data-file "data/received/final_dataset.csv" --num-samples 5

# Federated learning step uses the same dataset
./run_bagging.sh
```

#### C. `create_global_processor.py`
New script that creates the global processor:

```bash
python create_global_processor.py \
    --data-file "data/received/final_dataset.csv" \
    --output-dir "outputs" \
    --force
```

## Key Improvements

### ✅ Consistent Preprocessing
- **Same feature scaling** across all phases
- **Same categorical encoding** across all phases  
- **Same data transformations** across all phases

### ✅ Temporal Data Integrity
- **No data leakage** between train/test splits
- **Temporal splitting** based on `Stime` column
- **Proper train/validation separation**

### ✅ Phase Alignment
- **Ray Tune** uses same preprocessing as **Federated Learning**
- **Individual client training** matches **federated aggregation**
- **Hyperparameters** transfer effectively between phases

## Usage Instructions

### For GitHub Actions Workflow
Your workflow automatically runs the fixed version:

1. **Ray Tune Step**: Creates global processor and tunes hyperparameters
2. **Federated Learning Step**: Uses same global processor for training
3. **Results**: Should now show consistent performance across phases

### For Local Development

#### Option 1: Use the workflow scripts (Recommended)
```bash
# Run the complete pipeline with consistent preprocessing
source venv/bin/activate
./run_bagging.sh
```

#### Option 2: Manual step-by-step
```bash
# Step 1: Create global processor
python create_global_processor.py \
    --data-file "data/received/final_dataset.csv" \
    --output-dir "outputs"

# Step 2: Run Ray Tune (optional)
bash run_ray_tune.sh --data-file "data/received/final_dataset.csv"

# Step 3: Run federated learning
./run_bagging.sh
```

## Expected Results

After this fix, you should see:

1. **Consistent Learning Curves**: Both individual client training and federated learning should show similar learning patterns
2. **Improved Federated Performance**: Accuracy should improve beyond the previous stagnant 0.5453
3. **Effective Hyperparameter Transfer**: Parameters tuned in Ray Tune should work effectively in federated learning
4. **Reproducible Results**: Same preprocessing ensures consistent results across runs

## Files Modified

1. **`run_bagging.sh`** - Added global processor creation step
2. **`.github/workflows/cml.yaml`** - Updated to use consistent dataset
3. **`dataset.py`** - Added global processor creation/loading functions
4. **`create_global_processor.py`** - New script for processor creation
5. **`ray_tune_xgboost_updated.py`** - Updated to use global processor
6. **`client.py`** - Updated to load global processor
7. **`server.py`** - Updated to use global processor

## Validation

To verify the fix is working:

1. **Check processor creation**: Look for `✓ Global feature processor ready at outputs/global_feature_processor.pkl`
2. **Monitor learning curves**: Both phases should show similar learning patterns
3. **Check accuracy progression**: Federated learning accuracy should improve over rounds
4. **Verify consistency**: Individual client metrics should align with federated metrics

## Troubleshooting

### Issue: "Global feature processor not found"
**Solution**: Ensure `run_bagging.sh` is being used, which creates the processor automatically.

### Issue: "Still seeing stagnant accuracy"
**Check**: 
- Verify `outputs/global_feature_processor.pkl` exists
- Check that all clients and server are loading the same processor
- Ensure temporal splitting is working correctly

### Issue: "Ray Tune parameters not transferring"
**Check**:
- Both Ray Tune and federated learning are using `data/received/final_dataset.csv`
- Global processor is created before both phases
- `tune_results/best_params.json` exists and is being applied 
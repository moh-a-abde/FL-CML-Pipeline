#!/usr/bin/env python3

"""
Test script to validate Fix 4: Class Schema Inconsistency fix.

This script tests:
1. All configuration files have num_class=11 (matching dataset)
2. Dataset actually has 11 classes (0-10)
3. XGBoost parameters are consistent across all files
4. No class schema mismatches
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path (go up two levels from tests/unit/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import local modules
from src.core.dataset import load_csv_data
from src.config.legacy_constants import BST_PARAMS
from src.config.tuned_params import TUNED_PARAMS

def test_utils_num_class():
    """Test that utils.py has num_class=11."""
    print("Testing utils.py num_class configuration...")
    
    try:
        from src.config.legacy_constants import BST_PARAMS
        
        num_class = BST_PARAMS.get('num_class')
        if num_class == 11:
            print(f"✅ utils.py: num_class = {num_class} (correct)")
            return True
        else:
            print(f"❌ utils.py: num_class = {num_class} (expected 11)")
            return False
            
    except Exception as e:
        print(f"❌ Error importing utils.py: {e}")
        return False

def test_ray_tune_num_class():
    """Test that ray_tune_xgboost_updated.py has num_class=11."""
    print("Testing ray_tune_xgboost_updated.py num_class configuration...")
    
    try:
        # Read the file and check for num_class: 11
        with open('src/tuning/ray_tune_xgboost.py', 'r') as f:
            content = f.read()
            
        # Count occurrences of num_class: 11
        count_11 = content.count("'num_class': 11")
        count_10 = content.count("'num_class': 10")
        
        if count_11 >= 2 and count_10 == 0:
            print(f"✅ ray_tune_xgboost_updated.py: Found {count_11} instances of num_class=11, {count_10} instances of num_class=10")
            return True
        else:
            print(f"❌ ray_tune_xgboost_updated.py: Found {count_11} instances of num_class=11, {count_10} instances of num_class=10")
            return False
            
    except Exception as e:
        print(f"❌ Error reading ray_tune_xgboost_updated.py: {e}")
        return False

def test_tuned_params_num_class():
    """Test that tuned_params.py has num_class=11."""
    print("Testing tuned_params.py num_class configuration...")
    
    try:
        from src.config.tuned_params import TUNED_PARAMS
        
        num_class = TUNED_PARAMS.get('num_class')
        if num_class == 11:
            print(f"✅ tuned_params.py: num_class = {num_class} (correct)")
            return True
        else:
            print(f"❌ tuned_params.py: num_class = {num_class} (expected 11)")
            return False
            
    except Exception as e:
        print(f"❌ Error importing tuned_params.py: {e}")
        return False

def test_dataset_classes():
    """Test that the dataset actually has 11 classes."""
    print("Testing dataset class count...")
    
    try:
        # Load the dataset
        dataset_path = "data/received/final_dataset.csv"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            return False
            
        dataset = load_csv_data(dataset_path)
        
        # Check train and test classes
        train_classes = set(dataset["train"]["label"])
        test_classes = set(dataset["test"]["label"])
        all_classes = train_classes.union(test_classes)
        
        if len(all_classes) == 11 and min(all_classes) == 0 and max(all_classes) == 10:
            print(f"✅ Dataset: Found {len(all_classes)} classes: {sorted(all_classes)}")
            print(f"   Train classes: {len(train_classes)}, Test classes: {len(test_classes)}")
            return True
        else:
            print(f"❌ Dataset: Found {len(all_classes)} classes: {sorted(all_classes)}")
            print(f"   Expected 11 classes [0-10]")
            return False
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_xgboost_compatibility():
    """Test that XGBoost can handle the configuration without errors."""
    print("Testing XGBoost compatibility...")
    
    try:
        import xgboost as xgb
        import numpy as np
        from src.config.legacy_constants import BST_PARAMS
        
        # Create dummy data with 11 classes
        n_samples = 100
        n_features = 20
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 11, n_samples)  # Classes 0-10
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Test parameters
        params = BST_PARAMS.copy()
        params['num_boost_round'] = 5  # Small number for quick test
        
        # Try to train (should not raise an error)
        model = xgb.train(params, dtrain, num_boost_round=5, verbose_eval=False)
        
        # Test prediction
        predictions = model.predict(dtrain)
        
        # Check prediction shape (should be n_samples x 11)
        expected_shape = (n_samples, 11)
        if predictions.shape == expected_shape:
            print(f"✅ XGBoost compatibility: Predictions shape {predictions.shape} (correct)")
            return True
        else:
            print(f"❌ XGBoost compatibility: Predictions shape {predictions.shape} (expected {expected_shape})")
            return False
            
    except Exception as e:
        print(f"❌ XGBoost compatibility error: {e}")
        return False

def test_client_utils_consistency():
    """Test that client_utils.py has consistent num_class."""
    print("Testing client_utils.py consistency...")
    
    try:
        # Read the file and check for num_class
        with open('src/federated/client_utils.py', 'r') as f:
            content = f.read()
            
        if "'num_class': 11" in content:
            print("✅ client_utils.py: num_class = 11 (correct)")
            return True
        else:
            print("❌ client_utils.py: num_class not set to 11")
            return False
            
    except Exception as e:
        print(f"❌ Error reading client_utils.py: {e}")
        return False

def run_all_tests():
    """Run all Fix 4 tests."""
    print("=" * 60)
    print("TESTING FIX 4: CLASS SCHEMA INCONSISTENCY")
    print("=" * 60)
    
    tests = [
        test_utils_num_class,
        test_ray_tune_num_class,
        test_tuned_params_num_class,
        test_dataset_classes,
        test_xgboost_compatibility,
        test_client_utils_consistency,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print("FIX 4 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Fix 4 (Class Schema Inconsistency) is working correctly.")
        print("✅ num_class=11 is consistent across all files")
        print("✅ Dataset has 11 classes (0-10)")
        print("✅ XGBoost can handle the configuration")
        return True
    else:
        print("❌ Some tests failed. Fix 4 needs attention.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 
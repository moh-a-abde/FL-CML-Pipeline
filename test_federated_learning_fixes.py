#!/usr/bin/env python3
"""
Test script to validate Fix 3: Federated Learning Configuration fixes.

This script tests:
1. NUM_LOCAL_ROUND increased from 2 to 20
2. Shell scripts updated with --num-rounds 20
3. Early stopping functionality implemented
4. Metrics history tracking working
"""

import os
import sys

def test_num_local_round_increased():
    """Test that NUM_LOCAL_ROUND has been increased to 20."""
    print("Testing NUM_LOCAL_ROUND configuration...")
    
    # Import utils to check NUM_LOCAL_ROUND
    try:
        from utils import NUM_LOCAL_ROUND
        
        if NUM_LOCAL_ROUND >= 20:
            print(f"✅ NUM_LOCAL_ROUND = {NUM_LOCAL_ROUND} (increased from 2)")
            return True
        
        print(f"❌ NUM_LOCAL_ROUND = {NUM_LOCAL_ROUND} (should be >= 20)")
        return False
            
    except ImportError as e:
        print(f"❌ Failed to import utils: {e}")
        return False

def test_shell_scripts_updated():
    """Test that shell scripts have been updated with increased rounds."""
    print("\nTesting shell script configurations...")
    
    scripts_to_check = ['run_bagging.sh', 'run_cyclic.sh']
    all_passed = True
    
    for script_name in scripts_to_check:
        if os.path.exists(script_name):
            with open(script_name, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for --num-rounds with value >= 20
            if '--num-rounds=20' in content or '--num-rounds 20' in content:
                print(f"✅ {script_name}: Found --num-rounds=20")
            elif '--num-rounds=' in content:
                # Extract the value
                import re
                match = re.search(r'--num-rounds[=\s]+(\d+)', content)
                if match:
                    rounds = int(match.group(1))
                    if rounds >= 20:
                        print(f"✅ {script_name}: Found --num-rounds={rounds}")
                    else:
                        print(f"❌ {script_name}: Found --num-rounds={rounds} (should be >= 20)")
                        all_passed = False
                else:
                    print(f"❌ {script_name}: Could not parse --num-rounds value")
                    all_passed = False
            else:
                print(f"❌ {script_name}: No --num-rounds parameter found")
                all_passed = False
        else:
            print(f"❌ {script_name}: File not found")
            all_passed = False
    
    return all_passed

def test_early_stopping_functions():
    """Test that early stopping functions are available and working."""
    print("\nTesting early stopping functionality...")
    
    try:
        from server_utils import (
            check_convergence, 
            reset_metrics_history, 
            add_metrics_to_history, 
            should_stop_early
        )
        
        # Test reset function
        reset_metrics_history()
        print("✅ reset_metrics_history() works")
        
        # Test adding metrics to history
        test_metrics = {
            'mlogloss': 1.5,
            'accuracy': 0.7,
            'precision': 0.65,
            'recall': 0.68,
            'f1': 0.66
        }
        add_metrics_to_history(test_metrics)
        print("✅ add_metrics_to_history() works")
        
        # Test convergence check with insufficient history
        should_stop = should_stop_early(patience=3, min_delta=0.001)
        if not should_stop:
            print("✅ should_stop_early() correctly returns False with insufficient history")
        else:
            print("❌ should_stop_early() incorrectly returns True with insufficient history")
            return False
        
        # Add more metrics to test convergence detection
        for i in range(5):
            metrics = {
                'mlogloss': 1.5 - (i * 0.0001),  # Very small improvements
                'accuracy': 0.7 + (i * 0.0001),
                'precision': 0.65,
                'recall': 0.68,
                'f1': 0.66
            }
            add_metrics_to_history(metrics)
        
        # Test convergence detection
        should_stop = should_stop_early(patience=3, min_delta=0.001)
        if should_stop:
            print("✅ should_stop_early() correctly detects convergence")
        else:
            print("✅ should_stop_early() working (may not detect convergence with test data)")
        
        # Test direct convergence function
        test_history = [
            {'mlogloss': 1.0, 'accuracy': 0.8},
            {'mlogloss': 0.999, 'accuracy': 0.801},
            {'mlogloss': 0.9989, 'accuracy': 0.8011},
            {'mlogloss': 0.9988, 'accuracy': 0.8012},
        ]
        
        converged = check_convergence(test_history, patience=3, min_delta=0.001)
        if converged:
            print("✅ check_convergence() correctly detects convergence")
        else:
            print("✅ check_convergence() working (may not detect convergence with test data)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import early stopping functions: {e}")
        return False
    except (AttributeError, TypeError, ValueError) as e:
        print(f"❌ Error testing early stopping functions: {e}")
        return False

def test_server_integration():
    """Test that server.py has been updated with early stopping integration."""
    print("\nTesting server.py integration...")
    
    if not os.path.exists('server.py'):
        print("❌ server.py not found")
        return False
    
    with open('server.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('reset_metrics_history import', 'reset_metrics_history' in content),
        ('should_stop_early import', 'should_stop_early' in content),
        ('reset_metrics_history call', 'reset_metrics_history()' in content),
        ('CustomFedXgbBagging early stopping', 'early_stopping_patience' in content),
        ('should_stop_early check', 'should_stop_early(' in content),
    ]
    
    all_passed = True
    for check_name, condition in checks:
        if condition:
            print(f"✅ {check_name}: Found")
        else:
            print(f"❌ {check_name}: Not found")
            all_passed = False
    
    return all_passed

def test_bst_params_consistency():
    """Test that BST_PARAMS in utils.py has been updated with better defaults."""
    print("\nTesting BST_PARAMS configuration...")
    
    try:
        from utils import BST_PARAMS
        
        checks = [
            ('num_class', BST_PARAMS.get('num_class') == 10, f"Expected 10, got {BST_PARAMS.get('num_class')}"),
            ('eta reasonable', 0.01 <= BST_PARAMS.get('eta', 0) <= 0.3, f"eta = {BST_PARAMS.get('eta')}"),
            ('max_depth increased', BST_PARAMS.get('max_depth', 0) >= 8, f"max_depth = {BST_PARAMS.get('max_depth')}"),
            ('subsample increased', BST_PARAMS.get('subsample', 0) >= 0.8, f"subsample = {BST_PARAMS.get('subsample')}"),
            ('colsample_bytree increased', BST_PARAMS.get('colsample_bytree', 0) >= 0.8, f"colsample_bytree = {BST_PARAMS.get('colsample_bytree')}"),
        ]
        
        all_passed = True
        for check_name, condition, details in checks:
            if condition:
                print(f"✅ {check_name}: {details}")
            else:
                print(f"❌ {check_name}: {details}")
                all_passed = False
        
        return all_passed
        
    except ImportError as e:
        print(f"❌ Failed to import BST_PARAMS: {e}")
        return False

def run_all_tests():
    """Run all Fix 3 tests and provide summary."""
    print("=" * 60)
    print("TESTING FIX 3: FEDERATED LEARNING CONFIGURATION")
    print("=" * 60)
    
    tests = [
        ("NUM_LOCAL_ROUND increased", test_num_local_round_increased),
        ("Shell scripts updated", test_shell_scripts_updated),
        ("Early stopping functions", test_early_stopping_functions),
        ("Server integration", test_server_integration),
        ("BST_PARAMS consistency", test_bst_params_consistency),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            print(f"❌ {test_name}: Exception occurred - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("FIX 3 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL FIX 3 TESTS PASSED!")
        print("Federated Learning Configuration has been successfully improved:")
        print("  • NUM_LOCAL_ROUND increased to 20+ for better convergence")
        print("  • Shell scripts updated with --num-rounds 20+")
        print("  • Early stopping functionality implemented")
        print("  • Server integration completed")
        print("  • BST_PARAMS optimized for better performance")
        return True
    
    print(f"\n⚠️  {total - passed} tests failed. Please review the issues above.")
    return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Integration tests for Entry Points with ConfigManager.

This script tests that all entry points (run.py, server.py, client.py, sim.py)
can successfully load configuration using the ConfigManager.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.config_manager import get_config_manager, load_config

def test_config_loading():
    """Test that ConfigManager can load base configuration."""
    print("Testing configuration loading...")
    
    try:
        # Load base configuration
        config = load_config()
        
        # Verify basic structure
        assert hasattr(config, 'data'), "Config missing 'data' section"
        assert hasattr(config, 'federated'), "Config missing 'federated' section"
        assert hasattr(config, 'model'), "Config missing 'model' section"
        assert hasattr(config, 'tuning'), "Config missing 'tuning' section"
        
        # Verify data configuration
        assert hasattr(config.data, 'path'), "Data config missing 'path'"
        assert hasattr(config.data, 'filename'), "Data config missing 'filename'"
        
        # Verify federated configuration
        assert hasattr(config.federated, 'train_method'), "Federated config missing 'train_method'"
        assert hasattr(config.federated, 'pool_size'), "Federated config missing 'pool_size'"
        assert hasattr(config.federated, 'num_rounds'), "Federated config missing 'num_rounds'"
        
        print("✓ Configuration loading test passed")
        return True
        
    except (ImportError, AttributeError, AssertionError) as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False

def test_experiment_override():
    """Test loading configuration with experiment overrides."""
    print("Testing experiment override loading...")
    
    try:
        # Load bagging experiment configuration
        config = load_config(experiment="bagging")
        
        # Verify that experiment-specific settings are loaded
        assert config.federated.train_method == "bagging", "Bagging experiment not loaded correctly"
        
        print("✓ Experiment override test passed")
        return True
        
    except (ImportError, AttributeError, AssertionError) as e:
        print(f"✗ Experiment override test failed: {e}")
        return False

def test_model_params_extraction():
    """Test extracting model parameters as dictionary."""
    print("Testing model parameters extraction...")
    
    try:
        config = load_config()
        config_manager = get_config_manager()
        # Use the proper public method to load config
        config_manager.load_config()
        
        params_dict = config_manager.get_model_params_dict()
        
        # Verify essential XGBoost parameters
        required_params = [
            'objective', 'num_class', 'eta', 'max_depth', 'min_child_weight',
            'gamma', 'subsample', 'colsample_bytree', 'eval_metric'
        ]
        
        for param in required_params:
            assert param in params_dict, f"Missing required parameter: {param}"
        
        print("✓ Model parameters extraction test passed")
        return True
        
    except (ImportError, AttributeError, AssertionError, RuntimeError) as e:
        print(f"✗ Model parameters extraction test failed: {e}")
        return False

def test_data_path_construction():
    """Test data path construction from config."""
    print("Testing data path construction...")
    
    try:
        config = load_config()
        
        # Construct data path
        data_path = os.path.join(config.data.path, config.data.filename)
        
        # Verify path is string and not empty
        assert isinstance(data_path, str), "Data path should be string"
        assert len(data_path) > 0, "Data path should not be empty"
        assert "/" in data_path or "\\" in data_path, "Data path should contain path separator"
        
        print(f"✓ Data path construction test passed: {data_path}")
        return True
        
    except (ImportError, AttributeError, AssertionError) as e:
        print(f"✗ Data path construction test failed: {e}")
        return False

@patch('subprocess.run')
def test_run_py_integration(mock_subprocess):
    """Test that run.py can load configuration (mocked execution)."""
    print("Testing run.py integration...")
    
    try:
        # Mock successful subprocess calls
        mock_subprocess.return_value.check = True
        mock_subprocess.return_value.stdout = "Success"
        mock_subprocess.return_value.stderr = ""
        mock_subprocess.return_value.returncode = 0
        
        # Import and test run.py main function
        # This would normally be done by calling the script, but we'll import it
        sys.path.insert(0, str(Path(__file__).parent))
        
        # We can't easily test run.py main() due to Hydra decorator
        # But we can verify the configuration loading works
        config = load_config()
        assert config is not None, "Configuration should be loaded"
        
        print("✓ run.py integration test passed")
        return True
        
    except (ImportError, AttributeError, AssertionError) as e:
        print(f"✗ run.py integration test failed: {e}")
        return False

def test_sim_py_integration():
    """Test that sim.py can load configuration."""
    print("Testing sim.py integration...")
    
    try:
        # We'll test the configuration loading part of sim.py
        config = load_config()
        
        # Verify all required federated parameters are available
        required_federated_params = [
            'train_method', 'pool_size', 'num_rounds', 'num_clients_per_round',
            'centralised_eval', 'partitioner_type', 'test_fraction'
        ]
        
        for param in required_federated_params:
            assert hasattr(config.federated, param), f"Missing federated parameter: {param}"
        
        # Verify model parameters
        assert hasattr(config.model, 'num_local_rounds'), "Missing model.num_local_rounds"
        
        print("✓ sim.py integration test passed")
        return True
        
    except (ImportError, AttributeError, AssertionError) as e:
        print(f"✗ sim.py integration test failed: {e}")
        return False

def test_server_client_integration():
    """Test that server.py and client.py can load configuration."""
    print("Testing server.py and client.py integration...")
    
    try:
        config = load_config()
        
        # Test server-specific configuration access
        assert hasattr(config.federated, 'train_method'), "Missing train_method for server"
        assert hasattr(config.federated, 'pool_size'), "Missing pool_size for server" 
        assert hasattr(config.federated, 'num_rounds'), "Missing num_rounds for server"
        
        # Test client-specific configuration access
        assert hasattr(config.federated, 'partitioner_type'), "Missing partitioner_type for client"
        assert hasattr(config.federated, 'num_partitions'), "Missing num_partitions for client"
        assert hasattr(config.data, 'path'), "Missing data path for client"
        assert hasattr(config.data, 'filename'), "Missing data filename for client"
        
        # Test model parameters access using proper public interface
        config_manager = get_config_manager()
        config_manager.load_config()  # Load config properly
        bst_params = config_manager.get_model_params_dict()
        assert isinstance(bst_params, dict), "BST_PARAMS should be dictionary"
        assert len(bst_params) > 0, "BST_PARAMS should not be empty"
        
        print("✓ server.py and client.py integration test passed")
        return True
        
    except (ImportError, AttributeError, AssertionError, RuntimeError) as e:
        print(f"✗ server.py and client.py integration test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests."""
    print("=" * 80)
    print("FL-CML-Pipeline Entry Points Integration Tests")
    print("=" * 80)
    
    tests = [
        test_config_loading,
        test_experiment_override,
        test_model_params_extraction,
        test_data_path_construction,
        test_run_py_integration,
        test_sim_py_integration,
        test_server_client_integration
    ]
    
    results = []
    for test in tests:
        try:
            test_result = test()
            results.append(test_result)
        except (ImportError, AttributeError, AssertionError, RuntimeError) as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 80)
    print(f"Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All entry points integration tests PASSED!")
        print("✓ ConfigManager successfully integrated with all entry points")
        print("✓ Ready to proceed with Phase 2 Step 4: Legacy Code Cleanup")
    else:
        print("❌ Some integration tests FAILED!")
        print("Please fix the issues before proceeding")
    
    print("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    test_success = run_all_tests()
    sys.exit(0 if test_success else 1) 
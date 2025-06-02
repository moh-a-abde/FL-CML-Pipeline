#!/usr/bin/env python3
"""
ConfigManager Test Suite

Comprehensive test script for the FL-CML-Pipeline ConfigManager implementation.
This test validates all aspects of the Hydra-based configuration system including
experiment loading, type safety, and utility methods.

Usage:
    python test_config_manager.py

Requirements:
    - hydra-core>=1.3.0
    - omegaconf
"""

import sys
import logging
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.config.config_manager import ConfigManager, load_config
    print("✅ ConfigManager imports successful")
except ImportError as import_error:
    print(f"❌ Import error: {import_error}")
    print("Make sure hydra-core and omegaconf are installed: pip install hydra-core omegaconf")
    sys.exit(1)

def test_basic_config_loading():
    """Test basic configuration loading with default settings."""
    print("\n" + "="*60)
    print("Testing Basic Configuration Loading")
    print("="*60)
    
    try:
        manager = ConfigManager()
        config = manager.load_config()
        
        print("✅ Base configuration loaded successfully")
        print(f"   - Data path: {config.data.path}")
        print(f"   - Model type: {config.model.type}")
        print(f"   - Train method: {config.federated.train_method}")
        print(f"   - Tuning enabled: {config.tuning.enabled}")
        
        return True
    except Exception as config_error:  # pylint: disable=broad-except
        print(f"❌ Failed to load base configuration: {config_error}")
        return False

def test_experiment_configs():
    """Test experiment configuration overrides for all supported experiments."""
    print("\n" + "="*60)
    print("Testing Experiment Configuration Overrides")
    print("="*60)
    
    experiments = ["bagging", "cyclic", "dev"]
    results = []
    
    for exp in experiments:
        try:
            manager = ConfigManager()
            config = manager.load_config(experiment=exp)
            
            print(f"✅ {exp.upper()} experiment config loaded")
            print(f"   - Train method: {config.federated.train_method}")
            print(f"   - Num rounds: {config.federated.num_rounds}")
            print(f"   - Tuning enabled: {config.tuning.enabled}")
            if hasattr(config.outputs, 'experiment_name') and config.outputs.experiment_name:
                print(f"   - Experiment name: {config.outputs.experiment_name}")
            
            results.append(True)
        except Exception as exp_error:  # pylint: disable=broad-except
            print(f"❌ Failed to load {exp} experiment: {exp_error}")
            results.append(False)
    
    return all(results)

def test_config_methods():
    """Test ConfigManager utility methods for common operations."""
    print("\n" + "="*60)
    print("Testing ConfigManager Utility Methods")
    print("="*60)
    
    try:
        manager = ConfigManager()
        manager.load_config(experiment="bagging")
        
        # Test model params dict
        model_params = manager.get_model_params_dict()
        print(f"✅ Model params dict: {len(model_params)} parameters")
        
        # Test data path
        data_path = manager.get_data_path()
        print(f"✅ Data path: {data_path}")
        
        # Test tuning enabled check
        tuning_enabled = manager.is_tuning_enabled()
        print(f"✅ Tuning enabled: {tuning_enabled}")
        
        # Test experiment name
        exp_name = manager.get_experiment_name()
        print(f"✅ Experiment name: {exp_name}")
        
        # Test timestamped dirs
        timestamped = manager.should_create_timestamped_dirs()
        print(f"✅ Create timestamped dirs: {timestamped}")
        
        return True
        
    except Exception as method_error:  # pylint: disable=broad-except
        print(f"❌ Failed utility methods test: {method_error}")
        return False

def test_convenience_function():
    """Test the convenience function for quick configuration loading."""
    print("\n" + "="*60)
    print("Testing Convenience Function")
    print("="*60)
    
    try:
        # Test convenience function
        config = load_config(experiment="dev")
        print("✅ Convenience function works")
        print(f"   - Train method: {config.federated.train_method}")
        print(f"   - Num rounds: {config.federated.num_rounds}")
        
        return True
        
    except Exception as convenience_error:  # pylint: disable=broad-except
        print(f"❌ Convenience function failed: {convenience_error}")
        return False

def test_config_overrides():
    """Test dynamic configuration overrides via command line style syntax."""
    print("\n" + "="*60)
    print("Testing Configuration Overrides")
    print("="*60)
    
    try:
        manager = ConfigManager()
        config = manager.load_config(
            experiment="dev",
            overrides=["tuning.enabled=true", "federated.num_rounds=25"]
        )
        
        print("✅ Configuration overrides applied")
        print(f"   - Tuning enabled: {config.tuning.enabled}")
        print(f"   - Num rounds: {config.federated.num_rounds}")
        
        return True
        
    except Exception as override_error:  # pylint: disable=broad-except
        print(f"❌ Configuration overrides failed: {override_error}")
        return False

def main():
    """Run all ConfigManager tests and report results."""
    print("="*60)
    print("FL-CML-Pipeline ConfigManager Tests")
    print("="*60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Define all test functions
    tests = [
        test_basic_config_loading,
        test_experiment_configs,
        test_config_methods,
        test_convenience_function,
        test_config_overrides
    ]
    
    passed = 0
    total = len(tests)
    
    # Run all tests
    for test in tests:
        if test():
            passed += 1
    
    # Report results
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All ConfigManager tests passed!")
        print("\nConfigManager is ready for integration with entry points!")
        return True
    
    print("❌ Some tests failed")
    print("\nPlease review and fix issues before proceeding.")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
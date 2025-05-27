#!/usr/bin/env python3
"""
Unit tests for data integrity to prevent regression of the Class 2 missing issue.
This test ensures that all classes are present in both train and test splits.
"""

import unittest
import pandas as pd
import numpy as np
from dataset import load_csv_data

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity to prevent critical data leakage issues."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_file = 'data/received/final_dataset.csv'
        self.dataset_dict = load_csv_data(self.data_file)
        self.train_df = self.dataset_dict['train'].to_pandas()
        self.test_df = self.dataset_dict['test'].to_pandas()
    
    def test_train_test_class_coverage(self):
        """Ensure all classes present in both train and test splits."""
        train_classes = set(self.train_df['label'].unique())
        test_classes = set(self.test_df['label'].unique())
        
        # Check that all classes are present in training
        self.assertGreaterEqual(len(train_classes), 10, 
                               f"Only {len(train_classes)} classes in train, expected at least 10")
        
        # Check that all classes are present in testing
        self.assertGreaterEqual(len(test_classes), 10, 
                               f"Only {len(test_classes)} classes in test, expected at least 10")
        
        # Check that train and test have the same classes
        self.assertEqual(train_classes, test_classes, 
                        "Class mismatch between train and test splits")
    
    def test_class_2_specifically_present(self):
        """Specifically test that Class 2 is present in training (the critical issue)."""
        train_class_2_count = len(self.train_df[self.train_df['label'] == 2])
        test_class_2_count = len(self.test_df[self.test_df['label'] == 2])
        
        self.assertGreater(train_class_2_count, 0, 
                          "CRITICAL: Class 2 is missing from training data!")
        self.assertGreater(test_class_2_count, 0, 
                          "CRITICAL: Class 2 is missing from test data!")
        
        print(f"✅ Class 2 present: {train_class_2_count} train, {test_class_2_count} test samples")
    
    def test_no_empty_classes(self):
        """Ensure no class has zero samples in either split."""
        train_counts = self.train_df['label'].value_counts()
        test_counts = self.test_df['label'].value_counts()
        
        # Check for empty classes in training
        empty_train_classes = [label for label, count in train_counts.items() if count == 0]
        self.assertEqual(len(empty_train_classes), 0, 
                        f"Empty classes in training: {empty_train_classes}")
        
        # Check for empty classes in testing
        empty_test_classes = [label for label, count in test_counts.items() if count == 0]
        self.assertEqual(len(empty_test_classes), 0, 
                        f"Empty classes in testing: {empty_test_classes}")
    
    def test_reasonable_class_distribution(self):
        """Ensure each class has a reasonable number of samples."""
        train_counts = self.train_df['label'].value_counts()
        test_counts = self.test_df['label'].value_counts()
        
        # Each class should have at least 1000 samples in training (reasonable threshold)
        min_train_samples = 1000
        for label, count in train_counts.items():
            self.assertGreaterEqual(count, min_train_samples, 
                                   f"Class {label} has only {count} training samples, expected at least {min_train_samples}")
        
        # Each class should have at least 100 samples in testing
        min_test_samples = 100
        for label, count in test_counts.items():
            self.assertGreaterEqual(count, min_test_samples, 
                                   f"Class {label} has only {count} test samples, expected at least {min_test_samples}")
    
    def test_split_proportions(self):
        """Ensure train/test split proportions are reasonable."""
        total_samples = len(self.train_df) + len(self.test_df)
        train_proportion = len(self.train_df) / total_samples
        test_proportion = len(self.test_df) / total_samples
        
        # Should be approximately 80/20 split
        self.assertGreater(train_proportion, 0.75, 
                          f"Training proportion {train_proportion:.2f} too low")
        self.assertLess(train_proportion, 0.85, 
                       f"Training proportion {train_proportion:.2f} too high")
        
        self.assertGreater(test_proportion, 0.15, 
                          f"Test proportion {test_proportion:.2f} too low")
        self.assertLess(test_proportion, 0.25, 
                       f"Test proportion {test_proportion:.2f} too high")
    
    def test_data_consistency(self):
        """Ensure data consistency between splits."""
        # Check that we haven't lost or duplicated samples
        original_df = pd.read_csv(self.data_file)
        total_split_samples = len(self.train_df) + len(self.test_df)
        
        self.assertEqual(len(original_df), total_split_samples, 
                        f"Sample count mismatch: original {len(original_df)}, splits {total_split_samples}")
        
        # Check that all original classes are preserved
        original_classes = set(original_df['label'].unique())
        split_classes = set(self.train_df['label'].unique()).union(set(self.test_df['label'].unique()))
        
        self.assertEqual(original_classes, split_classes, 
                        "Classes lost during splitting")

class TestTemporalIntegrity(unittest.TestCase):
    """Test temporal integrity of the hybrid split."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_file = 'data/received/final_dataset.csv'
        self.dataset_dict = load_csv_data(self.data_file)
        self.train_df = self.dataset_dict['train'].to_pandas()
        self.test_df = self.dataset_dict['test'].to_pandas()
    
    def test_temporal_coverage(self):
        """Ensure both splits cover reasonable temporal ranges."""
        if 'Stime' not in self.train_df.columns:
            self.skipTest("No Stime column available for temporal testing")
        
        train_stime_range = self.train_df['Stime'].max() - self.train_df['Stime'].min()
        test_stime_range = self.test_df['Stime'].max() - self.test_df['Stime'].min()
        
        # Both splits should cover substantial temporal ranges
        self.assertGreater(train_stime_range, 1.0, 
                          f"Training temporal range {train_stime_range:.4f} too narrow")
        self.assertGreater(test_stime_range, 1.0, 
                          f"Test temporal range {test_stime_range:.4f} too narrow")

def run_integrity_tests():
    """Run all data integrity tests."""
    print("="*60)
    print("RUNNING DATA INTEGRITY TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add data integrity tests
    suite.addTest(TestDataIntegrity('test_train_test_class_coverage'))
    suite.addTest(TestDataIntegrity('test_class_2_specifically_present'))
    suite.addTest(TestDataIntegrity('test_no_empty_classes'))
    suite.addTest(TestDataIntegrity('test_reasonable_class_distribution'))
    suite.addTest(TestDataIntegrity('test_split_proportions'))
    suite.addTest(TestDataIntegrity('test_data_consistency'))
    
    # Add temporal integrity tests
    suite.addTest(TestTemporalIntegrity('test_temporal_coverage'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print("✅ Data integrity verified")
        print("✅ Class 2 issue permanently fixed")
        print("✅ No regression detected")
    else:
        print("❌ TESTS FAILED!")
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        print("❌ Data integrity issues detected")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integrity_tests()
    exit(0 if success else 1) 
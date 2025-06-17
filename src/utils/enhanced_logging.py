"""
Enhanced logging utilities for the Federated Learning Pipeline.

This module provides improved logging capabilities with:
- Better formatting and visual structure
- Timing information and performance metrics
- Progress tracking
- Configuration summaries
- Result summaries
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
from contextlib import contextmanager


class EnhancedLogger:
    """Enhanced logger for the federated learning pipeline."""
    
    def __init__(self, name: str = "fl_pipeline", log_file: Optional[str] = None):
        """Initialize the enhanced logger.
        
        Args:
            name: Logger name (changed from "flwr" to avoid conflicts)
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.start_time = time.time()
        self.step_times = {}
        self.step_counter = 0
        
        # Setup enhanced formatting
        self._setup_logging(log_file)
    
    def _setup_logging(self, log_file: Optional[str] = None):
        """Setup logging configuration with enhanced formatting."""
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def pipeline_start(self, config: Any):
        """Log pipeline start with configuration summary."""
        self.start_time = time.time()
        
        self.logger.info("Starting Federated Learning Pipeline with Enhanced Monitoring")
        self.logger.info("=" * 100)
        self.logger.info("PIPELINE CONFIGURATION SUMMARY")
        self.logger.info("=" * 100)
        
        # Configuration summary
        self.logger.info("Dataset: %s/%s", config.data.path, config.data.filename)
        self.logger.info("Training Method: %s", config.federated.train_method)
        self.logger.info("Federated Rounds: %d", config.federated.num_rounds)
        self.logger.info("Pool Size: %d", config.federated.pool_size)
        self.logger.info("Clients per Round: %d", config.federated.num_clients_per_round)
        self.logger.info("Local Rounds: %d", config.model.num_local_rounds)
        self.logger.info("Hyperparameter Tuning: %s", 'Enabled' if config.tuning.enabled else 'Disabled')
        self.logger.info("Centralized Evaluation: %s", 'Enabled' if config.federated.centralised_eval else 'Disabled')
        self.logger.info("Random Seed: %d", config.data.seed)
        
        # Model parameters summary
        self.logger.info("")
        self.logger.info("MODEL PARAMETERS")
        self.logger.info("-" * 50)
        key_params = ['eta', 'max_depth', 'min_child_weight', 'gamma', 'subsample', 'colsample_bytree']
        for param in key_params:
            if hasattr(config.model.params, param):
                value = getattr(config.model.params, param)
                self.logger.info("   %s: %s", param, value)
        
        self.logger.info("=" * 100)
    
    def step_start(self, step_name: str, description: str, command: Optional[str] = None):
        """Log the start of a pipeline step."""
        self.step_counter += 1
        step_start_time = time.time()
        self.step_times[step_name] = step_start_time
        
        self.logger.info("")
        self.logger.info("STEP %d: %s", self.step_counter, step_name.upper())
        self.logger.info("-" * 100)
        self.logger.info("Description: %s", description)
        if command:
            self.logger.info("Command: %s", command)
        self.logger.info("Started at: %s", datetime.now().strftime('%H:%M:%S'))
        self.logger.info("Running...")
    
    def step_complete(self, step_name: str, success: bool = True):
        """Log the completion of a pipeline step."""
        if step_name in self.step_times:
            duration = time.time() - self.step_times[step_name]
            duration_str = str(timedelta(seconds=int(duration)))
            
            if success:
                self.logger.info("")
                self.logger.info("Step %d (%s) completed successfully!", self.step_counter, step_name)
                self.logger.info("Duration: %s", duration_str)
            else:
                self.logger.error("")
                self.logger.error("Step %d (%s) failed!", self.step_counter, step_name)
                self.logger.error("Duration: %s", duration_str)
            
            self.logger.info("-" * 100)
    
    def log_data_statistics(self, data_stats: Dict[str, Any]):
        """Log dataset statistics."""
        self.logger.info("")
        self.logger.info("DATASET STATISTICS")
        self.logger.info("-" * 50)
        
        # Basic statistics
        if 'total_samples' in data_stats:
            self.logger.info("Total samples: %d", data_stats['total_samples'])
        if 'features' in data_stats:
            self.logger.info("Features: %s", data_stats['features'])
        
        # Split information
        if 'train_samples' in data_stats:
            self.logger.info("Train samples: %d", data_stats['train_samples'])
        if 'test_samples' in data_stats:
            self.logger.info("Test samples: %d", data_stats['test_samples'])
        
        # Class distribution
        if 'class_distribution' in data_stats:
            try:
                self.logger.info("")
                self.logger.info("Class Distribution:")
                for class_id, counts in data_stats['class_distribution'].items():
                    train_count = int(counts.get('train', 0))
                    test_count = int(counts.get('test', 0))
                    total_count = int(counts.get('total', train_count + test_count))
                    self.logger.info("   Class %s: %s train, %s test, %s total", 
                                   class_id, f"{train_count:,}", f"{test_count:,}", f"{total_count:,}")
            except (ValueError, TypeError) as e:
                self.logger.warning("Could not format class_distribution: %s", e)
        
        self.logger.info("-" * 50)
    
    def log_federated_progress(self, round_num: int, total_rounds: int, metrics: Optional[Dict[str, float]] = None):
        """Log federated learning round progress."""
        progress_pct = (round_num / total_rounds) * 100
        progress_bar = "█" * int(progress_pct // 5) + "░" * (20 - int(progress_pct // 5))
        
        self.logger.info("Round %d/%d [%s] %.1f%%", round_num, total_rounds, progress_bar, progress_pct)
        
        if metrics:
            # Log detailed metrics
            self.logger.info("Performance Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info("   %s: %.6f", metric, value)
                else:
                    self.logger.info("   %s: %s", metric, value)
    
    def pipeline_complete(self, results_dir: str, final_metrics: Optional[Dict[str, Any]] = None):
        """Log pipeline completion with summary."""
        total_duration = time.time() - self.start_time
        duration_str = str(timedelta(seconds=int(total_duration)))
        
        self.logger.info("")
        self.logger.info("FEDERATED LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 100)
        self.logger.info("Total Execution Time: %s", duration_str)
        self.logger.info("Results Directory: %s", results_dir)
        
        if final_metrics:
            self.logger.info("")
            self.logger.info("FINAL RESULTS")
            self.logger.info("-" * 50)
            for metric, value in final_metrics.items():
                if isinstance(value, float):
                    self.logger.info("   %s: %.6f", metric, value)
                else:
                    self.logger.info("   %s: %s", metric, value)
        
        self.logger.info("")
        self.logger.info("KEY IMPROVEMENTS ACHIEVED:")
        self.logger.info("   - Consistent preprocessing across all phases")
        self.logger.info("   - Temporal splitting to prevent data leakage")
        self.logger.info("   - Global feature processor for uniform data representation")
        self.logger.info("   - Distributed learning with privacy preservation")
        self.logger.info("   - Robust evaluation and monitoring")
        
        # Step timing summary
        if self.step_times:
            self.logger.info("")
            self.logger.info("STEP TIMING SUMMARY")
            self.logger.info("-" * 50)
            for step_name, start_time in self.step_times.items():
                duration = time.time() - start_time
                self.logger.info("   %s: %s", step_name, str(timedelta(seconds=int(duration))))
        
        self.logger.info("=" * 100)
        self.logger.info("Pipeline execution completed at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# Global enhanced logger instance
_enhanced_logger = None

def get_enhanced_logger(log_file: Optional[str] = None) -> EnhancedLogger:
    """Get the global enhanced logger instance."""
    # pylint: disable=global-statement
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedLogger(log_file=log_file)
    return _enhanced_logger

def setup_enhanced_logging(log_file: Optional[str] = None) -> EnhancedLogger:
    """Setup enhanced logging for the pipeline."""
    # pylint: disable=global-statement
    global _enhanced_logger
    # Reset the global logger to ensure clean setup
    _enhanced_logger = None
    return get_enhanced_logger(log_file) 
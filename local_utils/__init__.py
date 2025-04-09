"""
local_utils package for extending the FL-CML-Pipeline functionality
without modifying the core codebase.
"""

from local_utils.smote_processor import apply_smote_to_benign, apply_smote_wrapper

__all__ = ['apply_smote_to_benign', 'apply_smote_wrapper'] 
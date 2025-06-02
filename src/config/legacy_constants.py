"""
Legacy constants for the FL-CML-Pipeline.

Note: This file previously contained hardcoded constants and argument parsers
that have been migrated to the ConfigManager system. It is kept for backward
compatibility but should not be used in new code.

For configuration management, use:
    from src.config.config_manager import ConfigManager
    config_manager = ConfigManager()
    model_params = config_manager.get_model_params_dict()
"""

# This file has been cleaned up as part of Step 4: Legacy Code Cleanup
# All constants and argument parsers have been migrated to ConfigManager
# See configs/base.yaml for the current configuration structure

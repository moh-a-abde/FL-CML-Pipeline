# FL-CML-Pipeline Refactoring Quick Start Guide

## Immediate Actions for Refactoring Agent

### 🚀 Start Here

1. **Read the main plan**: Review `ARCHITECT_REFACTORING_PLAN.md` for the complete refactoring strategy
2. **Create progress file**: Start with `progress/phase1_structure.md` to track your work
3. **Begin Phase 1**: Focus on restructuring the project directories first

### 📋 Key Priorities

1. **Fix Critical Issues First**
   - Configuration management chaos
   - Code duplication (especially DMatrix creation)
   - Global state in server_utils.py
   - File organization mess

2. **Preserve Functionality**
   - All fixes for Class 2 data leakage must be maintained
   - Hyperparameter tuning improvements must be kept
   - Model performance should not regress

3. **Test Continuously**
   - After each file move, verify imports work
   - Run existing tests to ensure no breakage
   - Document any failing tests for later fix

### 🛠️ Phase 1 Checklist (Start Here)

```bash
# 1. Create new directory structure
mkdir -p src/{core,federated,tuning,models,config,utils}
mkdir -p src/federated/strategies
mkdir -p tests/{unit,integration,fixtures}
mkdir -p scripts
mkdir -p configs/{experiment,hydra}
mkdir -p docs
mkdir -p archive/{fixes,old_implementations}

# 2. Create __init__.py files
touch src/__init__.py
touch src/core/__init__.py
touch src/federated/__init__.py
touch src/federated/strategies/__init__.py
touch src/tuning/__init__.py
touch src/models/__init__.py
touch src/config/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# 3. Start moving files (example commands)
mv dataset.py src/core/
mv client.py client_utils.py src/federated/
mv server.py server_utils.py src/federated/
mv ray_tune_xgboost_updated.py src/tuning/ray_tune_xgboost.py
mv ray_tune_xgboost.py archive/old_implementations/
mv visualization_utils.py src/utils/visualization.py
```

### 📁 File Movement Map

| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| dataset.py | src/core/dataset.py | Split FeatureProcessor into separate file |
| client.py | src/federated/client.py | Keep imports to client_utils |
| client_utils.py | src/federated/client_utils.py | Will be refactored later |
| server.py | src/federated/server.py | Remove monkey patching |
| server_utils.py | src/federated/utils.py | Extract strategies to separate files |
| ray_tune_xgboost_updated.py | src/tuning/ray_tune_xgboost.py | Primary version |
| ray_tune_xgboost.py | archive/old_implementations/ | Old version |
| utils.py | src/config/legacy_constants.py | Will be replaced by Hydra |
| visualization_utils.py | src/utils/visualization.py | Clean up imports |
| All test_*.py files | tests/unit/ or tests/integration/ | Organize by type |
| All *.sh scripts | scripts/ | Update paths inside |
| All fix summaries | archive/fixes/ | Keep for reference |

### ⚠️ Critical Warnings

1. **DO NOT DELETE** the FeatureProcessor logic - it fixes critical data leakage
2. **DO NOT CHANGE** the hybrid temporal-stratified split in dataset.py
3. **DO NOT MODIFY** the expanded hyperparameter search space
4. **PRESERVE** all Class 2 fixes that were implemented

### 🔄 Import Update Pattern

When moving files, update imports following this pattern:

```python
# Old import
from dataset import load_csv_data, FeatureProcessor
from client_utils import XgbClient

# New import
from src.core.dataset import load_csv_data
from src.core.feature_processor import FeatureProcessor
from src.federated.client_utils import XgbClient
```

### 📊 Progress Tracking

After each major step:
1. Update `progress/phase1_structure.md`
2. Commit with descriptive message: `refactor: move dataset.py to src/core/`
3. Test that imports still work
4. Document any issues encountered

### 🎯 Success Criteria for Phase 1

- [ ] All Python files moved to appropriate src/ subdirectories
- [ ] All scripts moved to scripts/
- [ ] All tests moved to tests/
- [ ] All fix summaries archived
- [ ] All imports updated and working
- [ ] run.py still executes without errors
- [ ] Basic smoke test passes

### 💡 Tips

1. **Use git mv**: Preserves file history
   ```bash
   git mv dataset.py src/core/dataset.py
   ```

2. **Test imports incrementally**: Don't move everything at once
   ```python
   python -c "from src.core.dataset import load_csv_data"
   ```

3. **Keep a rollback branch**: Before starting
   ```bash
   git checkout -b refactoring-backup
   git checkout main
   git checkout -b phase1-restructure
   ```

4. **Update one module at a time**: Start with standalone modules like utils

### 🚧 Known Issues to Fix

1. **Circular imports**: Watch for these when reorganizing
2. **Hardcoded paths**: Update any absolute paths in scripts
3. **Missing dependencies**: Some imports might be missing in requirements.txt
4. **Test paths**: Tests might have hardcoded paths to data files

### 📞 When to Stop and Ask

Stop and document if you encounter:
- Circular import dependencies that can't be easily resolved
- Tests that fail after moving files (beyond simple import fixes)
- Functionality that breaks and the cause isn't obvious
- Design decisions that could go multiple ways

### 🎉 Quick Wins

Start with these easy moves that shouldn't break anything:
1. Move all `*.sh` files to `scripts/`
2. Move all `test_*.py` files to `tests/`
3. Archive all `*_SUMMARY.md` and `*_ANALYSIS.md` files
4. Move `visualization_utils.py` to `src/utils/`

Good luck! Remember to commit frequently and document everything in the progress files. 
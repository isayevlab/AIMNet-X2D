# Phase 1: Foundation Modernization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modernize AIMNet-X2D codebase by completing print→logging conversion, updating typing imports to Python 3.12 syntax, and adding return type hints.

**Architecture:** Three parallel workstreams that can be executed sequentially within each file to minimize merge conflicts. Each file is touched once, applying all three modernizations together.

**Tech Stack:** Python 3.12, logging framework (src/utils/logging.py), built-in generics (list, dict, tuple), union syntax (X | None)

---

## Summary Statistics

| Category | Count | Files |
|----------|-------|-------|
| Print statements to convert | 392 | 19 files |
| Typing imports to modernize | 33 | 33 files |
| Return type hints to add | ~168 | ~24 files |

## Execution Strategy

**Batch by module** to keep related changes together:
1. Batch A: `src/main/` (4 files, 222 prints) - Highest impact
2. Batch B: `src/inference/` (4 files, 96 prints)
3. Batch C: `src/config/` (4 files, 29 prints)
4. Batch D: `src/training/` (3 files, 20 prints)
5. Batch E: `src/datasets/` (4 files, 5 prints)
6. Batch F: `src/models/` (4 files, 7 prints)
7. Batch G: `src/utils/` + `src/data/` (5 files, 4 prints)
8. Batch H: Root `src/` (1 file, 9 prints)

---

## Batch A: src/main/ Module (222 print statements)

### Task A1: Modernize src/main/runner.py (90 prints)

**Files:**
- Modify: `src/main/runner.py`

**Step 1: Update typing imports**

Change line ~11:
```python
# Before
from typing import Dict, Any, Optional, Tuple

# After
from typing import Any
```

Then update all type annotations in the file:
- `Dict[str, Any]` → `dict[str, Any]`
- `Optional[X]` → `X | None`
- `Tuple[X, Y]` → `tuple[X, Y]`
- `List[X]` → `list[X]`

**Step 2: Add logger import**

Add after other imports:
```python
from utils.logging import get_logger

logger = get_logger(__name__)
```

**Step 3: Convert print statements to logging**

Pattern replacements:
- `print(f"...")` → `logger.info(f"...")`
- `print(f"WARNING: ...")` → `logger.warning(f"...")`
- `print(f"ERROR: ...")` → `logger.error(f"...")`
- Debug/verbose prints → `logger.debug(f"...")`

**Step 4: Add return type hints**

Add `-> ReturnType` to all functions missing them.

**Step 5: Run tests**

```bash
python -m pytest tests/ -q --tb=short
```
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/main/runner.py
git commit -m "Modernize src/main/runner.py: logging, typing, return hints"
```

---

### Task A2: Modernize src/main/utils.py (52 prints)

**Files:**
- Modify: `src/main/utils.py`

**Step 1: Update typing imports**

Change line ~13:
```python
# Before
from typing import Dict, Any, Optional, Tuple

# After
from typing import Any
```

**Step 2: Add logger import**

```python
from utils.logging import get_logger

logger = get_logger(__name__)
```

**Step 3: Convert print statements to logging**

Same pattern as Task A1.

**Step 4: Add return type hints**

**Step 5: Run tests**

```bash
python -m pytest tests/ -q --tb=short
```

**Step 6: Commit**

```bash
git add src/main/utils.py
git commit -m "Modernize src/main/utils.py: logging, typing, return hints"
```

---

### Task A3: Modernize src/main/cli.py (42 prints)

**Files:**
- Modify: `src/main/cli.py`

**Step 1: Update typing imports**

```python
# Before
from typing import Optional

# After (remove entirely, use X | None inline)
```

**Step 2: Add logger import**

```python
from utils.logging import get_logger

logger = get_logger(__name__)
```

**Step 3: Convert print statements to logging**

**Step 4: Add return type hints**

**Step 5: Run tests and commit**

```bash
python -m pytest tests/ -q --tb=short
git add src/main/cli.py
git commit -m "Modernize src/main/cli.py: logging, typing, return hints"
```

---

### Task A4: Modernize src/main/hyperopt.py (38 prints)

**Files:**
- Modify: `src/main/hyperopt.py`

**Step 1: Update typing imports**

```python
# Before
from typing import Dict, Any, Optional

# After
from typing import Any
```

**Step 2: Add logger and convert prints**

**Step 3: Add return type hints**

**Step 4: Run tests and commit**

```bash
python -m pytest tests/ -q --tb=short
git add src/main/hyperopt.py
git commit -m "Modernize src/main/hyperopt.py: logging, typing, return hints"
```

---

## Batch B: src/inference/ Module (96 print statements)

### Task B1: Modernize src/inference/pipeline.py (79 prints)

**Files:**
- Modify: `src/inference/pipeline.py`

**Step 1: Update typing imports**

```python
# Before
from typing import Optional, Tuple, Dict, Any

# After
from typing import Any
```

**Step 2: Add logger import**

```python
from utils.logging import get_logger

logger = get_logger(__name__)
```

**Step 3: Convert 79 print statements to logging**

**Step 4: Add return type hints to all methods**

**Step 5: Run tests and commit**

```bash
python -m pytest tests/ -q --tb=short
git add src/inference/pipeline.py
git commit -m "Modernize src/inference/pipeline.py: logging, typing, return hints"
```

---

### Task B2: Modernize src/inference/embeddings.py (11 prints)

**Files:**
- Modify: `src/inference/embeddings.py`

**Steps:** Same pattern - typing → logger → prints → return hints → test → commit

```bash
git commit -m "Modernize src/inference/embeddings.py: logging, typing, return hints"
```

---

### Task B3: Modernize src/inference/preprocessing.py (6 prints)

**Files:**
- Modify: `src/inference/preprocessing.py`

**Steps:** Same pattern

```bash
git commit -m "Modernize src/inference/preprocessing.py: logging, typing, return hints"
```

---

### Task B4: Modernize remaining inference files (0 prints but typing needed)

**Files:**
- Modify: `src/inference/config.py`
- Modify: `src/inference/uncertainty.py`
- Modify: `src/inference/engine.py` (already has logger)

**Steps:** Update typing imports only, verify logger already present in engine.py

```bash
git add src/inference/config.py src/inference/uncertainty.py src/inference/engine.py
git commit -m "Modernize src/inference typing imports"
```

---

## Batch C: src/config/ Module (29 print statements)

### Task C1: Modernize src/config/args.py (14 prints)

**Files:**
- Modify: `src/config/args.py`

**Steps:** Same pattern

```bash
git commit -m "Modernize src/config/args.py: logging, typing, return hints"
```

---

### Task C2: Modernize src/config/paths.py (10 prints)

**Files:**
- Modify: `src/config/paths.py`

```bash
git commit -m "Modernize src/config/paths.py: logging, typing, return hints"
```

---

### Task C3: Modernize src/config/experiment.py (3 prints)

**Files:**
- Modify: `src/config/experiment.py`

```bash
git commit -m "Modernize src/config/experiment.py: logging, typing, return hints"
```

---

### Task C4: Modernize src/config/validation.py (2 prints)

**Files:**
- Modify: `src/config/validation.py`

```bash
git commit -m "Modernize src/config/validation.py: logging, typing, return hints"
```

---

## Batch D: src/training/ Module (20 print statements)

### Task D1: Modernize src/training/extractors.py (13 prints)

**Files:**
- Modify: `src/training/extractors.py`

```bash
git commit -m "Modernize src/training/extractors.py: logging, typing, return hints"
```

---

### Task D2: Modernize src/training/evaluator.py (4 prints)

**Files:**
- Modify: `src/training/evaluator.py`

```bash
git commit -m "Modernize src/training/evaluator.py: logging, typing, return hints"
```

---

### Task D3: Modernize src/training/predictor.py (3 prints)

**Files:**
- Modify: `src/training/predictor.py`

```bash
git commit -m "Modernize src/training/predictor.py: logging, typing, return hints"
```

---

### Task D4: Modernize src/training/trainer.py (already has logger, just typing)

**Files:**
- Modify: `src/training/trainer.py`

**Note:** This file already uses logging. Only update typing imports.

```bash
git commit -m "Modernize src/training/trainer.py typing imports"
```

---

## Batch E: src/datasets/ Module (5 print statements)

### Task E1: Modernize src/datasets/loaders.py (4 prints)

**Files:**
- Modify: `src/datasets/loaders.py`

```bash
git commit -m "Modernize src/datasets/loaders.py: logging, typing, return hints"
```

---

### Task E2: Modernize src/datasets/molecular.py (1 print)

**Files:**
- Modify: `src/datasets/molecular.py`

```bash
git commit -m "Modernize src/datasets/molecular.py: logging, typing, return hints"
```

---

### Task E3: Modernize remaining datasets files (typing only)

**Files:**
- Modify: `src/datasets/features.py` (already has logger)
- Modify: `src/datasets/io.py`
- Modify: `src/datasets/utils.py`

```bash
git add src/datasets/features.py src/datasets/io.py src/datasets/utils.py
git commit -m "Modernize src/datasets typing imports"
```

---

## Batch F: src/models/ Module (7 print statements)

### Task F1: Modernize src/models/gnn.py (7 prints)

**Files:**
- Modify: `src/models/gnn.py`

```bash
git commit -m "Modernize src/models/gnn.py: logging, typing, return hints"
```

---

### Task F2: Modernize remaining models files (typing only)

**Files:**
- Modify: `src/models/layers.py`
- Modify: `src/models/pooling.py`
- Modify: `src/models/losses.py`
- Modify: `src/models/normalizers.py`

```bash
git add src/models/layers.py src/models/pooling.py src/models/losses.py src/models/normalizers.py
git commit -m "Modernize src/models typing imports"
```

---

## Batch G: src/utils/ + src/data/ (4 print statements)

### Task G1: Modernize src/utils/optimization.py (4 prints)

**Files:**
- Modify: `src/utils/optimization.py`

```bash
git commit -m "Modernize src/utils/optimization.py: logging, typing, return hints"
```

---

### Task G2: Modernize remaining utils files (typing only)

**Files:**
- Modify: `src/utils/distributed.py`
- Modify: `src/utils/activation.py`
- Modify: `src/utils/logging.py`

```bash
git add src/utils/distributed.py src/utils/activation.py src/utils/logging.py
git commit -m "Modernize src/utils typing imports"
```

---

### Task G3: Modernize src/data/preprocessing.py (already has logger, just typing)

**Files:**
- Modify: `src/data/preprocessing.py`

```bash
git commit -m "Modernize src/data/preprocessing.py typing imports"
```

---

## Batch H: Root src/ (9 print statements)

### Task H1: Modernize src/trial_utils.py (9 prints)

**Files:**
- Modify: `src/trial_utils.py`

```bash
git commit -m "Modernize src/trial_utils.py: logging, typing, return hints"
```

---

## Final Verification

### Task Z1: Run full test suite

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All 60+ tests pass

### Task Z2: Verify no remaining issues

```bash
# Check for remaining print statements (should be 0 or only in __main__ blocks)
grep -rn "print(" src/ --include="*.py" | grep -v "if __name__" | wc -l

# Check for deprecated typing imports (should be 0)
grep -rn "from typing import.*List\|from typing import.*Dict\|from typing import.*Tuple\|from typing import.*Optional" src/ --include="*.py" | wc -l
```

### Task Z3: Final commit summary

```bash
git log --oneline -20
```

---

## Quick Reference: Typing Modernization

| Before (deprecated) | After (Python 3.12) |
|---------------------|---------------------|
| `from typing import List` | Use `list` directly |
| `from typing import Dict` | Use `dict` directly |
| `from typing import Tuple` | Use `tuple` directly |
| `from typing import Optional` | Use `X \| None` |
| `from typing import Union` | Use `X \| Y` |
| `List[str]` | `list[str]` |
| `Dict[str, Any]` | `dict[str, Any]` |
| `Tuple[int, str]` | `tuple[int, str]` |
| `Optional[int]` | `int \| None` |
| `Union[int, str]` | `int \| str` |

**Keep importing:** `Any`, `Callable`, `TypeVar`, `Generic`, `Protocol`

---

## Quick Reference: Logging Levels

| Level | Use Case |
|-------|----------|
| `logger.debug()` | Verbose details, loop iterations, internal state |
| `logger.info()` | Normal operations, progress, configuration |
| `logger.warning()` | Recoverable issues, deprecations, fallbacks |
| `logger.error()` | Failures that don't crash, handled exceptions |

---

## Estimated Scope

| Batch | Files | Prints | Est. Time |
|-------|-------|--------|-----------|
| A: main/ | 4 | 222 | 40 min |
| B: inference/ | 6 | 96 | 25 min |
| C: config/ | 4 | 29 | 15 min |
| D: training/ | 4 | 20 | 15 min |
| E: datasets/ | 5 | 5 | 10 min |
| F: models/ | 5 | 7 | 10 min |
| G: utils+data/ | 4 | 4 | 10 min |
| H: root | 1 | 9 | 5 min |
| **Total** | **33** | **392** | **~2 hours** |

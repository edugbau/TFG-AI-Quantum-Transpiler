# MO Tuning Ref Point and Live Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a session-wide hypervolume reference point for MO tuning, add a manual reference-point mode that skips warm-up, and surface live tuning state in the GUI.

**Architecture:** Keep tuning logic centralized in `src/mo_module/tuning.py`, expose structured progress events from the tuner, and make the GUI a thin consumer of those events. The GUI must let the user choose calibrated or manual reference-point mode and show the calibrated/manual reference point explicitly while tuning runs.

**Tech Stack:** Python 3.10, Optuna, numpy, customtkinter, pytest.

---

### Task 1: Tuning Core and Validation

**Files:**
- Modify: `.worktrees/mo-tuning-status-refpoint/src/mo_module/tuning.py`
- Test: `.worktrees/mo-tuning-status-refpoint/tests/test_mo_module/test_tuning.py`

- [ ] Add failing tests for fixed reference-point scoring, manual reference-point configuration, calibrated warm-up derivation, progress callbacks, and invalid budgets/ranges.
- [ ] Run `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_mo_module/test_tuning.py -q` in `.worktrees/mo-tuning-status-refpoint` and verify the new tests fail for the intended reasons.
- [ ] Implement the minimal tuning changes in `src/mo_module/tuning.py`:
  - explicit reference-point mode (`calibrated` or `manual`);
  - fixed session reference point storage;
  - warm-up calibration helper;
  - manual reference-point validation;
  - structured progress callback events;
  - stricter validation for budgets and search-space ranges.
- [ ] Re-run `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_mo_module/test_tuning.py -q` in `.worktrees/mo-tuning-status-refpoint` and verify all tuning tests pass.

### Task 2: GUI Live Status and Manual Reference Point Controls

**Files:**
- Modify: `.worktrees/mo-tuning-status-refpoint/src/mo_module/benchmark/benchmark_gui.py`
- Test: `.worktrees/mo-tuning-status-refpoint/tests/test_mo_module/test_tuning.py`

- [ ] Add failing tests for any new GUI-adjacent parsing or formatting helpers that can be tested headlessly.
- [ ] Run `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_mo_module/test_tuning.py -q` in `.worktrees/mo-tuning-status-refpoint` and verify the new test fails first.
- [ ] Implement the GUI changes in `src/mo_module/benchmark/benchmark_gui.py`:
  - reference-point mode selector;
  - warm-up control for calibrated mode;
  - manual reference-point entry for manual mode;
  - tuning progress bar and detailed status labels;
  - event-driven updates for current trial, best-so-far score, and explicit calibrated/manual reference point.
- [ ] Re-run `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_mo_module/test_tuning.py -q` in `.worktrees/mo-tuning-status-refpoint` and verify tests remain green.

### Task 3: Documentation and Full Verification

**Files:**
- Modify: `.worktrees/mo-tuning-status-refpoint/src/mo_module/docs/tuning.md`
- Modify: `.worktrees/mo-tuning-status-refpoint/src/mo_module/docs/internal_documentation.md`

- [ ] Update tuning documentation to describe calibrated and manual reference-point modes, warm-up semantics, explicit reference-point reporting, and the live GUI status model.
- [ ] Run `C:\Users\Eduardo\Desktop\universidad\TFG-Quantum-Transpiler\.venv\Scripts\python.exe -m pytest tests/test_mo_module -q` in `.worktrees/mo-tuning-status-refpoint` and verify the full MO suite passes.
- [ ] Run `git status --short` in `.worktrees/mo-tuning-status-refpoint` and inspect that only the intended files changed.

---
name: mo-optimization
description: Use when modifying src/mo_module fitness, encoding, operators, Pareto analysis, or tuning workflows for quantum layout optimization.
compatibility: opencode
metadata:
  module: mo_module
  scope: project
---

# MO Optimization

## Overview

Guide changes inside `src/mo_module/` so layout optimization stays focused on the project's multi-objective contract: valid layout encodings, deterministic evolutionary search, and Pareto analysis over the active objectives.

## When to Use

- Editing `src/mo_module/` problem definitions, fitness evaluation, or operators.
- Adjusting Pareto metrics, Hypervolume analysis, or selection helpers.
- Touching Optuna tuning logic for MO search configuration.

## Quick Reference

- Keep the layout representation valid for the backend and circuit width.
- Treat `depth` and `cnot_equivalent` as the active optimization contract.
- Use `pymoo` primitives for problem modeling and search execution.
- Keep MO output consumable by `src/integration/`, not directly by `src/rl_module/`.

## Core Pattern

- Encode layouts as stable logical-to-physical assignments.
- Evaluate candidates through the shared transpilation-backed cost path.
- Extract and analyze non-dominated solutions with Pareto-aware helpers.
- Tune search parameters without changing module boundaries.

## Common Mistakes

- Treating `two_qubit_gates` as the optimization contract instead of `cnot_equivalent`.
- Letting `mo_module` own MO -> RL orchestration instead of `src/integration/`.
- Adding tuning knobs before the base encoding or operator semantics are stable.

## Project References

- `src/mo_module/docs/internal_documentation.md`
- `src/mo_module/docs/tuning.md`
- `src/mo_module/optimizer.py`
- `src/mo_module/pareto.py`
